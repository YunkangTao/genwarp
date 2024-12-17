# Load models.

import sys

sys.path.append('./extern/Depth-Anything-V2/metric_depth')

import cv2
import torch
import torchvision

torchvision.disable_beta_transforms_warning()
import numpy as np
import torch.nn.functional as F
from depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from extern.ZoeDepth.zoedepth.utils.misc import colorize
from genwarp import GenWarp
from genwarp.ops import (
    camera_lookat,
    focal_length_to_fov,
    get_projection_matrix,
    sph2cart,
)


# Crop the image to the shorter side.
def crop(img: Image) -> Image:
    W, H = img.size
    if W < H:
        left, right = 0, W
        top, bottom = np.ceil((H - W) / 2.0), np.floor((H - W) / 2.0) + W
    else:
        left, right = np.ceil((W - H) / 2.0), np.floor((W - H) / 2.0) + H
        top, bottom = 0, H
    return img.crop((left, top, right, bottom))


def prepare_models(dav2_outdoor, dav2_model):
    dav2_model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    # Depth Anything V2
    dav2_model_config = {
        **dav2_model_configs[dav2_model],
        # 20 for indoor model, 80 for outdoor model
        'max_depth': 80 if dav2_outdoor else 20,
    }
    depth_anything = DepthAnythingV2(**dav2_model_config)

    # Change the path to the
    dav2_model_fn = f'depth_anything_v2_metric_{"vkitti" if dav2_outdoor else "hypersim"}_{dav2_model}.pth'
    depth_anything.load_state_dict(torch.load(f'./checkpoints_dav2/{dav2_model_fn}', map_location='cpu'))
    depth_anything = depth_anything.to('cuda').eval()

    # GenWarp
    genwarp_cfg = dict(pretrained_model_path='./checkpoints', checkpoint_name='multi1', half_precision_weights=True)
    genwarp_nvs = GenWarp(cfg=genwarp_cfg)

    return depth_anything, genwarp_nvs


def prepare_frames(video_file):
    # 创建视频捕捉对象
    cap = cv2.VideoCapture(video_file)

    # 获取视频的宽度和高度
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 检查视频是否成功打开
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_file}")

    frames = []

    while True:
        # 逐帧读取
        ret, frame = cap.read()

        # 如果没有读取到帧，则退出循环
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    # 释放视频捕捉对象
    cap.release()

    return frames, width, height


def prepare_camera_poses(camera_pose_file):
    whole_camera_para = []

    with open(camera_pose_file, 'r', encoding='utf-8') as file:
        # 读取所有行
        lines = file.readlines()

        # 确保文件至少有两行
        if len(lines) < 2:
            print("文件内容不足两行，无法读取数据。")
            return whole_camera_para

        # 跳过第一行，从第二行开始处理
        for idx, line in enumerate(lines[1:], start=2):
            # 去除首尾空白字符并按空格分割
            parts = line.strip().split()

            # 检查每行是否有19个数字
            if len(parts) != 19:
                print(f"警告：第 {idx} 行的数字数量不是19，跳过该行。")
                continue

            try:
                # 将字符串转换为浮点数
                numbers = [float(part) for part in parts]
                whole_camera_para.append(numbers)
            except ValueError as ve:
                print(f"警告：第 {idx} 行包含非数字字符，跳过该行。错误详情: {ve}")
                continue

    return whole_camera_para


def get_src_proj_mtx(focal_length_x_norm, focal_length_y_norm, height, width, res, src_image):
    """
    根据相机内参和图像处理步骤计算投影矩阵。

    参数:
    - focal_length_x_norm (float): 归一化的x方向焦距 (fx / width)
    - focal_length_y_norm (float): 归一化的y方向焦距 (fy / height)
    - height (int): 原始图像高度
    - width (int): 原始图像宽度
    - res (int): 图像缩放后的尺寸 (res, res)
    - src_image (torch.Tensor): 源图像张量，用于确定设备类型

    返回:
    - src_proj_mtx (torch.Tensor): 投影矩阵，形状为 (1, 4, 4)
    """
    # 将归一化焦距转换为像素单位
    focal_length_x = focal_length_x_norm * width
    focal_length_y = focal_length_y_norm * height

    # 裁剪为中心正方形
    cropped_size = min(width, height)
    scale_crop_x = cropped_size / width
    scale_crop_y = cropped_size / height

    # 调整焦距以适应裁剪后的图像
    focal_length_x_cropped = focal_length_x * scale_crop_x
    focal_length_y_cropped = focal_length_y * scale_crop_y

    # 缩放图像
    scale_resize = res / cropped_size
    focal_length_x_resized = focal_length_x_cropped * scale_resize
    focal_length_y_resized = focal_length_y_cropped * scale_resize

    # 计算垂直视场角 (fovy) 使用调整后的焦距和缩放后的高度
    fovy = 2.0 * torch.atan(torch.tensor(res / (2.0 * focal_length_y_resized)))
    fovy = fovy.unsqueeze(0)  # 形状调整为 (1,)

    near, far = 0.01, 100.0
    aspect_wh = 1.0  # 因为图像被缩放为正方形 (res, res)

    # 获取投影矩阵
    src_proj_mtx = get_projection_matrix(fovy=fovy, aspect_wh=aspect_wh, near=near, far=far).to(src_image)

    return src_proj_mtx


def get_rel_view_mtx(src_wc, tar_wc, src_depth, src_image):
    """
    计算源相机到目标相机的相对视图矩阵，同时利用源深度信息。

    参数:
    - src_wc (torch.Tensor): 源相机的 world2camera 外参矩阵，形状为 (3, 4)。
    - tar_wc (torch.Tensor): 目标相机的 world2camera 外参矩阵，形状为 (3, 4)。
    - src_depth (torch.Tensor): 源图像的深度信息，形状为 (N, 1, H, W)。
    - src_image (torch.Tensor): 源图像，用于确定设备和数据类型。

    返回:
    - rel_view_mtx (torch.Tensor): 相对视图矩阵，形状为 (4, 4)。
    """

    # 确保外参矩阵是浮点类型
    src_wc = src_wc.float()
    tar_wc = tar_wc.float()

    # 将 3x4 外参矩阵扩展为 4x4 矩阵
    def to_4x4(m):
        return torch.cat([m, torch.tensor([[0, 0, 0, 1]], device=m.device, dtype=m.dtype)], dim=0)

    src_wc_4 = to_4x4(src_wc)  # 源相机的 world2camera，形状 (4, 4)
    tar_wc_4 = to_4x4(tar_wc)  # 目标相机的 world2camera，形状 (4, 4)

    # 计算源相机的 camera2world 矩阵（即 world2camera 的逆）
    src_cam_to_world = torch.inverse(src_wc_4)

    # 计算相对视图矩阵：目标 world2camera * 源 camera2world
    rel_view_mtx = tar_wc_4 @ src_cam_to_world  # 结果形状为 (4, 4)

    # 根据 src_image 的设备和数据类型调整输出矩阵
    return rel_view_mtx.to(src_image)


def process_one_frame(
    src_frame,
    src_camera_pose,
    tar_frame,
    tar_camera_pose,
    width,
    height,
    focal_length_x,
    focal_length_y,
    principal_point_x,
    principal_point_y,
    res,
    depth_anything,
    genwarp_nvs,
):
    # Load an image.
    # src_image = np.asarray(crop(Image.open(image_file).convert('RGB')).resize((res, res)))
    tar_image = np.asarray(crop(Image.fromarray(tar_frame)).resize((res, res)))
    src_image = np.asarray(crop(Image.fromarray(src_frame)).resize((res, res)))

    # Estimate the depth.
    src_depth = depth_anything.infer_image(src_image[..., ::-1].copy())

    # Go half precision.
    tar_image = torch.from_numpy(tar_image / 255.0).permute(2, 0, 1)[None].cuda().half()
    src_image = torch.from_numpy(src_image / 255.0).permute(2, 0, 1)[None].cuda().half()
    src_depth = torch.from_numpy(src_depth)[None, None].cuda().half()

    # Projection matrix.
    src_proj_mtx = get_src_proj_mtx(focal_length_x, focal_length_y, height, width, res, src_image)
    ## Use the same projection matrix for the source and the target.
    tar_proj_mtx = src_proj_mtx

    src_wc = torch.tensor(src_camera_pose[7:]).reshape((3, 4))
    tar_wc = torch.tensor(tar_camera_pose[7:]).reshape((3, 4))

    rel_view_mtx = get_rel_view_mtx(src_wc, tar_wc, src_depth, src_image)

    # GenWarp.
    renders = genwarp_nvs(src_image=src_image, src_depth=src_depth, rel_view_mtx=rel_view_mtx, src_proj_mtx=src_proj_mtx, tar_proj_mtx=tar_proj_mtx)

    warped = renders['warped']
    synthesized = renders['synthesized']

    # To pil image.
    src_pil = to_pil_image(src_image[0])
    tar_pil = to_pil_image(tar_image[0])
    depth_pil = to_pil_image(colorize(src_depth[0].float()))
    warped_pil = to_pil_image(warped[0])
    synthesized_pil = to_pil_image(synthesized[0])

    # Visualise.
    vis = Image.new('RGB', (res * 5, res * 1))
    vis.paste(src_pil, (res * 0, 0))
    vis.paste(tar_pil, (res * 1, 0))
    vis.paste(depth_pil, (res * 2, 0))
    vis.paste(warped_pil, (res * 3, 0))
    vis.paste(synthesized_pil, (res * 4, 0))

    return vis


def save_output(output_frames, output_path):
    # output_frames.save(output_path)
    # print(f"图像已保存到 {output_path}")
    # 检查列表是否为空
    if not output_frames:
        raise ValueError("图片列表为空！")

    # 获取视频的尺寸（以第一张图片的尺寸为准）
    first_image = output_frames[0]
    width, height = first_image.size
    frame_size = (width, height)

    # 定义视频的编码方式和帧率
    # 'mp4v' 是常用的编码器，可以根据需要更换
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24  # 帧率，可以根据需要调整

    # 创建VideoWriter对象
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for img in output_frames:
        # 确保图片尺寸一致
        if img.size != (width, height):
            img = img.resize((width, height))

        # 转换PIL Image为OpenCV的BGR格式的numpy数组
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 写入视频
        out.write(img_cv)

    # 释放资源
    out.release()
    print(f"视频已保存为{output_path}")


def main(dav2_outdoor, dav2_model, video_file, camera_pose_file, res, output_path):
    depth_anything, genwarp_nvs = prepare_models(dav2_outdoor, dav2_model)
    frames, width, height = prepare_frames(video_file)
    camera_poses = prepare_camera_poses(camera_pose_file)

    output_frames = []
    src_frame = frames[0]
    src_camera_pose = camera_poses[0]
    focal_length_x = src_camera_pose[1]
    focal_length_y = src_camera_pose[2]
    principal_point_x = src_camera_pose[3]
    principal_point_y = src_camera_pose[4]

    for frame, camera_pose in tqdm(zip(frames, camera_poses), total=len(frames), desc="Processing frames"):
        with torch.no_grad():
            vis = process_one_frame(
                src_frame,
                src_camera_pose,
                frame,
                camera_pose,
                width,
                height,
                focal_length_x,
                focal_length_y,
                principal_point_x,
                principal_point_y,
                res,
                depth_anything,
                genwarp_nvs,
            )
        output_frames.append(vis)

    save_output(output_frames, output_path)


if __name__ == "__main__":
    # Indoor or outdoor model selection for DepthAnythingV2
    dav2_outdoor = False  # Set True for outdoor, False for indoor
    dav2_model = 'vitl'  # ['vits', 'vitb', 'vitl']

    # Example.
    video_file = 'assets/realestate8bd5cb1a874d6fb2.mp4'
    camera_pose_file = 'assets/realestate8bd5cb1a874d6fb2.txt'

    # Resolution (the image will be cropped into square).
    res = 512  # in px

    output_path = "output/7.mp4"

    main(dav2_outdoor, dav2_model, video_file, camera_pose_file, res, output_path)
