import os
import sys
import cv2
from jaxtyping import Float
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Union, Dict
from splatting import cpu as splatting_cpu

if torch.cuda.is_available():
    from splatting import cuda as splatting_cuda
else:
    splatting_cuda = None
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

sys.path.append('./extern/Depth-Anything-V2/metric_depth')
from depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image
from genwarp.ops import get_projection_matrix


def save_images_to_mp4_opencv(images, output_path, fps=30):
    if not images:
        raise ValueError("图像列表为空")

    # 获取图像尺寸
    width, height = images[0].size
    frame_size = (width, height)

    # 定义视频编码器和视频编写器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 也可以尝试 'avc1' 或 'X264'
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for img in images:
        # 将PIL图像转换为numpy数组并从RGB转为BGR
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame)

    # 释放视频写入器
    out.release()


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


def preprocess_image(image: Float[Tensor, 'B C H W']) -> Float[Tensor, 'B C H W']:
    image = F.interpolate(image, (512, 512))
    return image


def get_viewport_matrix(
    width: int,
    height: int,
    batch_size: int = 1,
    device: torch.device = None,
) -> Float[Tensor, 'B 4 4']:
    N = torch.tensor([[width / 2, 0, 0, width / 2], [0, height / 2, 0, height / 2], [0, 0, 1 / 2, 1 / 2], [0, 0, 0, 1]], dtype=torch.float32, device=device)[None].repeat(
        batch_size, 1, 1
    )
    return N


class SummationSplattingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, frame, flow):
        assert frame.dtype == flow.dtype
        assert frame.device == flow.device
        assert len(frame.shape) == 4
        assert len(flow.shape) == 4
        assert frame.shape[0] == flow.shape[0]
        assert frame.shape[2] == flow.shape[2]
        assert frame.shape[3] == flow.shape[3]
        assert flow.shape[1] == 2
        ctx.save_for_backward(frame, flow)
        output = torch.zeros_like(frame)
        if frame.is_cuda:
            if splatting_cuda is not None:
                splatting_cuda.splatting_forward_cuda(frame, flow, output)
            else:
                raise RuntimeError("splatting.cuda is not available")
        else:
            splatting_cpu.splatting_forward_cpu(frame, flow, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        frame, flow = ctx.saved_tensors
        grad_frame = torch.zeros_like(frame)
        grad_flow = torch.zeros_like(flow)
        if frame.is_cuda:
            if splatting_cuda is not None:
                splatting_cuda.splatting_backward_cuda(frame, flow, grad_output, grad_frame, grad_flow)
            else:
                raise RuntimeError("splatting.cuda is not available")
        else:
            splatting_cpu.splatting_backward_cpu(frame, flow, grad_output, grad_frame, grad_flow)
        return grad_frame, grad_flow


SPLATTING_TYPES = ["summation", "average", "linear", "softmax"]


def splatting_function(
    splatting_type: str,
    frame: torch.Tensor,
    flow: torch.Tensor,
    importance_metric: Union[torch.Tensor, None] = None,
    eps: float = 1e-7,
) -> torch.Tensor:
    if splatting_type == "summation":
        assert importance_metric is None
    elif splatting_type == "average":
        assert importance_metric is None
        importance_metric = frame.new_ones([frame.shape[0], 1, frame.shape[2], frame.shape[3]])
        frame = torch.cat([frame, importance_metric], 1)
    elif splatting_type == "linear":
        assert isinstance(importance_metric, torch.Tensor)
        assert importance_metric.shape[0] == frame.shape[0]
        assert importance_metric.shape[1] == 1
        assert importance_metric.shape[2] == frame.shape[2]
        assert importance_metric.shape[3] == frame.shape[3]
        frame = torch.cat([frame * importance_metric, importance_metric], 1)
    elif splatting_type == "softmax":
        assert isinstance(importance_metric, torch.Tensor)
        assert importance_metric.shape[0] == frame.shape[0]
        assert importance_metric.shape[1] == 1
        assert importance_metric.shape[2] == frame.shape[2]
        assert importance_metric.shape[3] == frame.shape[3]
        importance_metric = importance_metric.exp()
        frame = torch.cat([frame * importance_metric, importance_metric], 1)
    else:
        raise NotImplementedError("splatting_type has to be one of {}, not '{}'".format(SPLATTING_TYPES, splatting_type))

    output = SummationSplattingFunction.apply(frame, flow)

    if splatting_type != "summation":
        output = output[:, :-1, :, :] / (output[:, -1:, :, :] + eps)

    return output


def forward_warper(
    image: Float[Tensor, 'B C H W'],
    screen: Float[Tensor, 'B (H W) 2'],
    pcd: Float[Tensor, 'B (H W) 4'],
    mvp_mtx: Float[Tensor, 'B 4 4'],
    viewport_mtx: Float[Tensor, 'B 4 4'],
    alpha: float = 0.5,
) -> Dict[str, Tensor]:
    H, W = image.shape[2:4]

    # Projection.
    points_c = pcd @ mvp_mtx.mT
    points_ndc = points_c / points_c[..., 3:4]
    # To screen.
    coords_new = points_ndc @ viewport_mtx.mT

    # Masking invalid pixels.
    invalid = coords_new[..., 2] <= 0
    coords_new[invalid] = -1000000 if coords_new.dtype == torch.float32 else -1e4

    # Calculate flow and importance for splatting.
    new_z = points_c[..., 2:3]
    flow = coords_new[..., :2] - screen[..., :2]
    ## Importance.
    importance = alpha / new_z
    importance -= importance.amin((1, 2), keepdim=True)
    importance /= importance.amax((1, 2), keepdim=True) + 1e-6
    importance = importance * 10 - 10
    ## Rearrange.
    importance = rearrange(importance, 'b (h w) c -> b c h w', h=H, w=W)
    flow = rearrange(flow, 'b (h w) c -> b c h w', h=H, w=W)

    # Splatting.
    warped = splatting_function('softmax', image, flow, importance, eps=1e-6)
    ## mask is 1 where there is no splat
    mask = (warped == 0.0).all(dim=1, keepdim=True).to(image.dtype)
    flow2 = rearrange(coords_new[..., :2], 'b (h w) c -> b c h w', h=H, w=W)

    output = dict(warped=warped, mask=mask, correspondence=flow2)

    return output


class Embedder:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs) -> Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 2,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed


def warp_function(
    src_image: Float[Tensor, 'B C H W'],
    src_depth: Float[Tensor, 'B C H W'],
    rel_view_mtx: Float[Tensor, 'B 4 4'],
    src_proj_mtx: Float[Tensor, 'B 4 4'],
    tar_proj_mtx: Float[Tensor, 'B 4 4'],
    # viewport_mtx: Float[Tensor, 'B 4 4'],
):
    device = "cuda"
    dtype = torch.float16

    batch_size = src_image.shape[0]

    viewport_mtx: Float[Tensor, 'B 4 4'] = get_viewport_matrix(512, 512, batch_size=1, device=device).to(dtype)

    # Rearrange and resize.
    src_image = preprocess_image(src_image)
    src_depth = preprocess_image(src_depth)
    viewport_mtx = repeat(viewport_mtx, 'b h w -> (repeat b) h w', repeat=batch_size)

    B = src_image.shape[0]
    H, W = src_image.shape[2:4]
    src_scr_mtx = (viewport_mtx @ src_proj_mtx).to(src_proj_mtx)
    mvp_mtx = (tar_proj_mtx @ rel_view_mtx).to(rel_view_mtx)

    # Coordinate grids.
    grid: Float[Tensor, 'H W C'] = torch.stack(torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy'), dim=-1).to(device, dtype=dtype)

    # Unproject depth.
    screen = F.pad(grid, (0, 1), 'constant', 0)  # z=0 (z doesn't matter)
    screen = F.pad(screen, (0, 1), 'constant', 1)  # w=1
    screen = repeat(screen, 'h w c -> b h w c', b=B)
    screen_flat = rearrange(screen, 'b h w c -> b (h w) c')
    # To eye coordinates.
    eye = screen_flat @ torch.linalg.inv_ex(src_scr_mtx.float())[0].mT.to(dtype)
    # Overwrite depth.
    eye = eye * rearrange(src_depth, 'b c h w -> b (h w) c')
    eye[..., 3] = 1

    # Coordinates embedding.
    coords = torch.stack((grid[..., 0] / H, grid[..., 1] / W), dim=-1)
    embedder = get_embedder(2)
    embed = repeat(embedder(coords), 'h w c -> b c h w', b=B)

    # Warping.
    input_image: Float[Tensor, 'B C H W'] = torch.cat([embed, src_image], dim=1)
    output_image = forward_warper(input_image, screen_flat[..., :2], eye, mvp_mtx=mvp_mtx, viewport_mtx=viewport_mtx)
    # warped_embed = output_image['warped'][:, : embed.shape[1]]
    warped_image = output_image['warped'][:, embed.shape[1] :]

    return warped_image


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


def convert_camera_extrinsics(w2c):
    # 获取设备和数据类型，以确保缩放矩阵与w2c在同一设备和数据类型
    device = w2c.device
    dtype = w2c.dtype

    # 定义缩放矩阵，x和y轴取反，z轴保持不变
    S = torch.diag(torch.tensor([1, -1, -1], device=device, dtype=dtype))

    # 将缩放矩阵应用于旋转和平移部分
    R = w2c[:, :3]  # 3x3
    t = w2c[:, 3]  # 3

    new_R = S @ R  # 矩阵乘法
    new_t = S @ t  # 向量乘法

    # 构建新的外参矩阵
    new_w2c = torch.cat((new_R, new_t.unsqueeze(1)), dim=1)  # 3x4

    return new_w2c


def get_rel_view_mtx(src_wc, tar_wc, src_image):
    src_wc = convert_camera_extrinsics(src_wc)
    tar_wc = convert_camera_extrinsics(tar_wc)

    # 将第一个 W2C 矩阵扩展为 4x4 齐次变换矩阵
    T1 = torch.eye(4, dtype=src_wc.dtype, device=src_wc.device)
    T1[:3, :3] = src_wc[:, :3]
    T1[:3, 3] = src_wc[:, 3]

    # 将第二个 W2C 矩阵扩展为 4x4 齐次变换矩阵
    T2 = torch.eye(4, dtype=tar_wc.dtype, device=tar_wc.device)
    T2[:3, :3] = tar_wc[:, :3]
    T2[:3, 3] = tar_wc[:, 3]

    # 计算第一个视图矩阵的逆
    T1_inv = torch.inverse(T1)

    # 计算相对视图矩阵
    rel_view_mtx = T2 @ T1_inv

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
):
    # Load an image.
    # src_image = np.asarray(crop(Image.open(image_file).convert('RGB')).resize((res, res)))
    src_image = np.asarray(crop(Image.fromarray(src_frame)).resize((res, res)))
    tar_image = np.asarray(crop(Image.fromarray(tar_frame)).resize((res, res)))

    # Estimate the depth.
    src_depth = depth_anything.infer_image(src_image[..., ::-1].copy())

    # Go half precision.
    tar_image = torch.from_numpy(tar_image / 255.0).permute(2, 0, 1)[None].cuda().half()
    src_image = torch.from_numpy(src_image / 255.0).permute(2, 0, 1)[None].cuda().half()
    src_image = torch.randn_like(src_image)
    src_depth = torch.from_numpy(src_depth)[None, None].cuda().half()

    # Projection matrix.
    src_proj_mtx = get_src_proj_mtx(focal_length_x, focal_length_y, height, width, res, src_image)
    ## Use the same projection matrix for the source and the target.
    tar_proj_mtx = src_proj_mtx

    src_wc = torch.tensor(src_camera_pose[7:]).reshape((3, 4))
    tar_wc = torch.tensor(tar_camera_pose[7:]).reshape((3, 4))

    rel_view_mtx = get_rel_view_mtx(src_wc, tar_wc, src_image)

    warped_image = warp_function(
        src_image,
        src_depth,
        rel_view_mtx,
        src_proj_mtx,
        tar_proj_mtx,
        # viewport_mtx,
    )
    warped_pil = to_pil_image(warped_image[0])

    return warped_pil


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

    return depth_anything


def main(video_file, camera_pose_file, output_path):
    depth_anything = prepare_models(dav2_outdoor=False, dav2_model='vitl')
    frames, width, height = prepare_frames(video_file)
    camera_poses = prepare_camera_poses(camera_pose_file)
    res = 32

    output_frames = []
    src_frame = frames[0]
    src_camera_pose = camera_poses[0]
    focal_length_x = src_camera_pose[1]
    focal_length_y = src_camera_pose[2]
    principal_point_x = src_camera_pose[3]
    principal_point_y = src_camera_pose[4]

    for frame, camera_pose in tqdm(zip(frames, camera_poses), total=len(frames), desc="Processing frames"):
        with torch.no_grad():
            warped_pil = process_one_frame(
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
            )
        output_frames.append(warped_pil)

    save_images_to_mp4_opencv(output_frames, output_path, fps=8)


if __name__ == "__main__":
    video_file = "/mnt/chenyang_lei/Datasets/easyanimate_dataset/realestate_dataset/train_clips/XDj-cBQKGLY/6368bc9ee243f179.mp4"
    camera_pose_file = "/mnt/chenyang_lei/Datasets/easyanimate_dataset/realestate_dataset/train_poses/6368bc9ee243f179.txt"
    output_path = "output_video.mp4"

    main(video_file, camera_pose_file, output_path)
