import gc
import random
from contextlib import contextmanager

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from func_timeout import FunctionTimedOut, func_timeout

VIDEO_READER_TIMEOUT = 20


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()


def get_video_reader_batch(video_reader, batch_index):
    total_frames = video_reader.get_batch(batch_index).asnumpy()

    # 第一步：垂直分割成 2 行，每行高度为 512
    rows = np.split(total_frames, 2, axis=1)  # 生成列表，包含 2 个数组，每个数组形状为 (25, 512, 1536, 3)

    # 第二步：对每一行进行水平分割，分成 3 列，每列宽度为 512
    videos = []
    for row in rows:
        cols = np.split(row, 3, axis=2)  # 每行分成 3 列，shape 为 (25, 512, 512, 3)
        videos.extend(cols)  # 将分割后的列添加到 videos 列表中

    pixel_values = videos[3]
    masks = videos[4][:, :, :, 0:1]
    ground_truth = videos[5]

    return pixel_values, masks, ground_truth


def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)

    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame


def read_data(
    video_path,
    video_sample_n_frames,
    video_length_drop_end,
    video_length_drop_start,
    video_sample_stride,
    larger_side_of_image_and_video,
    enable_bucket,
    video_transforms,
    text_drop_ratio,
):
    with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
        min_sample_n_frames = min(video_sample_n_frames, int(len(video_reader) * (video_length_drop_end - video_length_drop_start) // video_sample_stride))
        if min_sample_n_frames == 0:
            raise ValueError(f"No Frames in video.")

        video_length = int(video_length_drop_end * len(video_reader))
        clip_length = min(video_length, (min_sample_n_frames - 1) * video_sample_stride + 1)
        start_idx = random.randint(int(video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

        try:
            sample_args = (video_reader, batch_index)
            pixel_values, masks, ground_truth = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args)
            resized_frames = []
            for i in range(len(pixel_values)):
                frame = pixel_values[i]
                resized_frame = resize_frame(frame, larger_side_of_image_and_video)
                resized_frames.append(resized_frame)
            pixel_values = np.array(resized_frames)
        except FunctionTimedOut:
            raise ValueError(f"Read timeout.")
        except Exception as e:
            raise ValueError(f"Failed to extract frames from video. Error is {e}.")

        if not enable_bucket:
            pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.0
            del video_reader
        else:
            pixel_values = pixel_values

        if not enable_bucket:
            pixel_values = video_transforms(pixel_values)

        # Random use no text generation
        if random.random() < text_drop_ratio:
            text = ''
    return pixel_values, masks, ground_truth


def process_mask(masks, h, w):
    """
    处理 mask，将其从 (25, 512, 512, 1) 转换为 (25, 1, 256, 256) 的 torch.uint8 张量，
    且值为 0 或 1。

    Args:
        masks (np.ndarray): 输入的 mask，形状为 (25, 512, 512, 1)，值范围 0-255
        h (int): 输出的高度，默认 256
        w (int): 输出的宽度，默认 256

    Returns:
        torch.Tensor: 处理后的 mask，形状为 (25, 1, 256, 256)，dtype=torch.uint8，值为 0 或 1
    """
    # 去除最后一个维度，形状变为 (25, 512, 512)
    masks = np.squeeze(masks, axis=-1)

    # 转换为浮点型的 torch 张量
    masks = torch.from_numpy(masks).float()  # (25, 512, 512)

    # 添加通道维度，形状变为 (25, 1, 512, 512)
    masks = masks.unsqueeze(1)

    # 使用双线性插值将 mask 缩放到 (256, 256)
    # 对于二值 mask，建议使用 'nearest' 模式以避免插值引入中间值
    masks = F.interpolate(masks, size=(h, w), mode='nearest')

    # 将 mask 的值二值化为 0 或 1
    masks = (masks > 128).to(torch.uint8)

    return masks


def inverse_normalize(tensor, mean, std):
    """
    对标准化的 tensor 进行逆标准化。

    参数:
    - tensor (torch.Tensor): 标准化后的 tensor，形状为 [N, C, H, W]
    - mean (list或tuple): 每个通道的均值
    - std (list或tuple): 每个通道的标准差

    返回:
    - torch.Tensor: 逆标准化后的 tensor
    """
    # 确保 mean 和 std 是 tensor，并且 shape 可用于广播
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return tensor * std + mean


def save_tensors_side_by_side_to_mp4(pixel_tensor, mask_tensor, filename, fps=25, normalize=True, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    将像素值和 mask tensor 并排保存为 MP4 视频，并且可选地进行逆标准化。

    参数：
    - pixel_tensor (torch.Tensor): 视频的像素值，形状为 [帧数, 3, 高, 宽]
    - mask_tensor (torch.Tensor): mask 值，形状为 [帧数, 1, 高, 宽]
    - filename (str): 输出的 MP4 文件路径
    - fps (int): 帧率，默认 25
    - normalize (bool): 是否进行逆标准化，默认 True
    - mean (list或tuple): 标准化时的均值，默认 [0.5, 0.5, 0.5]
    - std (list或tuple): 标准化时的标准差，默认 [0.5, 0.5, 0.5]
    """
    # 逆标准化
    if normalize:
        pixel_tensor = inverse_normalize(pixel_tensor, mean, std)

    # 确保 tensors 在 CPU 上并转为 numpy
    pixel_tensor = pixel_tensor.cpu()
    mask_tensor = mask_tensor.cpu()

    num_frames, _, height, width = pixel_tensor.shape

    # 将 pixel tensor 从 [N, 3, H, W] 变为 [N, H, W, 3] 并转换为 numpy 数组
    pixel_np = pixel_tensor.permute(0, 2, 3, 1).numpy()

    # 将 pixel_np 从 [0, 1] 或 [0, 255] 缩放到 [0, 255] 并转换为 uint8
    if pixel_np.dtype in [np.float32, np.float64]:
        # 假设逆标准化后 pixel_np 在 [0, 1] 或 [0, 255] 之间
        if pixel_np.max() <= 1.0:
            pixel_np = np.clip(pixel_np * 255, 0, 255).astype(np.uint8)
        else:
            pixel_np = np.clip(pixel_np, 0, 255).astype(np.uint8)
    else:
        pixel_np = np.clip(pixel_np, 0, 255).astype(np.uint8)

    # 将 mask tensor 从 [N, 1, H, W] 变为 [N, H, W, 1] 并转换为 numpy 数组
    mask_np = mask_tensor.permute(0, 2, 3, 1).numpy()

    # 将 mask 转换为 3 通道并缩放到 0-255
    mask_np = (mask_np > 0).astype(np.uint8) * 255
    mask_np = np.repeat(mask_np, 3, axis=3)  # [N, H, W, 3]

    # 确保 pixel 和 mask 的高度和宽度一致
    assert pixel_np.shape[1] == mask_np.shape[1] and pixel_np.shape[2] == mask_np.shape[2], "Pixel and mask dimensions must match."

    # 合并 pixel 和 mask，左右并排
    # 新的宽度为 width * 2
    combined_np = np.concatenate((pixel_np, mask_np), axis=2)  # [N, H, W*2, 3]

    # 使用 imageio 写入 MP4
    # 确保使用 'ffmpeg' 作为后端，以支持 MP4 格式
    writer = imageio.get_writer(filename, fps=fps, codec='libx264', format='mp4')

    for frame in combined_np:
        writer.append_data(frame)

    writer.close()
    print(f"视频已成功保存到 {filename}")


def main(video_path):
    video_sample_n_frames = 25
    video_length_drop_end = 0.9
    video_length_drop_start = 0.1
    video_sample_stride = 1
    image_sample_size = 1024
    image_sample_size = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
    video_sample_size = 256
    video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
    larger_side_of_image_and_video = max(min(image_sample_size), min(video_sample_size))
    enable_inpaint = True
    enable_bucket = False
    video_transforms = transforms.Compose(
        [
            transforms.Resize(min(video_sample_size)),
            transforms.CenterCrop(video_sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    text_drop_ratio = -1

    pixel_values, masks, ground_truth = read_data(
        video_path,
        video_sample_n_frames,
        video_length_drop_end,
        video_length_drop_start,
        video_sample_stride,
        larger_side_of_image_and_video,
        enable_bucket,
        video_transforms,
        text_drop_ratio,
    )

    if enable_inpaint and not enable_bucket:
        h, w = pixel_values.shape[2], pixel_values.shape[3]  # torch.Size([25, 3, 256, 256])
        masks = process_mask(masks, h, w)
        # masks = torch.tensor(masks, dtype=torch.uint8).permute(0, 3, 1, 2) / 255.0  #  mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

        mask_pixel_values = pixel_values * (1 - masks) + torch.ones_like(pixel_values) * -1 * masks

    save_tensors_side_by_side_to_mp4(pixel_values, masks, 'output_video.mp4', fps=5)

    print('done')


if __name__ == "__main__":
    video_path = "assets/6368bc9ee243f179.mp4"
    main(video_path)
