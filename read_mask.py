import gc
import cv2
import torch
import random
import numpy as np
from contextlib import contextmanager
from decord import VideoReader
from func_timeout import FunctionTimedOut, func_timeout
import torchvision.transforms as transforms

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
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames


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
            pixel_values = func_timeout(VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args)
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
    return pixel_values, text, 'video'


def main(video_path):
    video_sample_n_frames = 25
    video_length_drop_end = 0.9
    video_length_drop_start = 0.1
    video_sample_stride = 3
    image_sample_size = 1024
    video_sample_size = 256
    larger_side_of_image_and_video = max(min(image_sample_size), min(video_sample_size))
    enable_bucket = False
    video_transforms = transforms.Compose(
        [
            transforms.Resize(min(video_sample_size)),
            transforms.CenterCrop(video_sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    text_drop_ratio = -1

    pixel_value, mask, ground_truth = read_data(
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


if __name__ == "__main__":
    video_path = "assets/6368bc9ee243f179.mp4"
    main(video_path)
