from pathlib import Path
import re

import cv2
from PIL import Image
from functools import partial

from typing import Tuple, List
# from beartype.door import is_bearable

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader
from torchvision import transforms as T, utils

from einops import rearrange
import os
import pickle as pkl
import random

# helper functions

from scipy import interpolate
import torchvision.utils as vutils
from PIL import Image


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def pair(val):
    return val if isinstance(val, tuple) else (val, val)


def bgr_to_rgb(video_tensor):
    video_tensor = video_tensor[[2, 1, 0], :, :, :]
    return video_tensor



def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class VideoFrameTransform:
    """Picklable video frame transform for DataLoader worker processes."""

    def __init__(self, image_size, use_augmentation=False, norm_type="imagenet"):
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.to_pil = T.ToPILImage()
        self.resize = T.Resize(image_size)
        self.to_tensor = T.ToTensor()

        # Add normalization options
        if norm_type == "imagenet":
            self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif norm_type == "symmetric":
            self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        else:
            self.normalize = torch.nn.Identity()

    def __call__(self, img):
        # 1. FIX: Convert BGR (from cv2) to RGB before PIL conversion
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not isinstance(img, Image.Image):
            img = self.to_pil(img)

        img = self.resize(img)

        if self.use_augmentation:
            shear_x = random.uniform(-5, 5)
            shear_y = random.uniform(-5, 5)
            contrast_factor = random.uniform(0.6, 1.4)
            img = T.functional.adjust_contrast(img, contrast_factor)
            img = T.functional.affine(
                img,
                angle=0,
                translate=[0, 0],
                scale=1.0,
                shear=[shear_x, shear_y],
            )

        tensor = self.to_tensor(img)

        # 2. FIX: Apply normalization
        return self.normalize(tensor)

# image related helpers functions and dataset
def z_normalize(data):
    """
    Perform z-score normalization on the input data.
    """
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def save_tensor_images(tensor, output_dir="output_images"):
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 텐서의 값을 0-255 범위로 정규화
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.byte()

    # 각 프레임을 개별적으로 저장
    for frame in range(tensor.shape[1]):  # 프레임 차원을 순회
        # 현재 프레임의 모든 채널 선택
        frame_data = tensor[:, frame, :, :]
        
        # 채널 순서 변경 (C, H, W) -> (H, W, C)
        frame_data = frame_data.permute(1, 2, 0)
        
        # NumPy 배열로 변환
        frame_array = frame_data.numpy()
        
        # RGB 이미지로 변환 (채널이 3개인 경우)
        if frame_array.shape[2] == 3:
            img = Image.fromarray(frame_array, 'RGB')
        else:
            img = Image.fromarray(frame_array[:,:,0], 'L')
        
        # 이미지 저장
        img.save(os.path.join(output_dir, f"frame_{frame:03d}.png"))
        
class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg', 'jpeg', 'png']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(
            f'{folder}').glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')

        self.transform = T.Compose([
            T.Lambda(partial(convert_image_to_fn, 'RGB')),
            T.Resize(image_size),
            # T.RandomHorizontalFlip(),
            # T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)

# tensor of shape (channels, frames, height, width) -> gif

# handle reading and writing gif


CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif


def video_tensor_to_pil_first_image(tensor):

    tensor = bgr_to_rgb(tensor)
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images

    return first_img


def video_tensor_to_gif(
    tensor,
    path,
    duration=120,
    loop=0,
    optimize=True
):

    tensor = torch.clamp(tensor, min=0, max=1) # clipping underflow and overflow
    #tensor = bgr_to_rgb(tensor)
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs,
                   loop=loop, optimize=optimize)
    return images

# gif -> (channels, frame, height, width) tensor


def gif_to_tensor(
    path,
    channels=3,
    transform=T.ToTensor()
):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)

# handle reading and writing mp4



def tensor_to_video(
    tensor,                # Pytorch video tensor
    path: str,             # Path of the video to be saved
    fps=8,              # Frames per second for the saved video
    video_format=('m', 'p', '4', 'v')
):
    # Import the video and cut it into frames.
    tensor = tensor.cpu()*255.  # TODO: have a better function for that? Not using cv2?

    num_frames, height, width = tensor.shape[-3:]

    # Changes in this line can allow for different video formats.
    fourcc = cv2.VideoWriter_fourcc(*video_format)
    video = cv2.VideoWriter(path, fourcc, fps, (width, height))

    frames = []

    for idx in range(num_frames):
        numpy_frame = tensor[:, idx, :, :].numpy()
        numpy_frame = np.uint8(rearrange(numpy_frame, 'c h w -> h w c'))
        video.write(numpy_frame)

    video.release()

    cv2.destroyAllWindows()

    return video


def crop_center(
    img,        # tensor
    cropx,      # Length of the final image in the x direction.
    cropy       # Length of the final image in the y direction.
) -> torch.Tensor:
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:(starty + cropy), startx:(startx + cropx), :]

def sort_key(file_path):
    # Extract the numerical parts from the file name using regex
    match = re.findall(r'(\d+)', file_path.stem)
    if match:
        return [int(part) for part in match]
    return str(file_path)
# video dataset

def save_tensor_as_grid(tensor, grid_size, save_path="grid_image.png"):
    """
    4x4 그리드로 텐서를 저장하는 함수.

    Args:
        tensor (torch.Tensor): 텐서 크기 (C, N, H, W) 또는 (N, C, H, W).
            - C: 채널 수 (3이면 RGB, 1이면 그레이스케일).
            - N: 이미지 개수.
            - H, W: 이미지 높이와 너비.
        grid_size (int): 그리드의 행과 열 수. (예: 4이면 4x4 그리드).
        save_path (str): 저장할 이미지 파일 경로.
    """
    # 텐서 차원 맞추기: (N, C, H, W)
    if tensor.shape[0] == 3 or tensor.shape[0] == 1:
        tensor = tensor.permute(1, 0, 2, 3)  # (C, N, H, W) -> (N, C, H, W)

    # 그리드 생성
    grid = vutils.make_grid(tensor, nrow=grid_size, padding=2)

    # 텐서를 PIL 이미지로 변환
    grid_image = (grid * 255).byte().permute(1, 2, 0).numpy()  # RGB 순서로 변환
    image = Image.fromarray(grid_image)

    # 이미지 저장
    image.save(save_path)
    print(f"4x4 그리드 이미지를 '{save_path}'로 저장했습니다.")

def process_ekg(ekg_data, target_length=2250, repetitions=16):
    processed_ekg = np.zeros((12, target_length))
    
    # 원본 데이터를 16번 반복
    repeated_data = np.tile(ekg_data, repetitions)
    
    # 반복된 데이터의 길이
    original_length = len(repeated_data)
    
    if original_length < target_length:
        # Interpolation
        x = np.linspace(0, 1, original_length)
        f = interpolate.interp1d(x, repeated_data, kind='linear')
        x_new = np.linspace(0, 1, target_length)
        processed_ekg[1] = f(x_new)
    elif original_length > target_length:
        # Resampling
        x = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, target_length)
        processed_ekg[1] = np.interp(x_new, x, repeated_data)
    else:
        processed_ekg[1] = repeated_data[:target_length]
    
    return processed_ekg


def video_to_tensor(
    path: str,
    transform,              # Path of the video to be imported
    num_frames=-1,        # Number of frames to be stored in the output tensor
    crop_size=None
) -> torch.Tensor:          # shape (1, channels, frames, height, width)

    video = cv2.VideoCapture(path)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # print ("PATH", path)
    # print ("TOTAL frame : ",total_frames )
    frames = []
    check = True
    
    shear_x = random.uniform(-5, 5)
    shear_y = random.uniform(-5, 5)
    contrast_factor = random.uniform(0.6, 1.4)

    while check:
        check, frame = video.read()

        if not check:
            continue
        # frame = np.transpose(frame, (2, 0, 1))
        # 고정된 augmentation 값들로 transform 적용
        # frame = transform(frame, shear_x, shear_y, contrast_factor)
        frame = transform(frame)
        frames.append(rearrange(frame, '... -> 1 ...'))
         
    # convert list of frames to numpy array
    frames = np.array(np.concatenate(frames, axis=0))
    # frames = rearrange(frames, 'f c h w -> c f h w')
    frames = rearrange(frames, 'f c h w -> c f h w')

    frames_torch = torch.tensor(frames).float()

    return frames_torch



def process_ultrasound_image(video_tensor):
    # 입력 텐서의 shape 확인
    B, T, H, W = video_tensor.shape
    
    # 결과를 저장할 텐서 초기화
    result = torch.zeros_like(video_tensor)
    
    for b in range(B):
        for t in range(T):
            
            # 현재 프레임 추출
            frame = video_tensor[b, t].cpu().numpy() #shape of 128 128
            
            # frame_ = (frame*255).astype(np.uint8)

            # Save the grayscale image using OpenCV
            # output_path_gray = "/home/local/PARTNERS/sk1064/project/EchoHub/dataset/frame_image_gray.jpg"
            # cv2.imwrite(output_path_gray, frame_)
            
            # If you want to convert the numpy array to PIL Image and save it

            # print ("FRAME SIZE CHECK " *10, np.max(frame_))
            # 임계값 설정 (이 값은 이미지에 따라 조정이 필요할 수 있습니다)
            threshold = 0.1
            
            # 각 열에서 첫 번째로 임계값을 넘는 픽셀 찾기
            first_pixels = np.argmax(frame > threshold, axis=0)
            
            # 가장 위에 있는 픽셀의 y 좌표 찾기
            top_y = np.min(first_pixels[first_pixels > 0])
            
            # 128x128 크기로 자르기
            # cropped = frame[top_y:top_y+128, :128]
            cropped = frame[top_y:top_y+224, :224]
            
            # 크기가 128x128이 아닌 경우 인터폴레이션을 사용하여 리사이징
            # if cropped.shape != (128, 128):
            #     cropped = cv2.resize(cropped, (128, 128), interpolation=cv2.INTER_LINEAR)
            if cropped.shape != (224, 224):
                cropped = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            # 결과 텐서에 저장
            result[b, t] = torch.from_numpy(cropped).float()
    
    return result

class EchoDataset_from_Video_mp4(Dataset):
    def __init__(
        self,
        folder,
        image_size = [224, 224],
        channels = 3,
    ):
        super().__init__()
        self.folder = folder
        
        self.image_size = image_size
        self.channels = channels
        
        self.transform_for_videos = VideoFrameTransform(self.image_size, use_augmentation=False, norm_type="imagenet")

        self.transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])
        self.paths = os.listdir(folder)
        
        self.mp4_to_tensor = partial(
            video_to_tensor, transform=self.transform_for_videos, crop_size=self.image_size, num_frames=10)

        force_num_frames = True
        
        self.cast_num_frames_fn = partial(
            cast_num_frames, frames=32) if force_num_frames else identity
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        path = os.path.join(self.folder, path)
        tensor = self.mp4_to_tensor(str(path))

        tensor = self.cast_num_frames_fn(tensor)
        
        # print ("Check Final output : ", tensor.size())
        
        # save_tensor_as_grid(tensor, grid_size=4, save_path="grid_image.png")
        # data =  {"image":tensor , "p_id" : self.paths[index]}
        return tensor
        
         
class EchoDataset_from_Video(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels=3,
        num_frames=11,
        horizontal_flip=False,
        force_num_frames=True,
        exts=['gif', 'mp4'],
        sample_texts=None  # 新增参数
    ):
        super().__init__()
        self.folder = os.path.join(folder, "mp4")
        self.folder_ekg= os.path.join(folder, "ekg")
        
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(
            f'{folder}').glob(f'**/*.{ext}')]
        self.paths.sort(key=sort_key)
        self.sample_texts = sample_texts
        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

        # TODO: rework so it is faster, for now it works but is bad
        # self.transform_for_videos = T.Compose([
        #     T.ToPILImage(),  # added to PIL conversion because video is read with cv2
        #     T.Resize(image_size),
        #     # T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
        #     T.ToTensor()
        # ])

        self.gif_to_tensor = partial(
            gif_to_tensor, channels=self.channels, transform=self.transform)
        # self.mp4_to_tensor = partial(
        #     video_to_tensor, transform=self.transform_for_videos, crop_size=self.image_size, num_frames=num_frames)

        
        self.cast_num_frames_fn = partial(
            cast_num_frames, frames=num_frames) if force_num_frames else identity


        self.transform_for_videos = VideoFrameTransform(image_size, use_augmentation=True)
        self.mp4_to_tensor = partial(
            video_to_tensor, transform=self.transform_for_videos, crop_size=self.image_size, num_frames=num_frames)
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        ext = path.suffix

        if ext == '.gif':
            tensor = self.gif_to_tensor(path)
        elif ext == '.mp4':
            tensor = self.mp4_to_tensor(str(path))
        else:
            raise ValueError(f'unknown extension {ext}')

        tensor = self.cast_num_frames_fn(tensor)

        return tensor

def collate_tensors_and_strings(batch):
    tensors, ekgs = zip(*batch)
    
    # Process tensors (assuming they are already in the correct format)
    tensors = torch.stack(tensors, dim=0)
    
    # Process EKGs
    processed_ekgs = []
    for ekg in ekgs:
        processed_ekgs.append(ekg)
    
    processed_ekgs = torch.stack(processed_ekgs, dim=0)
    
    # print ("BATCH COLLATE ", tensors.size(), processed_ekgs.size())
    
    return tensors, processed_ekgs


# ---------------------------------------------------------------------------
#  Multiclip Dataset (V-JEPA style: N clips per video)
# ---------------------------------------------------------------------------


def read_all_frames_mp4(path: str, transform) -> list:
    """
    Read all frames from an mp4 video and apply transform to each frame.

    Args:
        path: Path to the video file.
        transform: Callable that maps a single frame (np.ndarray HxWxC BGR) to a
                   torch tensor (C, H, W).

    Returns:
        List of transformed frame tensors, each of shape (C, H, W).
    """
    video = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frame = transform(frame)
        frames.append(frame)
    video.release()
    return frames


def multiclip_video_to_tensor(
    path: str,
    transform,
    num_clips: int = 3,
    frames_per_clip: int = 16,
    random_clip_sampling: bool = True,
    allow_clip_overlap: bool = False,
    frame_step: int = 1,
    filter_short_videos: bool = False,
):
    """
    Read a video and extract ``num_clips`` clips, each ``frames_per_clip`` frames.

    Sampling strategy mirrors V-JEPA's VideoDataset._sample_from_vr:
      - Partition the video into ``num_clips`` equal-length segments.
      - From each segment, sample ``frames_per_clip`` frames with stride
        ``frame_step`` (similar to V-JEPA's frame_step parameter).
      - If the segment is shorter than the required clip length, padding or
        overlap logic applies depending on ``allow_clip_overlap``.

    Args:
        path: Path to the mp4 file.
        transform: Per-frame callable returning (C, H, W) tensor.
        num_clips: Number of clips to extract.
        frames_per_clip: Number of frames per clip.
        random_clip_sampling: If True, sample a random window within each segment.
        allow_clip_overlap: If True, clips may overlap across segment boundaries
            when segments are too short.
        frame_step: Temporal stride between consecutive sampled frames
            (1 = consecutive frames).
        filter_short_videos: If True, skip videos shorter than the required
            total span. Returns ([], []) for skipped videos.

    Returns:
        clip_tensors: List of tensors, each of shape (C, frames_per_clip, H, W).
        clip_indices: List of tensors, each of shape (frames_per_clip,),
            containing the global frame indices of the sampled frames.
    """
    frames = read_all_frames_mp4(path, transform)
    total_frames = len(frames)

    if total_frames == 0:
        return [], []

    # Determine partition length (frames per video segment)
    if num_clips > 1:
        partition_len = total_frames // num_clips
    else:
        partition_len = total_frames

    clip_len = frames_per_clip * frame_step  # number of frames spanned by one clip

    if filter_short_videos and total_frames < clip_len:
        return [], []

    clip_tensors = []
    clip_indices = []

    for i in range(num_clips):
        seg_start = i * partition_len
        seg_end = seg_start + partition_len

        if partition_len >= clip_len:
            # Enough frames in segment: pick a random window
            end_idx = clip_len
            if random_clip_sampling:
                end_idx = np.random.randint(clip_len, partition_len + 1)
            start_idx = end_idx - clip_len

            sample_positions = np.linspace(start_idx, end_idx - 1, num=frames_per_clip).astype(int)
            indices = np.clip(sample_positions, start_idx, end_idx - 1)
            indices = indices + seg_start
        else:
            # Segment too short
            if not allow_clip_overlap:
                # Take all available frames, pad last frame if needed
                sample_positions = np.linspace(0, partition_len - 1, num=min(partition_len, frames_per_clip), dtype=int)
                indices = sample_positions.copy()
                if len(indices) < frames_per_clip:
                    indices = np.concatenate([indices, np.full(frames_per_clip - len(indices), partition_len - 1)])
                indices = np.clip(indices, 0, partition_len - 1)
                indices = indices + seg_start
            else:
                # Allow overlap: sample across segment boundary
                sample_len = min(clip_len, total_frames) - 1
                sample_positions = np.linspace(0, sample_len, num=frames_per_clip, dtype=int)
                indices = np.clip(sample_positions, 0, sample_len - 1)
                clip_step = 0
                if total_frames > clip_len and num_clips > 1:
                    clip_step = (total_frames - clip_len) // (num_clips - 1)
                indices = indices + i * clip_step
                indices = np.clip(indices, 0, total_frames - 1)

        # Gather frames
        # NOTE: frame_step is handled via linspace above, so indices are already spaced
        clip_frames = [frames[idx] for idx in indices]

        # Stack: list of (C, H, W) -> (C, frames_per_clip, H, W)
        clip_tensor = torch.stack(clip_frames, dim=1)
        clip_tensors.append(clip_tensor)
        clip_indices.append(torch.from_numpy(indices.astype(np.int64)))

    return clip_tensors, clip_indices


class MulticlipVideoDataset(Dataset):
    """
    Video dataset that returns ``num_clips`` clips per video.

    Compatible with the V-JEPA-style ClipAggregation / MulticlipEncoder
    processing pipeline. Each __getitem__ returns a list of clip tensors,
    a list of clip_indices, the label, and the patient_id.

    The returned tuple is:
        (clip_tensors, label, clip_indices, patient_id)
    where:
        - clip_tensors: list of tensors, each (C, frames_per_clip, H, W)
        - label: int (or float) class label
        - clip_indices: list of tensors, each (frames_per_clip,) frame indices
        - patient_id: str or None

    Args:
        samples: List of (video_path, label, patient_id) tuples.
        transform: Per-frame transform callable.
        num_clips: Number of clips per video.
        frames_per_clip: Number of frames per clip.
        frame_step: Temporal stride between frames.
        random_clip_sampling: Whether to randomise clip start positions.
        allow_clip_overlap: Whether to allow clip overlap for short videos.
        filter_short_videos: Skip videos shorter than required span.
    """

    def __init__(
        self,
        samples: list,
        transform,
        num_clips: int = 3,
        frames_per_clip: int = 16,
        frame_step: int = 1,
        random_clip_sampling: bool = True,
        allow_clip_overlap: bool = False,
        filter_short_videos: bool = False,
    ):
        super().__init__()
        self.samples = samples  # list of (path, label, patient_id)
        self.transform = transform
        self.num_clips = num_clips
        self.frames_per_clip = frames_per_clip
        self.frame_step = frame_step
        self.random_clip_sampling = random_clip_sampling
        self.allow_clip_overlap = allow_clip_overlap
        self.filter_short_videos = filter_short_videos

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, patient_id = self.samples[idx]

        clip_tensors, clip_indices = multiclip_video_to_tensor(
            path=str(video_path),
            transform=self.transform,
            num_clips=self.num_clips,
            frames_per_clip=self.frames_per_clip,
            random_clip_sampling=self.random_clip_sampling,
            allow_clip_overlap=self.allow_clip_overlap,
            frame_step=self.frame_step,
            filter_short_videos=self.filter_short_videos,
        )

        return clip_tensors, label, clip_indices, patient_id


def default_multiclip_collate(batch):
    """
    Collate function for MulticlipVideoDataset batches.

    Handles the nested list structure:
      batch[i] = (clip_tensors, label, clip_indices, patient_id)

    Uses PyTorch's default_collate recursively for the nested tensor
    structure, then appends patient_ids as a raw list.
    """
    from torch.utils.data import default_collate

    patient_ids = [item[3] if len(item) > 3 else None for item in batch]

    # Collate everything except patient_ids
    batch_collatable = [(item[0], item[1], item[2]) for item in batch]
    collated_clips, collated_labels, collated_indices = default_collate(batch_collatable)

    # collated_clips is a list (one entry per clip position),
    #   each entry is a stacked tensor [B, C, T, H, W]
    # collated_indices is a list (one entry per clip position),
    #   each entry is a stacked tensor [B, T]

    return collated_clips, collated_labels, collated_indices, patient_ids


def DataLoader(*args, **kwargs):
    return PytorchDataLoader(*args, collate_fn=collate_tensors_and_strings, **kwargs)
