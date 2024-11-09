import torch
import numpy as np
import cv2
from tqdm import tqdm 
from dataclasses import dataclass


@dataclass
class Sam2Output:
    high_res_masks: torch.Tensor
    obj_ptr: torch.Tensor
    object_score_logits: torch.Tensor
    maskmem_features: torch.Tensor
    maskmem_pos_enc: torch.Tensor

def get_tensor_images(np_images: np.ndarray, new_size: int, img_mean: tuple = (0.485, 0.456, 0.406), img_std: tuple = (0.229, 0.224, 0.225)) -> torch.Tensor:

    images = torch.zeros(len(np_images), 3, new_size, new_size, dtype=torch.float32)

    for n, img in enumerate(tqdm(np_images, desc="frame loading (JPEG)")):

        img_np = cv2.resize(img,(new_size, new_size))
        if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
            img_np = img_np / 255.0

        img = torch.from_numpy(img_np).permute(2, 0, 1)
        images[n] = img

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    images -= img_mean
    images /= img_std
    return images


def load_np_images(
    images: np.ndarray | list,
    image_size,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
):
    if isinstance(images, list):
        images = np.asarray(images)
    _, h, w, _ = images.shape
    return get_tensor_images(np_images=images, new_size=image_size, img_mean=img_mean, img_std=img_std), h, w
