# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    from src.utils.cluster import dataset_paths
except ImportError:
    dataset_paths = None

from src.utils.logging import get_logger
from torch.utils.data import default_collate

logger = get_logger("Datasets utils")


def default_collate_with_patient_ids(batch):
    """
    Uses PyTorch's native default_collate for standard elements,
    but manually extracts and passes through patient_ids and video_paths as lists.
    Handles 5-element batches (clips, label, clip_indices, patient_id, video_path).
    """
    # 1. Extract patient IDs and video paths
    patient_ids = [item[3] if len(item) > 3 else None for item in batch]
    video_paths = [item[4] if len(item) > 4 else f"video_{i}" for i, item in enumerate(batch)]

    # 2. Create a new batch containing ONLY the elements PyTorch knows how to collate
    # (clips, label, clip_indices)
    batch_without_pids = [(item[0], item[1], item[2]) for item in batch]

    # 3. Let PyTorch's battle-tested default_collate handle the tensors exactly as it did before
    collated_clips, collated_labels, collated_indices = default_collate(batch_without_pids)

    # 4. Return the fully collated tensors plus our raw list of patient IDs and video paths
    return (collated_clips, collated_labels, collated_indices, patient_ids, video_paths)


def get_dataset_paths(datasets: list[str]):
    paths = []
    for d in datasets:
        try:
            path = dataset_paths().get(d)
        except Exception:
            raise Exception(f"Unknown dataset: {d}")
        paths.append(path)
    logger.info(f"Datapaths {paths}")
    return paths
