# Copyright (c) 2026 Jonas Fallmann
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone mode collapse evaluation for EchoFM foundation models.

Evaluates the latent space of a frozen encoder on given datasets, reporting:
  - Intra-video temporal similarity (average pairwise cosine similarity of tokens within a video)
  - Active channel count (embedding dimensions with variance above a threshold)
  - Effective rank (dimensions needed to explain 90% of variance via SVD)
  - Inter-video global cosine similarity (average pairwise similarity between videos)

Usage:
    python eval_mode_collapse.py --config config/mode_collapse_config.yaml
"""

import os
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.dataset import EchoDataset_from_Video_mp4
from EchoFM.models_mae import mae_vit_base_patch16, mae_vit_large_patch16

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class VideoDatasetFromCSV(Dataset):
    """Dataset that loads echo videos from a CSV manifest.

    CSV format: patient_id, video_path, class
    (labels are parsed but not required for collapse evaluation.)
    """

    def __init__(self, csv_path, video_folder=None, resolution=224, num_frames=32):
        self.resolution = resolution
        self.num_frames = num_frames
        csv_path = str(csv_path)

        rows = []
        with open(csv_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                row = [part.strip() for part in line.split(",")]
                if row and len(row) >= 3:
                    rows.append(row)

        # Skip header row if present
        if rows:
            first_row = [str(v).strip().lower() for v in rows[0][:3]]
            if any(kw in first_row for kw in {"patient", "video", "path", "filepath",
                                               "filename", "sample", "class", "label", "target"}):
                rows = rows[1:]

        self.data_rows = rows
        if video_folder is None:
            video_folder = os.path.dirname(os.path.abspath(csv_path))
        else:
            video_folder = str(video_folder)

        self.video_folder = video_folder
        self.dataset = EchoDataset_from_Video_mp4(
            video_folder,
            image_size=[resolution, resolution],
            channels=3,
        )

        self.video_paths = []
        for row in self.data_rows:
            video_path = str(row[1]).strip()
            if not os.path.isabs(video_path):
                video_path = os.path.join(video_folder, video_path)
            self.video_paths.append(video_path)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_tensor = self.dataset.mp4_to_tensor(str(self.video_paths[idx]))
        video_tensor = self.dataset.cast_num_frames_fn(video_tensor)
        return video_tensor


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_encoder(model_name, checkpoint, device, model_kwargs):
    logger.info(f"Loading encoder: {model_name}")

    encoder_kwargs = {
        'img_size': 224,
        'in_chans': 3,
        'decoder_embed_dim': 512,
        'decoder_depth': 8,
        'decoder_num_heads': 16,
        'norm_pix_loss': False,
        'num_frames': 32,
        't_patch_size': 4,
    }
    encoder_kwargs.update(model_kwargs)

    if model_name == "mae_vit_base_patch16":
        encoder = mae_vit_base_patch16(**encoder_kwargs)
    elif model_name == "mae_vit_large_patch16":
        encoder = mae_vit_large_patch16(**encoder_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    encoder = encoder.to(device)

    if checkpoint and os.path.exists(checkpoint):
        logger.info(f"Loading checkpoint from {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded encoder with message: {msg}")
    else:
        logger.warning("No checkpoint provided, using random initialization.")

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    return encoder


# ---------------------------------------------------------------------------
# Collapse statistics
# ---------------------------------------------------------------------------

def accumulate_collapse_stats(outputs, global_embeddings, token_similarities):
    """Accumulates batch-level statistics for collapse evaluation.

    Args:
        outputs: encoder output tensor of shape (B, N, D)
        global_embeddings: list of pooled (B, D) CPU tensors
        token_similarities: list of scalar token similarity values

    Returns:
        tuple: updated (global_embeddings, token_similarities)
    """
    with torch.no_grad():
        # Intra-Video Token Similarity (Temporal Collapse)
        norm_outputs = F.normalize(outputs, p=2, dim=-1)
        token_sim_matrix = torch.bmm(norm_outputs, norm_outputs.transpose(1, 2))

        N = outputs.shape[1]
        if N > 1:
            mask = ~torch.eye(N, dtype=torch.bool, device=outputs.device)
            avg_token_sim = token_sim_matrix[:, mask].mean().item()
            token_similarities.append(avg_token_sim)

        # Global Embeddings (Global Collapse)
        pooled_outputs = outputs.mean(dim=1)  # (B, D)
        global_embeddings.append(pooled_outputs.cpu().float())

    return global_embeddings, token_similarities


def evaluate_collapse_metrics(global_embeddings, token_similarities,
                               variance_threshold=1e-4, max_samples_sim=1000):
    """Computes final mode collapse metrics from accumulated data.

    Args:
        global_embeddings: list of (B, D) CPU tensors
        token_similarities: list of scalar similarity values
        variance_threshold: threshold for "active" channel
        max_samples_sim: cap on samples for global similarity matrix

    Returns:
        dict with keys: temporal_similarity, active_channels, dim_90_variance,
                        global_cosine_similarity
    """
    Z = torch.cat(global_embeddings, dim=0)          # (M, D)
    total_channels = Z.shape[1]

    # A. Average Temporal/Token Similarity
    avg_temporal_sim = (sum(token_similarities) / len(token_similarities)
                        if token_similarities else float('nan'))

    # B. Active Channels
    variances = Z.var(dim=0)
    active_channels = (variances > variance_threshold).sum().item()

    # C. SVD / Effective Rank
    Z_centered = Z - Z.mean(dim=0)
    _, S, _ = torch.linalg.svd(Z_centered, full_matrices=False)
    explained_variance = (S ** 2) / (S ** 2).sum()
    cumulative_variance = torch.cumsum(explained_variance, dim=0)
    dim_90 = (cumulative_variance < 0.90).sum().item() + 1

    # D. Inter-Video Global Cosine Similarity
    Z_norm = F.normalize(Z, p=2, dim=1)
    subset_Z = Z_norm[:max_samples_sim]
    global_sim_matrix = torch.matmul(subset_Z, subset_Z.T)
    mask = ~torch.eye(subset_Z.shape[0], dtype=torch.bool, device=subset_Z.device)
    avg_global_sim = global_sim_matrix[mask].mean().item()

    return {
        "temporal_similarity": avg_temporal_sim,
        "active_channels": f"{active_channels} / {total_channels}",
        "dim_90_variance": f"{dim_90} / {total_channels}",
        "global_cosine_similarity": avg_global_sim,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_dataset(encoder, loader, device, use_bfloat16, dataset_name="dataset"):
    """Run encoder inference over a dataset and compute collapse metrics."""
    global_embeddings = []
    token_similarities = []

    for itr, batch in enumerate(loader):
        # Handle potential label / tuple batches
        if isinstance(batch, (tuple, list)):
            clips = batch[0]
        else:
            clips = batch

        clips = clips.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
            enabled=use_bfloat16 and torch.cuda.is_available(),
        ):
            with torch.no_grad():
                encoder_output = encoder.forward_encoder(clips, 0)
                if isinstance(encoder_output, (tuple, list)):
                    encoder_output = encoder_output[0]

                global_embeddings, token_similarities = accumulate_collapse_stats(
                    encoder_output, global_embeddings, token_similarities)

        if itr % 10 == 0:
            logger.info(f"  [{dataset_name}] processed {itr + 1} / {len(loader)} batches")

    return evaluate_collapse_metrics(global_embeddings, token_similarities)


def main(config):
    # --- Configuration ---
    output_folder = config.get("folder", "./mode_collapse_eval/")
    tag = config.get("tag", "mode_collapse")
    num_workers = config.get("num_workers", 8)
    batch_size = config.get("batch_size", 20)
    use_bfloat16 = config.get("use_bfloat16", False)
    seed = config.get("seed", 0)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Model
    model_cfg = config.get("model_kwargs", {})
    checkpoint = model_cfg.get("checkpoint")
    model_name = model_cfg.get("model_name", "mae_vit_base_patch16")
    model_kwargs = model_cfg.get("pretrain_kwargs", {})

    # Data params
    data_cfg = config.get("data", {})
    resolution = data_cfg.get("resolution", 224)
    frames_per_clip = data_cfg.get("frames_per_clip", 32)

    # Datasets to evaluate
    datasets_cfg = config.get("datasets", [])

    # --- Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load encoder ---
    encoder = load_encoder(model_name, checkpoint, device, model_kwargs)

    # --- Output directory ---
    out_dir = os.path.join(output_folder, tag)
    os.makedirs(out_dir, exist_ok=True)

    # --- Evaluate each dataset ---
    all_results = {}
    for ds_cfg in datasets_cfg:
        ds_name = ds_cfg.get("name", "unknown")
        ds_path = ds_cfg.get("path")

        if not ds_path or not os.path.exists(ds_path):
            logger.warning(f"Skipping '{ds_name}': path not found ({ds_path})")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating dataset: {ds_name}")
        logger.info(f"  Path: {ds_path}")
        logger.info(f"{'='*50}")

        if ds_path.endswith(".csv"):
            dataset = VideoDatasetFromCSV(ds_path, resolution=resolution, num_frames=frames_per_clip)
        else:
            dataset = EchoDataset_from_Video_mp4(ds_path, image_size=[resolution, resolution])

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        metrics = evaluate_dataset(encoder, loader, device, use_bfloat16, ds_name)

        logger.info(f"\n--- Mode Collapse Report [{ds_name}] ---")
        logger.info(f"  Intra-Video Temporal Similarity: {metrics['temporal_similarity']:.4f}")
        logger.info(f"  Active Dimensions:               {metrics['active_channels']}")
        logger.info(f"  Dims for 90% Variance:           {metrics['dim_90_variance']}")
        logger.info(f"  Inter-Video Global Similarity:    {metrics['global_cosine_similarity']:.4f}")

        all_results[ds_name] = metrics

    # --- Save results ---
    import json
    results_path = os.path.join(out_dir, "collapse_metrics.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mode collapse evaluation for EchoFM")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
