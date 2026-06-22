"""
Unified inference script for EchoFM frozen-encoder probe evaluation.

This script:
1. Loads a frozen EchoFM encoder (MAE ViT) from a pretrained checkpoint
2. Loads trained probe classifier(s) from an eval checkpoint
3. Runs inference on a test dataset (CSV or folder-based)
4. Saves per-sample predictions to a unified_predictions.json file

JSON output format (one entry per video):
  {
    "subject_id": "patient_001" or null,
    "video_id": "path/to/video.mp4",
    "true_label": 2,
    "probs": [0.01, 0.02, 0.95, 0.02]
  }
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attentive_probe import AttentiveClassifier
from linear_pooler import LinearClassifier, MLPClassifier
from EchoFM.models_mae import mae_vit_base_patch16, mae_vit_large_patch16
from EchoFM.util.multiclip import MulticlipEncoder
from data.dataset import VideoFrameTransform, MulticlipVideoDataset, default_multiclip_collate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EchoDatasetWithLabels(torch.utils.data.Dataset):
    """
    Dataset loader for Echo videos with CSV-based labels.

    CSV format: patient_id, video_path, class
    Returns (video_tensor, label, patient_id, video_path) per sample.
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
            if any(keyword in first_row for keyword in
                   {"patient", "video", "path", "filepath", "filename", "sample", "class", "label", "target"}):
                rows = rows[1:]

        self.data_rows = rows

        if video_folder is None:
            self.video_folder = os.path.dirname(os.path.abspath(csv_path))
        else:
            self.video_folder = str(video_folder)

        # Lazy-import the mp4 dataset (avoids heavy imports at module level)
        from data.dataset import EchoDataset_from_Video_mp4
        self.dataset = EchoDataset_from_Video_mp4(
            self.video_folder,
            image_size=[resolution, resolution],
            channels=3,
        )

        self.video_paths = []
        self.labels = []
        self.patient_ids = []

        for row in self.data_rows:
            patient_id = str(row[0]).strip()
            video_path = str(row[1]).strip()
            label = int(float(row[2])) if len(row) > 2 else 0

            if not os.path.isabs(video_path):
                full_path = os.path.join(self.video_folder, video_path)
            else:
                full_path = video_path

            self.video_paths.append(full_path)
            self.labels.append(label)
            self.patient_ids.append(patient_id)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]

        video_tensor = self.dataset.mp4_to_tensor(str(video_path))
        video_tensor = self.dataset.cast_num_frames_fn(video_tensor)

        return video_tensor, label, patient_id, video_path


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------

def load_encoder(model_name, checkpoint, device, model_kwargs):
    """Load and initialise the EchoFM MAE encoder (frozen, eval mode)."""
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
        raise ValueError(f"Unknown model name: {model_name}")

    encoder = encoder.to(device)

    if checkpoint and os.path.exists(checkpoint):
        logger.info(f"Loading pretrained weights from: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)

        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # Strip "module." prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Encoder loaded (missing/unexpected keys): {msg}")
    else:
        logger.warning("No encoder checkpoint provided, using random weights")

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    return encoder


# ---------------------------------------------------------------------------
# Probe loading
# ---------------------------------------------------------------------------

def load_probes(config_dict, probe_checkpoint_path, device):
    """
    Build probe classifiers from config and load weights from a training checkpoint.

    Returns:
        classifiers (list), num_classifier_heads, best_head_idx
    """
    args_exp = config_dict.get("experiment", {})
    args_classifier = args_exp.get("classifier", {})
    args_data = args_exp.get("data", {})

    probe_type = args_classifier.get("probe_type", "attentive")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)
    use_layernorm = args_classifier.get("use_layernorm", True)
    probe_dropout = args_classifier.get("dropout", 0.0)
    num_classes = args_data.get("num_classes", 2)

    # Embed dim detection - we'll set it after encoder is loaded, but we can
    # store the factory and reconstruct after we know embed_dim.
    # For now we store a callable; embed_dim will be filled in later.
    checkpoint_dict = torch.load(probe_checkpoint_path, map_location=device)

    # Determine number of heads from saved checkpoint
    classifiers_ckpt = checkpoint_dict.get("classifiers", [])
    num_classifier_heads = len(classifiers_ckpt)

    # Determine best head
    best_head_idx = 0
    if "best_val_acc_per_head" in checkpoint_dict:
        best_val_accs = checkpoint_dict["best_val_acc_per_head"]
        best_head_idx = int(np.argmax(best_val_accs))
        logger.info(
            f"Found {num_classifier_heads} classifier heads. "
            f"Best head (highest val acc): {best_head_idx} "
            f"with acc={best_val_accs[best_head_idx]:.4f}"
        )
    else:
        logger.info(
            f"Found {num_classifier_heads} classifier head(s). "
            f"Using head 0 (best_val_acc_per_head not in checkpoint)"
        )

    # Infer embed_dim from the first classifier's linear layer weight
    state0 = classifiers_ckpt[0]
    # Find the linear weight
    embed_dim = None
    for k, v in state0.items():
        if "linear.weight" in k or "regressor.weight" in k:
            embed_dim = v.shape[1]
            break
    if embed_dim is None:
        raise RuntimeError("Could not infer embed_dim from classifier checkpoint")

    # Build classifiers
    classifiers = []
    for head_idx in range(num_classifier_heads):
        if probe_type == "linear":
            clf = LinearClassifier(
                embed_dim=embed_dim,
                num_classes=num_classes,
                use_layernorm=use_layernorm,
                dropout=probe_dropout,
            ).to(device)
        elif probe_type == "mlp":
            clf = MLPClassifier(
                embed_dim=embed_dim,
                num_classes=num_classes,
                use_layernorm=use_layernorm,
                dropout=probe_dropout,
            ).to(device)
        else:  # attentive
            clf = AttentiveClassifier(
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=num_probe_blocks,
                num_classes=num_classes,
                use_activation_checkpointing=False,  # not needed for inference
            ).to(device)

        clf.load_state_dict(classifiers_ckpt[head_idx])
        clf.eval()
        classifiers.append(clf)

    logger.info(f"Loaded {len(classifiers)} classifier head(s) from {probe_checkpoint_path}")
    return classifiers, num_classifier_heads, best_head_idx


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(encoder, classifier, data_loader, device):
    """
    Run inference over the full dataset and collect per-sample predictions.

    Returns:
        list of dicts with keys: subject_id, video_id, true_label, probs
    """
    encoder.eval()
    classifier.eval()

    unified_predictions = []
    total_samples = len(data_loader.dataset)
    processed = 0

    logger.info(f"Running inference on {total_samples} samples...")

    # Folder-based datasets return just a tensor (no labels, no IDs)
    folder_based = isinstance(data_loader.dataset, torch.utils.data.Dataset) and \
                   not hasattr(data_loader.dataset, 'labels')

    with torch.no_grad():
        for batch in data_loader:
            # Batch format varies:
            #   EchoDatasetWithLabels: (clips, labels, patient_ids, video_paths)
            #   EchoDataset_from_Video_mp4: just a tensor
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                clips = batch[0]
                labels = batch[1]
                patient_ids = batch[2] if len(batch) > 2 else [None] * len(labels)
                video_paths = batch[3] if len(batch) > 3 else [None] * len(labels)
            else:
                clips = batch
                labels = torch.zeros(clips.shape[0], dtype=torch.long)
                patient_ids = [None] * clips.shape[0]
                video_paths = [None] * clips.shape[0]

            labels = labels.to(device, non_blocking=True)

            # Handle single-clip vs multiclip batch format
            if isinstance(clips, (list, tuple)) and all(isinstance(c, torch.Tensor) for c in clips):
                # Multiclip mode: clips is a list of [B, C, T, H, W] tensors
                batch_size = clips[0].shape[0]
                clips = [c.to(device, non_blocking=True) for c in clips]
            else:
                # Single-clip mode: clips is [B, C, T, H, W]
                clips = clips.to(device, non_blocking=True)
                batch_size = clips.shape[0]

            # Normalise patient IDs
            normalized_pids = _normalize_patient_ids(patient_ids, batch_size)

            # Forward through frozen encoder
            if isinstance(clips, (list, tuple)):
                # Multiclip: encoder wrapper handles the list
                encoder_output = encoder(clips)
                if isinstance(encoder_output, (tuple, list)):
                    encoder_output = encoder_output[0]
            else:
                # Single-clip: standard forward
                encoder_output = encoder.forward_encoder(clips, 0)
                # MAE encoder output is (x, mask, ids_restore); x is (B, N_patches, embed_dim)
                if isinstance(encoder_output, (tuple, list)):
                    encoder_output = encoder_output[0]

            # Ensure shape is (B, N, D) for the probe
            if encoder_output.dim() == 4:
                encoder_output = encoder_output.mean(dim=(2, 3))
            elif encoder_output.dim() == 2:
                encoder_output = encoder_output.unsqueeze(1)

            # Apply classifier and extract probabilities
            logits = classifier(encoder_output)
            probs = F.softmax(logits, dim=1)

            for i in range(batch_size):
                vid_path = video_paths[i] if isinstance(video_paths, (list, tuple)) else None
                if isinstance(vid_path, torch.Tensor):
                    vid_path = vid_path.item() if vid_path.numel() == 1 else str(vid_path)
                unified_predictions.append({
                    'subject_id': normalized_pids[i],
                    'video_id': str(vid_path) if vid_path is not None else None,
                    'true_label': labels[i].cpu().item(),
                    'probs': probs[i].detach().cpu().numpy().tolist(),
                })

            processed += batch_size
            if total_samples > 0:
                pct = min(processed, total_samples) / total_samples * 100
                if processed % max(1, (total_samples // 10)) < batch_size or processed >= total_samples:
                    logger.info(f"Processed {processed}/{total_samples} samples ({pct:.0f}%)")

    return unified_predictions


def _normalize_patient_ids(patient_ids, batch_size):
    """Normalise patient IDs to string or None."""
    if patient_ids is None:
        return [None] * batch_size
    if isinstance(patient_ids, torch.Tensor):
        patient_ids = patient_ids.detach().cpu().tolist()
    elif isinstance(patient_ids, np.ndarray):
        patient_ids = patient_ids.tolist()
    elif not isinstance(patient_ids, (list, tuple)):
        patient_ids = [patient_ids]

    normalized = []
    for pid in patient_ids:
        if pid is None:
            normalized.append(None)
            continue
        if isinstance(pid, float) and np.isnan(pid):
            normalized.append(None)
            continue
        pid_str = str(pid).strip()
        normalized.append(pid_str if pid_str and pid_str.lower() not in {"none", "nan"} else None)

    if len(normalized) < batch_size:
        normalized.extend([None] * (batch_size - len(normalized)))
    return normalized[:batch_size]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_multiclip_dataset(csv_path, num_clips, frames_per_clip, frame_step,
                            resolution, random_clip_sampling=True,
                            allow_clip_overlap=False, filter_short_videos=False,
                            video_folder=None):
    """Build a MulticlipVideoDataset from a CSV label file."""
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
    if rows:
        first_row = [str(v).strip().lower() for v in rows[0][:3]]
        if any(k in first_row for k in
               {"patient", "video", "path", "filepath", "filename", "sample", "class", "label", "target"}):
            rows = rows[1:]

    if video_folder is None:
        video_folder = os.path.dirname(os.path.abspath(csv_path))
    else:
        video_folder = str(video_folder)

    samples = []
    for row in rows:
        patient_id = str(row[0]).strip()
        video_path = str(row[1]).strip()
        label = int(float(row[2])) if len(row) > 2 else 0
        if not os.path.isabs(video_path):
            video_path = os.path.join(video_folder, video_path)
        samples.append((video_path, label, patient_id))

    transform = VideoFrameTransform(resolution, use_augmentation=False)
    return MulticlipVideoDataset(
        samples=samples,
        transform=transform,
        num_clips=num_clips,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a frozen EchoFM encoder + trained probe, "
                    "and export unified_predictions.json"
    )
    parser.add_argument(
        "--eval_config", required=True,
        help="Path to YAML config (same format as eval_attentive_probe.py)"
    )
    parser.add_argument(
        "--probe_checkpoint", required=True,
        help="Path to probe checkpoint (.pt) produced by eval_attentive_probe.py"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Directory to save unified_predictions.json"
    )
    parser.add_argument(
        "--test_data_path", default=None,
        help="Optional override for the test/val dataset path (CSV or folder)"
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="GPU device index (default: 0)"
    )
    parser.add_argument(
        "--head_idx", type=int, default=None,
        help="Override which classifier head to use (default: best from checkpoint)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Inference batch size (default: 32)"
    )
    args = parser.parse_args()

    # --- Device ---
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Load config ---
    import yaml
    with open(args.eval_config, 'r') as f:
        config = yaml.safe_load(f)

    # --- Load encoder ---
    model_kwargs_section = config.get("model_kwargs", {})
    model_name = model_kwargs_section.get("model_name", "mae_vit_large_patch16")
    checkpoint = model_kwargs_section.get("checkpoint", None)
    pretrain_kwargs = model_kwargs_section.get("pretrain_kwargs", {})

    encoder = load_encoder(model_name, checkpoint, device, pretrain_kwargs)

    # --- Load probes ---
    classifiers, num_heads, best_head_idx = load_probes(config, args.probe_checkpoint, device)

    # --- Choose head ---
    effective_head_idx = best_head_idx
    if args.head_idx is not None:
        logger.info(f"Overriding head index: using head {args.head_idx}")
        effective_head_idx = args.head_idx
    if effective_head_idx < 0 or effective_head_idx >= len(classifiers):
        logger.warning(f"head_idx {effective_head_idx} out of bounds, falling back to 0")
        effective_head_idx = 0
    classifier = classifiers[effective_head_idx]
    logger.info(f"Using classifier head {effective_head_idx}")

    # --- Prepare test data ---
    args_data = config.get("experiment", {}).get("data", {})
    resolution = args_data.get("resolution", 224)
    frames_per_clip = args_data.get("frames_per_clip", 32)
    num_clips = args_data.get("num_clips", 1)
    frame_step = args_data.get("frame_step", 1)
    allow_clip_overlap = args_data.get("allow_clip_overlap", False)
    random_clip_sampling = args_data.get("random_clip_sampling", False)  # deterministic for eval
    filter_short_videos = args_data.get("filter_short_videos", False)

    test_data_path = args.test_data_path or args_data.get("dataset_val", "")
    if not test_data_path:
        raise ValueError("No test data path provided (set --test_data_path or dataset_val in config)")

    logger.info(f"Loading test data from: {test_data_path}")

    if test_data_path.endswith('.csv'):
        if num_clips > 1:
            test_dataset = build_multiclip_dataset(
                test_data_path,
                num_clips=num_clips,
                frames_per_clip=frames_per_clip,
                frame_step=frame_step,
                resolution=resolution,
                random_clip_sampling=random_clip_sampling,
                allow_clip_overlap=allow_clip_overlap,
                filter_short_videos=filter_short_videos,
            )
            collate_fn = default_multiclip_collate
            # Wrap encoder for multiclip
            use_temporal_pos_embed = args_data.get("use_temporal_pos_embed", False)
            max_frames = args_data.get("max_frames", 256)
            encoder = MulticlipEncoder(
                encoder,
                use_temporal_pos_embed=use_temporal_pos_embed,
                max_frames=max_frames,
            ).to(device)
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            logger.info(f"Wrapped encoder in MulticlipEncoder (num_clips={num_clips})")
        else:
            test_dataset = EchoDatasetWithLabels(
                test_data_path,
                resolution=resolution,
                num_frames=frames_per_clip,
            )
            collate_fn = None
    else:
        # Folder-based dataset (no labels, no patient IDs)
        from data.dataset import EchoDataset_from_Video_mp4
        test_dataset = EchoDataset_from_Video_mp4(
            test_data_path,
            image_size=[resolution, resolution],
        )
        collate_fn = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    logger.info(f"Test dataset size: {len(test_dataset)} samples")

    # --- Run inference ---
    predictions = run_inference(encoder, classifier, test_loader, device)

    # --- Save ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "unified_predictions.json"
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)

    logger.info(f"Saved {len(predictions)} predictions to {output_file}")

    # Quick stats
    if predictions:
        correct = sum(
            1 for p in predictions
            if p['probs'] and np.argmax(p['probs']) == p['true_label']
        )
        acc = 100.0 * correct / len(predictions)
        logger.info(f"Dataset-level accuracy (chosen head {effective_head_idx}): {acc:.2f}%")


if __name__ == "__main__":
    main()
