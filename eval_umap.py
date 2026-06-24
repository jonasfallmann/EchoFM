#!/usr/bin/env python3
"""
UMAP projection of frozen EchoFM encoder features for video samples.

Extracts latent representations directly from the pretrained EchoFM model
(bypassing probes), computes UMAP, and plots coloured by diagnosis/class.

Usage:
    python eval_umap.py --config config/echo_probe_config.yaml
    python eval_umap.py --config config/echo_probe_config.yaml \\
        --checkpoint /path/to/model.pt --probe-layer -1

Config overrides (same as eval_attentive_probe.py):
    --checkpoint /path/to/other.pt
    --model_name mae_vit_base_patch16
    --batch_size 8
    --output-dir /path/to/output

UMAP overrides:
    --umap-n-neighbors 15
    --umap-min-dist 0.05
    --umap-metric euclidean

By default, extracts features from datasets listed in config key `umap.sets`
(a dict mapping set names -> CSV paths). Falls back to `experiment.data.dataset_val`
when no `umap.sets` is provided.
"""

import argparse
import logging
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import umap

from data.dataset import (
    EchoDataset_from_Video_mp4,
    VideoFrameTransform,
    MulticlipVideoDataset,
    default_multiclip_collate,
)
from EchoFM.models_mae import mae_vit_base_patch16, mae_vit_large_patch16
from EchoFM.util.multiclip import MulticlipEncoder

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
#  Dataset helper  (mirrors EchoDatasetWithLabels from eval_attentive_probe.py)
# ---------------------------------------------------------------------------

class EchoDatasetWithLabels(torch.utils.data.Dataset):
    """Dataset loader for Echo videos with CSV-based labels.
    CSV format: patient_id,video_path,label"""

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
            if any(k in first_row for k in
                   {"patient", "video", "path", "filepath",
                    "filename", "sample", "class", "label", "target"}):
                rows = rows[1:]

        self.data_rows = rows

        if video_folder is None:
            video_folder = os.path.dirname(os.path.abspath(csv_path))
        self.video_folder = str(video_folder)

        # Build sample list: (full_path, label, patient_id)
        self.samples = []
        for row in self.data_rows:
            patient_id = str(row[0]).strip()
            video_path = str(row[1]).strip()
            label = int(float(row[2])) if len(row) > 2 else 0

            if not os.path.isabs(video_path):
                full_path = os.path.join(self.video_folder, video_path)
            else:
                full_path = video_path

            self.samples.append((full_path, label, patient_id))

        # Use EchoDataset_from_Video_mp4 for video reading + transforms
        self.dataset = EchoDataset_from_Video_mp4(
            self.video_folder,
            image_size=[resolution, resolution],
            channels=3,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, patient_id = self.samples[idx]
        video_tensor = self.dataset.mp4_to_tensor(str(video_path))
        video_tensor = self.dataset.cast_num_frames_fn(video_tensor)
        return video_tensor, label, patient_id


# ---------------------------------------------------------------------------
#  Build multiclip dataset from CSV  (mirrors eval_attentive_probe.py)
# ---------------------------------------------------------------------------

def build_multiclip_dataset(
    csv_path, video_folder=None, transform=None,
    num_clips=4, frames_per_clip=32, frame_step=1,
    random_clip_sampling=True, allow_clip_overlap=False,
    filter_short_videos=False,
):
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
               {"patient", "video", "path", "filepath",
                "filename", "sample", "class", "label", "target"}):
            rows = rows[1:]

    if video_folder is None:
        video_folder = os.path.dirname(os.path.abspath(csv_path))

    samples = []
    for row in rows:
        patient_id = str(row[0]).strip()
        video_path = str(row[1]).strip()
        label = int(float(row[2])) if len(row) > 2 else 0
        if not os.path.isabs(video_path):
            video_path = os.path.join(video_folder, video_path)
        samples.append((video_path, label, patient_id))

    if transform is None:
        transform = VideoFrameTransform(224, use_augmentation=False)

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


# ---------------------------------------------------------------------------
#  Encoder loading  (mirrors load_encoder from eval_attentive_probe.py)
# ---------------------------------------------------------------------------

def load_encoder(model_name, checkpoint, device, model_kwargs, probe_layer=-1):
    """Load and initialize the frozen EchoFM encoder."""
    logger.info(f"Loading encoder: {model_name}")

    encoder_kwargs = {
        "img_size": 224,
        "in_chans": 3,
        "decoder_embed_dim": 512,
        "decoder_depth": 8,
        "decoder_num_heads": 16,
        "norm_pix_loss": False,
        "num_frames": 32,
        "t_patch_size": 4,
    }
    encoder_kwargs.update(model_kwargs)

    if model_name == "mae_vit_base_patch16":
        encoder = mae_vit_base_patch16(**encoder_kwargs)
    elif model_name == "mae_vit_large_patch16":
        encoder = mae_vit_large_patch16(**encoder_kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    encoder = encoder.to(device)

    # -- Layer selection for probing ---
    num_layers = len(encoder.blocks)
    logger.info(f"Encoder has {num_layers} transformer blocks")

    effective_layer = num_layers + probe_layer if probe_layer < 0 else probe_layer
    if effective_layer < 0 or effective_layer >= num_layers:
        logger.warning(
            f"probe_layer={probe_layer} out of range [{-num_layers}, {num_layers - 1}], "
            f"falling back to last layer ({num_layers - 1})"
        )
        effective_layer = num_layers - 1
    logger.info(f"Probing at layer {effective_layer} (0-indexed) of {num_layers}")

    encoder.probe_layer = effective_layer
    encoder.num_layers = num_layers

    # -- Load checkpoint ---
    if checkpoint and os.path.exists(checkpoint):
        logger.info(f"Loading checkpoint from {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = encoder.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded encoder with message: {msg}")
    elif checkpoint:
        logger.warning(f"Checkpoint not found: {checkpoint}")
    else:
        logger.warning("No checkpoint provided, using random initialization")

    return encoder


# ---------------------------------------------------------------------------
#  Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features_singleclip(
    encoder: torch.nn.Module,
    data_loader,
    device: torch.device,
    use_bfloat16: bool = True,
) -> dict:
    """
    Extract features for single-clip videos (num_clips = 1).

    Calls encoder.forward_encoder_layer(clips, mask_ratio=0, probe_layer)
    and mean-pools across tokens to get one D-dimensional vector per video.
    """
    all_embeddings = []
    all_labels = []
    all_patient_ids = []

    probe_layer = getattr(encoder, "probe_layer", -1)

    for batch in tqdm(data_loader, desc="Extracting features", unit="batch"):

        if isinstance(batch, (tuple, list)):
            if len(batch) == 3:
                clips, labels, patient_ids = batch
            elif len(batch) == 2:
                clips, labels = batch
                patient_ids = None
            else:
                clips = batch[0]
                labels = batch[1]
                patient_ids = batch[2] if len(batch) > 2 else None
        else:
            logger.warning("Unexpected batch format, skipping")
            continue

        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=use_bfloat16 and torch.cuda.is_available(),
        ):
            out = encoder.forward_encoder_layer(clips, mask_ratio=0.0,
                                                 layer_idx=probe_layer)
            if isinstance(out, (tuple, list)):
                out = out[0]  # tokens [B, N, D]

        pooled = out.mean(dim=1)  # [B, D]

        for i in range(clips.shape[0]):
            all_embeddings.append(pooled[i].cpu().float().numpy())
            all_labels.append(labels[i].item())
            if patient_ids is not None:
                pid = patient_ids[i]
                if isinstance(pid, torch.Tensor):
                    pid = pid.item()
                all_patient_ids.append(str(pid) if pid is not None else None)
            else:
                all_patient_ids.append(None)

    logger.info(f"Extracted {len(all_embeddings)} video embeddings "
                f"(dim={all_embeddings[0].shape[0]})")
    return {
        "embeddings": np.stack(all_embeddings, axis=0),
        "labels": np.array(all_labels),
        "patient_ids": all_patient_ids,
    }


@torch.no_grad()
def extract_features_multiclip(
    encoder: torch.nn.Module,
    data_loader,
    device: torch.device,
    use_bfloat16: bool = True,
) -> dict:
    """
    Extract features for multi-clip videos (num_clips > 1).

    MulticlipEncoder is already wrapped around the encoder externally.
    Calls encoder(clips, clip_indices) and mean-pools tokens.
    """
    all_embeddings = []
    all_labels = []
    all_patient_ids = []

    # Buffer for multi-clip videos: index → list of per-clip pooled tensors
    clip_buffer: dict[int, list[torch.Tensor]] = defaultdict(list)
    label_buffer: dict[int, int] = {}
    pid_buffer: dict[int, str | None] = {}
    global_idx = 0

    for batch in tqdm(data_loader, desc="Extracting features", unit="batch"):
        if isinstance(batch, (tuple, list)) and len(batch) >= 4:
            clips, labels, clip_indices, patient_ids = batch
        elif isinstance(batch, (tuple, list)) and len(batch) == 3:
            clips, labels, patient_ids = batch
            clip_indices = None
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            clips, labels = batch
            clip_indices = None
            patient_ids = None
        else:
            logger.warning("Unexpected batch format, skipping")
            continue

        if clip_indices is not None and isinstance(clip_indices[0], torch.Tensor):
            clip_indices = [ci.to(device, non_blocking=True) for ci in clip_indices]
        if isinstance(clips, (list, tuple)):
            clips = [c.to(device, non_blocking=True) for c in clips]
        else:
            clips = clips.to(device, non_blocking=True)

        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16,
            enabled=use_bfloat16 and torch.cuda.is_available(),
        ):
            out = encoder(clips, clip_indices=clip_indices)
            if isinstance(out, (tuple, list)):
                out = out[0]  # tokens [B, nc*N, D]

        # Average across clips for each video (since multiclip encoder
        # concatenates tokens across clips, we get [B, nc*N, D] and can
        # mean-pool directly to [B, D])
        pooled = out.mean(dim=1)  # [B, D] – simple mean-pool across all tokens

        batch_size = pooled.shape[0]
        for i in range(batch_size):
            clip_buffer[global_idx].append(pooled[i].cpu().float())
            label_buffer[global_idx] = labels[i].item()
            if patient_ids is not None:
                pid = patient_ids[i]
                if isinstance(pid, torch.Tensor):
                    pid = pid.item()
                pid_buffer[global_idx] = str(pid) if pid is not None else None
            else:
                pid_buffer[global_idx] = None
            global_idx += 1

    # Average across clips per video (for multi-video datasets where each video
    # may appear in multiple batches, or if the dataloader repeats)
    for idx in sorted(clip_buffer.keys()):
        stacked = torch.stack(clip_buffer[idx])  # [num_occurrences, D]
        all_embeddings.append(stacked.mean(dim=0).numpy())
        all_labels.append(label_buffer[idx])
        all_patient_ids.append(pid_buffer[idx])

    logger.info(f"Extracted {len(all_embeddings)} video embeddings "
                f"(dim={all_embeddings[0].shape[0]})")
    return {
        "embeddings": np.stack(all_embeddings, axis=0),
        "labels": np.array(all_labels),
        "patient_ids": all_patient_ids,
    }


# ---------------------------------------------------------------------------
#  UMAP + Plot
# ---------------------------------------------------------------------------

def compute_and_plot(
    all_features: dict,
    class_names: list,
    output_dir: str,
    umap_kwargs: dict | None = None,
):
    """
    all_features:  {set_name: {"embeddings": [N,D], "labels": [N]}, ...}
    class_names:   label-index -> human-readable name
    """
    if umap_kwargs is None:
        umap_kwargs = dict(
            n_neighbors=30, min_dist=0.1, n_components=2,
            metric="cosine", random_state=42,
        )

    set_labels = sorted(all_features.keys())

    # -- collect all points ---
    all_emb = []
    all_lbl = []
    all_set = []
    for sname in set_labels:
        feats = all_features[sname]
        all_emb.append(feats["embeddings"])
        all_lbl.append(feats["labels"])
        all_set.append(np.full(len(feats["labels"]), sname, dtype=object))

    X = np.concatenate(all_emb, axis=0)
    y = np.concatenate(all_lbl, axis=0)
    sets = np.concatenate(all_set, axis=0)

    logger.info(f"Total samples for UMAP: {len(X)} (dim={X.shape[1]})")
    logger.info(f"UMAP kwargs: {umap_kwargs}")

    # -- run UMAP ---
    reducer = umap.UMAP(**umap_kwargs)
    embedding_2d = reducer.fit_transform(X)

    unique = np.unique(y)
    n_classes = len(unique)

    # Custom class colours (fall back to tab20 for extra classes)
    custom_colours = ["#27AE60", "#F1C40F", "#E67E22", "#E74C3C"]
    cmap = matplotlib.colormaps.get_cmap("tab20")
    colours = [
        custom_colours[i] if i < len(custom_colours) else cmap(i % 20)
        for i in range(n_classes)
    ]

    # -------- Figure 1: two panels (class + split) --------
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    set_markers = {"train": "o", "val": "s", "test": "^"}

    # Panel 1: colour by class, marker by set
    ax = axes[0]
    for lbl in unique:
        for sname in set_labels:
            mask = (y == lbl) & (sets == sname)
            if not mask.any():
                continue
            name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
            ax.scatter(
                embedding_2d[mask, 0], embedding_2d[mask, 1],
                c=[colours[lbl]], marker=set_markers.get(sname, "o"),
                s=25, alpha=0.7, edgecolors="none",
                label=f"{name} ({sname})",
            )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=12, markerscale=1.5)

    # Panel 2: colour by set
    ax = axes[1]
    set_colours = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    for sname in set_labels:
        mask = sets == sname
        if not mask.any():
            continue
        ax.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            c=set_colours.get(sname, "gray"), marker="o",
            s=25, alpha=0.7, edgecolors="none", label=sname,
        )
    ax.legend(fontsize=12)

    plt.tight_layout()
    path1 = os.path.join(output_dir, "umap_projection.png")
    fig.savefig(path1, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {path1}")
    plt.close(fig)

    # -------- Figure 2: class-only --------
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    for lbl in unique:
        mask = y == lbl
        if not mask.any():
            continue
        name = class_names[lbl] if lbl < len(class_names) else f"Class {lbl}"
        ax2.scatter(
            embedding_2d[mask, 0], embedding_2d[mask, 1],
            c=[colours[lbl]], s=50, alpha=0.9, edgecolors="none",
            label=name,
        )
    ax2.legend(fontsize=14)
    plt.tight_layout()

    path2 = os.path.join(output_dir, "umap_projection_classes.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    logger.info(f"Saved {path2}")
    plt.close(fig2)

    # -------- Save raw data --------
    np.savez(
        os.path.join(output_dir, "umap_data.npz"),
        embedding_2d=embedding_2d,
        labels=y,
        sets=sets,
        class_names=np.array(class_names[:n_classes]),
    )
    logger.info(f"Saved umap_data.npz to {output_dir}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="UMAP projection of frozen EchoFM encoder features")
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config (same format as eval_attentive_probe.py)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Override model checkpoint path")
    p.add_argument("--model_name", type=str, default=None,
                   help="Override encoder model name")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size")
    p.add_argument("--probe-layer", type=int, default=None,
                   help="Override probe layer index (negative = from end, e.g. -1)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Override output directory")
    p.add_argument("--umap-n-neighbors", type=int, default=None,
                   help="Override UMAP n_neighbors")
    p.add_argument("--umap-min-dist", type=float, default=None,
                   help="Override UMAP min_dist")
    p.add_argument("--umap-metric", type=str, default=None,
                   help="Override UMAP metric (e.g. cosine, euclidean)")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--no-bfloat16", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # -- Load config ---
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # -- Resolve sections ---
    pretrain = cfg.get("model_kwargs", {})
    exp = cfg.get("experiment", {})
    data_cfg = exp.get("data", {})
    opt_cfg = exp.get("optimization", {})
    umap_cfg = cfg.get("umap", {})

    # --- Encoder params ---
    model_name = args.model_name or pretrain.get("model_name", "mae_vit_large_patch16")
    checkpoint = args.checkpoint or pretrain.get("checkpoint", None)
    pretrain_kwargs = pretrain.get("pretrain_kwargs", {})
    probe_layer = args.probe_layer if args.probe_layer is not None else pretrain.get("probe_layer", -1)

    # --- Data params ---
    resolution = data_cfg.get("resolution", 224)
    frames_per_clip = data_cfg.get("frames_per_clip", 32)
    frame_step = data_cfg.get("frame_step", 1)
    num_clips = data_cfg.get("num_clips", 1)
    allow_clip_overlap = data_cfg.get("allow_clip_overlap", False)
    random_clip_sampling = data_cfg.get("random_clip_sampling", True)
    filter_short_videos = data_cfg.get("filter_short_videos", False)
    use_temporal_pos_embed = data_cfg.get("use_temporal_pos_embed", False)
    max_frames = data_cfg.get("max_frames", 256)
    num_classes = data_cfg.get("num_classes", 2)
    batch_size = args.batch_size or opt_cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 8)
    use_bfloat16 = opt_cfg.get("use_bfloat16", False) and not args.no_bfloat16

    # --- UMAP params ---
    class_names = umap_cfg.get("class_names",
                               [f"Class {i}" for i in range(num_classes)])
    set_paths = umap_cfg.get("sets", {})
    if not set_paths:
        # fall back to experiment.data val set
        val_path = data_cfg.get("dataset_val")
        if val_path:
            set_paths = {"val": val_path}
        else:
            logger.error("No datasets specified. Add a 'umap.sets' dict to your config.")
            sys.exit(1)

    output_dir = args.output_dir or cfg.get("folder", "./umap_output")
    os.makedirs(output_dir, exist_ok=True)

    umap_kwargs = {
        "n_neighbors": args.umap_n_neighbors or umap_cfg.get("n_neighbors", 30),
        "min_dist":     args.umap_min_dist or umap_cfg.get("min_dist", 0.1),
        "metric":       args.umap_metric or umap_cfg.get("metric", "cosine"),
        "n_components": 2,
        "random_state": 42,
    }

    # --- Device ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Encoder: {model_name} | checkpoint: {checkpoint}")
    logger.info(f"Probe layer: {probe_layer}")
    logger.info(f"Multiclip: {num_clips} clips, {frames_per_clip} frames/clip, "
                f"stride={frame_step}")
    logger.info(f"UMAP sets: {list(set_paths.keys())}")
    logger.info(f"UMAP kwargs: {umap_kwargs}")

    # -- Load encoder ---
    logger.info("Loading frozen encoder ...")
    encoder = load_encoder(
        model_name=model_name,
        checkpoint=checkpoint,
        device=device,
        model_kwargs=pretrain_kwargs,
        probe_layer=probe_layer,
    )
    encoder.eval()

    # -- Wrap in MulticlipEncoder if needed ---
    if num_clips > 1:
        encoder = MulticlipEncoder(
            encoder,
            use_temporal_pos_embed=use_temporal_pos_embed,
            max_frames=max_frames,
        ).to(device)
        logger.info(f"Wrapped encoder in MulticlipEncoder "
                    f"(temporal_pos_embed={use_temporal_pos_embed})")

    embed_dim = getattr(encoder, "embed_dim", None)
    if embed_dim:
        logger.info(f"Encoder embed_dim = {embed_dim}")

    # -- Extract features for each set ---
    all_features = {}
    transform = VideoFrameTransform(resolution, use_augmentation=False)

    for sname, csv_path in set_paths.items():
        logger.info(f"Loading {sname}  from  {csv_path}")

        if num_clips > 1:
            dataset = build_multiclip_dataset(
                csv_path=csv_path,
                transform=transform,
                num_clips=num_clips,
                frames_per_clip=frames_per_clip,
                frame_step=frame_step,
                random_clip_sampling=random_clip_sampling,
                allow_clip_overlap=allow_clip_overlap,
                filter_short_videos=filter_short_videos,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=default_multiclip_collate,
            )
            feats = extract_features_multiclip(
                encoder, loader, device, use_bfloat16=use_bfloat16,
            )
        else:
            dataset = EchoDatasetWithLabels(
                csv_path,
                resolution=resolution,
                num_frames=frames_per_clip,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
            feats = extract_features_singleclip(
                encoder, loader, device, use_bfloat16=use_bfloat16,
            )

        all_features[sname] = feats
        logger.info(f"  {sname}: {feats['embeddings'].shape[0]} samples, "
                    f"{len(np.unique(feats['labels']))} unique classes")

    # -- UMAP + plot ---
    compute_and_plot(all_features, class_names, output_dir, umap_kwargs)
    logger.info("Done.")


if __name__ == "__main__":
    main()
