# Copyright (c) 2024 EchoFM Contributors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import math
import pprint

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Dataset

from attentive_probe import AttentiveClassifier
from linear_pooler import LinearClassifier, MLPClassifier
from data.dataset import EchoDataset_from_Video_mp4
from EchoFM.models_mae import mae_vit_base_patch16, mae_vit_large_patch16

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


class EchoDatasetWithLabels(Dataset):
    """
    Dataset loader for Echo videos with CSV-based labels.
    CSV format: filename,label[,patient_id] or path,label[,patient_id]
    """
    def __init__(self, csv_path, video_folder=None, resolution=224, num_frames=32):
        self.resolution = resolution
        self.num_frames = num_frames
        csv_path = str(csv_path)

        # Load CSV file with comma delimiter: patient_id, video_path, class
        rows = []
        with open(csv_path, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                row = [part.strip() for part in line.split(",")]
                if row and len(row) >= 3:  # Ensure we have at least 3 columns
                    rows.append(row)

        # Skip header row if first row contains header keywords
        if rows:
            first_row = [str(v).strip().lower() for v in rows[0][:3]]
            if any(keyword in first_row for keyword in {"patient", "video", "path", "filepath", "filename", "sample", "class", "label", "target"}):
                rows = rows[1:]

        self.data_rows = rows

        # Infer video folder from CSV path if not provided
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

        # Create mapping of video files to indices and labels
        # CSV format: patient_id, video_path, class
        self.video_paths = []
        self.labels = []
        self.patient_ids = []

        for row in self.data_rows:
            patient_id = str(row[0]).strip()
            video_path = str(row[1]).strip()
            label = int(float(row[2])) if len(row) > 2 else 0

            # Handle both absolute and relative paths
            if not os.path.isabs(video_path):
                full_path = os.path.join(video_folder, video_path)
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

        # Load video using the underlying dataset
        video_tensor = self.dataset.mp4_to_tensor(str(video_path))
        video_tensor = self.dataset.cast_num_frames_fn(video_tensor)

        return video_tensor, label, patient_id


pp = pprint.PrettyPrinter(indent=4)


class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [B, C] logits
        # targets: [B] class indices
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def main(args_eval, resume_preempt=False):
    """
    Main training loop for attentive probe evaluation.

    Args:
        args_eval: Configuration dictionary with evaluation parameters
        resume_preempt: Whether to resume from preemption (for checkpoint restart)
    """

    # --- PARSE CONFIGURATION PARAMETERS ---
    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY MODE")

    # -- EXPERIMENT
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 8)

    # -- PRETRAIN MODEL CONFIGURATION
    args_pretrain = args_eval.get("model_kwargs", {})
    checkpoint = args_pretrain.get("checkpoint")
    model_name = args_pretrain.get("model_name", "mae_vit_base_patch16")
    args_model = args_pretrain.get("pretrain_kwargs", {})

    args_exp = args_eval.get("experiment", {})

    # -- PROBE CONFIGURATION
    args_classifier = args_exp.get("classifier", {})
    probe_type = args_classifier.get("probe_type", "attentive")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 12)
    use_layernorm = args_classifier.get("use_layernorm", True)
    probe_dropout = args_classifier.get("dropout", 0.0)
    probe_layer = args_classifier.get("probe_layer", -1)

    # -- DATA CONFIGURATION
    args_data = args_exp.get("data", {})
    num_classes = args_data.get("num_classes", 2)
    train_data_path = args_data.get("dataset_train", "")
    val_data_path = args_data.get("dataset_val", "")
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 32)
    frame_step = args_data.get("frame_step", 1)

    # -- OPTIMIZATION CONFIGURATION
    args_opt = args_exp.get("optimization", {})
    batch_size = args_opt.get("batch_size", 32)
    num_epochs = args_opt.get("num_epochs", 20)
    use_bfloat16 = args_opt.get("use_bfloat16", False)

    multihead_kwargs = args_opt.get("multihead_kwargs", [{}])
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup", 0.0),
        )
        for kwargs in multihead_kwargs
    ]

    # --- INITIALIZE DISTRIBUTED TRAINING ---
    try:
        # Linux can use fork for DataLoader workers, which avoids strict pickling requirements.
        start_method = "spawn" if os.name == "nt" else "fork"
        mp.set_start_method(start_method)
    except RuntimeError:
        # Start method is already set by the parent process.
        pass

    cuda_device = args_eval.get("cuda_device", 0)
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{cuda_device}")
        torch.cuda.set_device(device)

    # Initialize distributed environment
    rank = 0
    world_size = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")

    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # --- SETUP LOGGING AND CHECKPOINTING ---
    folder = os.path.join(pretrain_folder, "attentive_probe_eval/") if pretrain_folder else "./attentive_probe_eval/"
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")

    # Make CSV logger
    if rank == 0:
        csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%.5f", "train_acc"),
            ("%.5f", "val_acc"),
            ("%.5f", "val_subject_acc"),
        )

    # --- INITIALIZE ENCODER MODEL ---
    logger.info(f"Loading encoder model: {model_name}")
    encoder = load_encoder(
        model_name=model_name,
        checkpoint=checkpoint,
        device=device,
        model_kwargs=args_model,
        probe_layer=probe_layer,
    )

    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    # Get embed dimension from encoder
    embed_dim = getattr(encoder, 'embed_dim', None)
    if embed_dim is None:
        # Try to infer from model
        embed_dim = 1024  # Default for ViT-base
        for name, param in encoder.named_parameters():
            if 'blocks' in name and 'attn.qkv' in name:
                # qkv has shape (3*embed_dim, embed_dim)
                embed_dim = param.shape[1]
                break

    logger.info(f"Encoder embed_dim: {embed_dim}")

    # --- INITIALIZE PROBES (MULTIPLE) ---
    classifiers = []
    for i, kwargs in enumerate(opt_kwargs):
        if probe_type == "attentive":
            classifier = AttentiveClassifier(
                embed_dim=embed_dim,
                num_heads=num_heads,
                depth=num_probe_blocks,
                num_classes=num_classes,
                use_activation_checkpointing=True,
            ).to(device)
        elif probe_type == "linear":
            classifier = LinearClassifier(
                embed_dim=embed_dim,
                num_classes=num_classes,
                use_layernorm=use_layernorm,
                dropout=probe_dropout,
            ).to(device)
        elif probe_type == "mlp":
            classifier = MLPClassifier(
                embed_dim=embed_dim,
                num_classes=num_classes,
                use_layernorm=use_layernorm,
                dropout=probe_dropout,
            ).to(device)
        else:
            raise ValueError(f"Unknown probe_type: {probe_type}. Expected 'attentive', 'linear', or 'mlp'.")

        # Wrap with DDP for multi-GPU
        if world_size > 1:
            classifier = DistributedDataParallel(classifier, static_graph=True)

        classifiers.append(classifier)
        if rank == 0:
            logger.info(f"Initialized {probe_type} probe {i+1}/{len(opt_kwargs)}")

    if rank == 0:
        logger.info(f"Sample probe architecture:\n{classifiers[0]}")

    # --- INITIALIZE DATA LOADERS ---
    train_sampler = None
    val_sampler = None

    if train_data_path and os.path.exists(train_data_path):
        if train_data_path.endswith('.csv'):
            train_dataset = EchoDatasetWithLabels(
                train_data_path,
                resolution=resolution,
                num_frames=frames_per_clip,
            )
        else:
            # Fallback to folder-based dataset
            train_dataset = EchoDataset_from_Video_mp4(train_data_path, image_size=[resolution, resolution])

        if world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
    else:
        logger.warning(f"Train data path not found: {train_data_path}")
        train_loader = None

    if val_data_path and os.path.exists(val_data_path):
        if val_data_path.endswith('.csv'):
            val_dataset = EchoDatasetWithLabels(
                val_data_path,
                resolution=resolution,
                num_frames=frames_per_clip,
            )
        else:
            # Fallback to folder-based dataset
            val_dataset = EchoDataset_from_Video_mp4(val_data_path, image_size=[resolution, resolution])

        if world_size > 1:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=False,
            )
    else:
        logger.warning(f"Val data path not found: {val_data_path}")
        val_loader = None

    if train_loader is None or val_loader is None:
        logger.error("Could not load training or validation data")
        return

    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")

    # --- INITIALIZE OPTIMIZER AND SCHEDULER ---
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    # --- LOAD CHECKPOINT IF RESUMING ---
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        # Adjust scheduler steps for resume
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    def save_checkpoint(
        epoch,
        mean_val_acc,
        best_val_acc,
        val_heads,
        best_per_head,
        mean_per_head,
        min_per_head,
        best_epoch_per_head,
        subject_val_acc,
        subject_heads,
        is_best=False,
    ):
        """Save checkpoint with per-epoch and best models."""
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else [None if s is None else s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
            "mean_val_acc": float(mean_val_acc),
            "best_val_acc": float(best_val_acc),
            "subject_val_acc": None if subject_val_acc is None else float(subject_val_acc),
            "val_acc_per_head": np.asarray(val_heads, dtype=float).tolist(),
            "best_val_acc_per_head": np.asarray(best_per_head, dtype=float).tolist(),
            "mean_val_acc_per_head": np.asarray(mean_per_head, dtype=float).tolist(),
            "min_val_acc_per_head": np.asarray(min_per_head, dtype=float).tolist(),
            "best_epoch_per_head": np.asarray(best_epoch_per_head, dtype=int).tolist(),
            "subject_val_acc_per_head": np.asarray(subject_heads, dtype=float).tolist() if subject_heads is not None else None,
        }

        if rank == 0:
            # Always save latest
            _latest_path = os.path.join(folder, "latest.pt")
            torch.save(save_dict, _latest_path)

            # Save per-epoch snapshot
            epoch_path = os.path.join(folder, f"epoch_{epoch:03d}.pt")
            torch.save(save_dict, epoch_path)

            # Save best checkpoint
            if is_best:
                best_path = os.path.join(folder, "best.pt")
                torch.save(save_dict, best_path)
                logger.info(f"Generated new best model: {best_path}")

    # --- INITIALIZE PER-HEAD TRACKING ---
    best_per_head = None
    sum_per_head = None
    min_per_head = None
    best_epoch_per_head = None
    count_epochs = 0
    best_val_acc = 0.0

    # --- MAIN TRAINING LOOP ---
    for epoch in range(start_epoch, num_epochs):
        logger.info("=" * 50)
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info("=" * 50)

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if val_only:
            train_acc = -1.0
        else:
            train_acc, _, _, _ = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
            )

        val_acc, val_heads, val_subject_acc, val_subject_heads = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
        )

        # --- UPDATE PER-HEAD RUNNING STATS ---
        count_epochs += 1
        if best_per_head is None:
            best_per_head = val_heads.copy()
            sum_per_head = val_heads.copy()
            min_per_head = val_heads.copy()
            best_epoch_per_head = np.full_like(val_heads, epoch + 1, dtype=int)
        else:
            improved = val_heads > best_per_head
            best_per_head = np.maximum(best_per_head, val_heads)
            best_epoch_per_head[improved] = epoch + 1
            sum_per_head += val_heads
            min_per_head = np.minimum(min_per_head, val_heads)

        mean_per_head = sum_per_head / count_epochs
        mean_val_acc = sum_per_head.max() / count_epochs

        # Determine if this is best epoch
        is_best = False
        if float(val_acc) > best_val_acc:
            best_val_acc = float(val_acc)
            is_best = True

        logger.info(
            "[%5d] train: %.3f%% test: %.3f%% subject: %.3f%% (Best: %.3f%%)"
            % (
                epoch + 1,
                train_acc,
                val_acc,
                -1.0 if val_subject_acc is None else val_subject_acc,
                best_val_acc,
            )
        )

        if rank == 0:
            csv_logger.log(epoch + 1, train_acc, val_acc, -1.0 if val_subject_acc is None else val_subject_acc)

        if val_only:
            return

        save_checkpoint(
            epoch + 1,
            mean_val_acc,
            best_val_acc,
            val_heads,
            best_per_head,
            mean_per_head,
            min_per_head,
            best_epoch_per_head,
            val_subject_acc,
            val_subject_heads,
            is_best=is_best,
        )

    logger.info("Training completed!")


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
):
    """
    Run one epoch of training or validation.
    
    Returns:
        (max_accuracy, per_classifier_accuracies)
    """
    from collections import defaultdict

    for c in classifiers:
        c.train(mode=training)

    criterion = torch.nn.CrossEntropyLoss()
    top1_meters = [AverageMeter() for _ in classifiers]
    subject_probs = defaultdict(list)
    subject_targets = {}

    for itr, batch in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        patient_ids = None
        # Handle batch layouts from local loader or ref_videodatasetr-style loader.
        if isinstance(batch, (tuple, list)):
            if len(batch) == 4:
                clips, labels, _, patient_ids = batch
            elif len(batch) == 3:
                clips, labels, patient_ids = batch
            elif len(batch) == 2:
                clips, labels = batch
            else:
                raise ValueError(f"Unsupported batch structure with length {len(batch)}")
        else:
            # No labels available, use dummy labels
            clips = batch
            batch_size = clips.shape[0] if isinstance(clips, torch.Tensor) else len(clips)
            labels = torch.randint(0, len(classifiers), (batch_size,), device=device)
        
        labels = labels.to(device, non_blocking=True)
        clips = clips.to(device, non_blocking=True)
        batch_size = clips.shape[0]
        patient_ids = _normalize_patient_ids(patient_ids, batch_size)

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
            enabled=use_bfloat16 and torch.cuda.is_available(),
        ):
            # Extract features from frozen encoder
            with torch.no_grad():
                # Encoder expects (B, C, T, H, W)
                # Use layer-specific forward if probing an intermediate layer
                if hasattr(encoder, 'probe_layer'):
                    encoder_output = encoder.forward_encoder_layer(clips, 0, encoder.probe_layer)
                else:
                    encoder_output = encoder.forward_encoder(clips, 0)
                
                # If encoder is MAE, it returns the last hidden states
                # MAE output shape is typically (B, num_patches, embed_dim)
                if isinstance(encoder_output, (tuple, list)):
                    encoder_output = encoder_output[0]

                # Ensure correct shape for classifier input
                if encoder_output.dim() == 4:
                    # (B, T, H, W) or (B, num_patches, H, W) - take mean across patches
                    encoder_output = encoder_output.mean(dim=(2, 3))
                elif encoder_output.dim() == 3:
                    # Already (B, num_patches, embed_dim) - good
                    pass
                elif encoder_output.dim() == 2:
                    # (B, embed_dim) - expand to (B, 1, embed_dim)
                    encoder_output = encoder_output.unsqueeze(1)

            # Forward through classifiers
            if training:
                outputs = [c(encoder_output) for c in classifiers]
                # Compute loss
                losses = [criterion(o, labels) for o in outputs]
                
                if use_bfloat16 and scaler[0] is not None:
                    [s.scale(l).backward() for s, l in zip(scaler, losses)]
                    [s.step(o) for s, o in zip(scaler, optimizer)]
                    [s.update() for s in scaler]
                else:
                    [l.backward() for l in losses]
                    [o.step() for o in optimizer]
                
                [o.zero_grad() for o in optimizer]
            else:
                with torch.no_grad():
                    outputs = [c(encoder_output) for c in classifiers]

            if not training and patient_ids is not None:
                outputs_cpu = [F.softmax(o, dim=1).detach().cpu() for o in outputs]
                for i, (label, patient_id) in enumerate(zip(labels.detach().cpu().tolist(), patient_ids)):
                    if patient_id is None:
                        continue
                    per_sample_head_probs = [outputs_cpu[h][i] for h in range(len(outputs_cpu))]
                    subject_probs[patient_id].append(per_sample_head_probs)
                    subject_targets[patient_id] = int(label)

            # Compute accuracy
            with torch.no_grad():
                softmax_outputs = [F.softmax(o, dim=1) for o in outputs]
                top1_accs = [
                    100.0 * out.max(dim=1).indices.eq(labels).sum() / batch_size
                    for out in softmax_outputs
                ]
                
                for t1m, t1a in zip(top1_meters, top1_accs):
                    t1m.update(float(t1a))

        if itr % 10 == 0:
            _agg_top1 = np.array([t1m.avg for t1m in top1_meters])
            mem_usage = torch.cuda.max_memory_allocated() / 1024.0**2 if torch.cuda.is_available() else 0
            logger.info(
                "[%5d] %.3f%% [%.3f%% %.3f%%] [mem: %.2e]"
                % (
                    itr,
                    _agg_top1.max(),
                    _agg_top1.mean(),
                    _agg_top1.min(),
                    mem_usage,
                )
            )

    _agg_top1 = np.array([t1m.avg for t1m in top1_meters])
    subject_level_accs = None
    subject_level_acc = None
    if not training and subject_probs:
        best_head_idx = int(np.argmax(_agg_top1)) if len(_agg_top1) > 0 else 0
        computed_subject_accs = []
        for head_idx in range(len(_agg_top1)):
            subj_preds = []
            subj_targets = []
            for patient_id, probs_list in subject_probs.items():
                head_probs = [sample_probs[head_idx] for sample_probs in probs_list]
                avg_prob = torch.stack(head_probs).mean(dim=0)
                subj_preds.append(torch.argmax(avg_prob).item())
                subj_targets.append(subject_targets[patient_id])
            computed_subject_accs.append(
                np.mean([p == t for p, t in zip(subj_preds, subj_targets)]) * 100 if subj_preds else 0.0
            )
        subject_level_accs = np.asarray(computed_subject_accs, dtype=float)
        subject_level_acc = float(subject_level_accs[best_head_idx]) if len(subject_level_accs) > 0 else None
        logger.info(
            f"Subject-level accuracy (best video head {best_head_idx}): {subject_level_acc:.2f}%"
        )

    return _agg_top1.max(), _agg_top1, subject_level_acc, subject_level_accs


def _normalize_patient_ids(patient_ids, batch_size):
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


def load_encoder(model_name, checkpoint, device, model_kwargs, probe_layer=-1):
    """Load and initialize the encoder model."""
    logger.info(f"Loading encoder: {model_name}")
    
    # Import model builder - use default kwargs optimized for EchoFM
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
    
    # Update with provided kwargs
    encoder_kwargs.update(model_kwargs)
    
    try:
        if model_name == "mae_vit_base_patch16":
            encoder = mae_vit_base_patch16(**encoder_kwargs)
        elif model_name == "mae_vit_large_patch16":
            encoder = mae_vit_large_patch16(**encoder_kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise
    
    encoder = encoder.to(device)

    # --- Layer selection for probing ---
    num_layers = len(encoder.blocks)
    logger.info(f"Encoder has {num_layers} transformer blocks (layers 0-{num_layers - 1})")

    # Compute effective layer index (supports negative indexing like Python: -1 = last layer)
    if probe_layer < 0:
        effective_layer = num_layers + probe_layer
    else:
        effective_layer = probe_layer

    # Clamp to valid range
    if effective_layer < 0 or effective_layer >= num_layers:
        logger.warning(
            f"probe_layer={probe_layer} out of valid range [{-num_layers}, {num_layers - 1}], "
            f"falling back to last layer ({num_layers - 1})"
        )
        effective_layer = num_layers - 1

    logger.info(f"Probing at layer {effective_layer} (0-indexed) of {num_layers} total layers")
    encoder.probe_layer = effective_layer
    encoder.num_layers = num_layers

    # Load checkpoint if provided
    if checkpoint and os.path.exists(checkpoint):
        logger.info(f"Loading checkpoint from {checkpoint}")
        try:
            ckpt = torch.load(checkpoint, map_location=device)
            
            # Handle different checkpoint formats
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
            
            # Remove "module." prefix if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            msg = encoder.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded encoder with message: {msg}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
    else:
        logger.warning("No checkpoint provided, using random initialization")

    return encoder


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    """Load training checkpoint."""
    checkpoint = torch.load(r_path, map_location=torch.device("cpu"))
    logger.info(f"Loading checkpoint from {r_path}")

    # Load classifiers
    pretrained_dict = checkpoint["classifiers"]
    for c, pd in zip(classifiers, pretrained_dict):
        is_wrapped = isinstance(c, DistributedDataParallel)
        has_module_prefix = any(k.startswith("module.") for k in pd.keys()) if pd else False

        if has_module_prefix and not is_wrapped:
            pd = {k.replace("module.", "", 1): v for k, v in pd.items()}
        elif not has_module_prefix and is_wrapped:
            pd = {"module." + k: v for k, v in pd.items()}

        c.load_state_dict(pd, strict=False)

    if val_only:
        logger.info("Loaded classifiers in val_only mode")
        return classifiers, opt, scaler, 0

    epoch = checkpoint["epoch"]
    logger.info(f"Loaded checkpoint from epoch {epoch}")

    # Load optimizers
    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    # Load scalers
    if scaler is not None and checkpoint.get("scaler") is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"]) if pd is not None]

    return classifiers, opt, scaler, epoch


def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    """Initialize optimizers and schedulers."""
    optimizers, schedulers, wd_schedulers, scalers = [], [], [], []

    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup", 0) * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]

        logger.info("Using AdamW optimizer")
        optimizers.append(torch.optim.AdamW(param_groups))
        schedulers.append(WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch)))
        wd_schedulers.append(CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch)))
        scalers.append(torch.cuda.amp.GradScaler() if use_bfloat16 else None)

    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule:
    """Cosine learning rate schedule with warmup."""

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps

            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule:
    """Cosine weight decay schedule."""

    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))

            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)

            group["weight_decay"] = new_wd


class AverageMeter:
    """Compute and store the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CSVLogger:
    """Simple CSV logger for tracking metrics."""

    def __init__(self, filename, *args):
        self.filename = filename
        self.columns = [col_name for _, col_name in args]
        self.formats = [col_format for col_format, _ in args]

        # Write header
        with open(filename, 'w') as f:
            f.write(','.join(self.columns) + '\n')

    def log(self, *args):
        with open(self.filename, 'a') as f:
            values = [format_str % val for format_str, val in zip(self.formats, args)]
            f.write(','.join(values) + '\n')


if __name__ == "__main__":
    import yaml

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/echo_probe_config.yaml")
    parser.add_argument("--val_only", action="store_true")
    parser.add_argument("--cuda_device", type=int, default=0, help="CUDA device index (default: 0)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.val_only:
        config["val_only"] = True

    config["cuda_device"] = args.cuda_device

    # Run evaluation
    main(config)







