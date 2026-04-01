#!/usr/bin/env python3
"""
VLM-PAR v3 Training — 84속성 논문 원본 구조 (Q=Image, K/V=Text)

논문: arXiv:2512.22217 (VLM-PAR)
데이터: RAP v2 (41,585장, 감시카메라, 92속성 중 행동 8개 제외 = 84속성)

손실: Focal Loss + Label Smoothing + pos_weight

사용법:
    python train_vlmpar_v3.py \
        --data-dir /path/to/RAPv2 \
        --epochs 50 \
        --batch-size 16 \
        --lr 3e-4 \
        --device cuda:0
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from scipy.io import loadmat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from vlmpar_model import VLMPARv3Wrapper, ATTR_NAMES, RAP_INDICES, NUM_ATTRS, ATTR_GROUPS


class RAPv2Dataset(Dataset):
    """RAP v2 데이터셋 (41,585장, 84속성)"""

    def __init__(self, data_dir, split='train', transform=None):
        self.transform = transform
        data_dir = Path(data_dir)
        data = loadmat(str(data_dir / 'RAP_annotation' / 'RAP_annotation.mat'))
        rap = data['RAP_annotation']

        num_images = 41585
        all_images = [rap[0][0][5][i][0][0].replace('_', '-') for i in range(num_images)]
        all_labels = rap[0][0][1][:num_images, RAP_INDICES].astype(np.float32)

        train_idx = rap[0][0][0][0][0][0][0][0][0, :] - 1
        test_idx = rap[0][0][0][0][0][0][0][1][0, :] - 1
        indices = train_idx.astype(int) if split == 'train' else test_idx.astype(int)
        indices = indices[indices < num_images]

        self.images = [all_images[i] for i in indices]
        self.labels = all_labels[indices]
        self.img_dir = data_dir / 'RAP_dataset'
        logger.info(f"RAP v2 {split}: {len(self.images)} images, {NUM_ATTRS} attrs")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(str(self.img_dir / self.images[idx])).convert('RGB')
        except Exception:
            img = Image.new('RGB', (64, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


class FocalBCELoss(nn.Module):
    """Focal Loss + Label Smoothing (논문 Section 3.5)"""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.05,
                 pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5

        if self.pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction='none')
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none')

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        return (focal_weight * bce).mean()


def compute_mA(preds: np.ndarray, labels: np.ndarray) -> float:
    binary = (preds > 0.5).astype(np.float32)
    n = preds.shape[1]
    mas = []
    for i in range(n):
        pos = labels[:, i] == 1
        neg = labels[:, i] == 0
        pa = (binary[pos, i] == 1).mean() if pos.sum() > 0 else 0
        na = (binary[neg, i] == 0).mean() if neg.sum() > 0 else 0
        mas.append((pa + na) / 2)
    return np.mean(mas) * 100


def compute_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    binary = (preds > 0.5).astype(np.float32)
    tp = (binary * labels).sum(axis=0)
    fp = (binary * (1 - labels)).sum(axis=0)
    fn = ((1 - binary) * labels).sum(axis=0)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.mean() * 100


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_p, all_l = [], []
    for imgs, labs in loader:
        logits = model(imgs.to(device))
        all_p.append(torch.sigmoid(logits).cpu().numpy())
        all_l.append(labs.numpy())
    preds = np.concatenate(all_p)
    labels = np.concatenate(all_l)

    mA = compute_mA(preds, labels)
    f1 = compute_f1(preds, labels)

    group_mA = {}
    for name, idx in ATTR_GROUPS.items():
        group_mA[name] = compute_mA(preds[:, idx], labels[:, idx])

    # 속성별 mA (디버깅용)
    attr_mA = {}
    for i, name in enumerate(ATTR_NAMES):
        pos = labels[:, i] == 1
        neg = labels[:, i] == 0
        pa = ((preds[pos, i] > 0.5).mean() if pos.sum() > 0 else 0)
        na = ((preds[neg, i] <= 0.5).mean() if neg.sum() > 0 else 0)
        attr_mA[name] = (pa + na) / 2 * 100

    return mA, f1, group_mA, attr_mA


def main():
    parser = argparse.ArgumentParser(description='VLM-PAR v3 Training (84 attrs, Paper Structure)')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--label-smoothing', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save-dir', type=str, default='checkpoints/vlmpar_v3')
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("=== VLM-PAR v3 (84 attrs, Q=Image, K/V=Text) ===")
    model = VLMPARv3Wrapper(device=device)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.2)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    train_set = RAPv2Dataset(args.data_dir, 'train', train_tf)
    test_set = RAPv2Dataset(args.data_dir, 'test', val_tf)
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # pos_weight 계산
    pos_ratio = train_set.labels.mean(axis=0)
    pos_weight = torch.tensor(
        (1 - pos_ratio) / (pos_ratio + 1e-6), dtype=torch.float32
    ).to(device).clamp(0.5, 10.0)

    logger.info(f"Pos ratio: min={pos_ratio.min():.3f} max={pos_ratio.max():.3f} mean={pos_ratio.mean():.3f}")

    criterion = FocalBCELoss(
        gamma=args.gamma,
        label_smoothing=args.label_smoothing,
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.AdamW(
        model.par_head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    logger.info(f"  Train={len(train_set)}, Test={len(test_set)}, Attrs={NUM_ATTRS}")
    logger.info(f"  LR={args.lr}, Epochs={args.epochs}, Batch={args.batch_size}")
    logger.info(f"  Focal(γ={args.gamma}) + LabelSmooth(ε={args.label_smoothing})")

    best_mA = 0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.par_head.train()
        model.siglip.eval()
        total_loss, n_batches = 0, 0
        t_epoch = time.time()

        for batch_idx, (imgs, labs) in enumerate(train_loader):
            imgs, labs = imgs.to(device), labs.to(device)
            logits = model(imgs)
            loss = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.par_head.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

            if batch_idx % 200 == 0:
                logger.info(f"  Ep{epoch} [{batch_idx}/{len(train_loader)}] loss={loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / n_batches
        epoch_time = time.time() - t_epoch

        mA, f1, group_mA, attr_mA = evaluate(model, test_loader, device)

        # 핵심 그룹만 간결하게
        key_groups = ['gender', 'head', 'upper_type', 'upper_color',
                      'lower_type', 'lower_color', 'shoes_color', 'direction']
        grp = ' | '.join(f"{k}={group_mA[k]:.1f}" for k in key_groups if k in group_mA)
        logger.info(f"Ep{epoch}/{args.epochs} [{epoch_time:.0f}s] loss={avg_loss:.4f} "
                    f"mA={mA:.2f}% F1={f1:.2f}% | {grp}")

        if mA > best_mA:
            best_mA = mA
            no_improve = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.par_head.state_dict(),
                'mA': mA, 'f1': f1,
                'group_mA': group_mA,
                'attr_mA': attr_mA,
                'args': vars(args),
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'vlmpar_v3_best.pth'))
            logger.info(f"  ★ Best mA={mA:.2f}% (saved)")

            # Best 시 속성별 상세 출력
            worst5 = sorted(attr_mA.items(), key=lambda x: x[1])[:5]
            best5 = sorted(attr_mA.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"  Best5: {', '.join(f'{n}={v:.1f}' for n, v in best5)}")
            logger.info(f"  Worst5: {', '.join(f'{n}={v:.1f}' for n, v in worst5)}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.par_head.state_dict(),
                'mA': mA, 'f1': f1,
            }, os.path.join(args.save_dir, f'vlmpar_v3_epoch{epoch}.pth'))

    logger.info(f"Done. Best mA={best_mA:.2f}%")


if __name__ == '__main__':
    main()
