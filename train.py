#!/usr/bin/env python3
"""
VLM-PAR v3 Training — 38속성 독립 Cross-Attention (논문 원본 구조)
"""

import os, sys, time, argparse, logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from scipy.io import loadmat

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from vlmpar_model import VLMPARWrapper, ATTR_NAMES, RAP_INDICES, NUM_ATTRS


class RAPv2Dataset(Dataset):
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
        # 범위 내 인덱스만
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
        except:
            img = Image.new('RGB', (64, 128))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


def compute_mA(preds, labels):
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

    # 그룹별
    groups = {
        'gender': [0], 'head': [1,2],
        'upper_type': list(range(3,12)), 'upper_color': list(range(12,24)),
        'lower_type': list(range(24,30)), 'lower_color': list(range(30,38)),
    }
    group_mA = {}
    for name, idx in groups.items():
        group_mA[name] = compute_mA(preds[:, idx], labels[:, idx])

    return mA, group_mA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save-dir', type=str, default='checkpoints/vlmpar')
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)

    # Model
    model = VLMPARWrapper(device=device)

    # Data
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    train_set = RAPv2Dataset(args.data_dir, 'train', train_tf)
    test_set = RAPv2Dataset(args.data_dir, 'test', val_tf)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True)

    # Loss
    pos_ratio = train_set.labels.mean(axis=0)
    pos_weight = torch.tensor((1 - pos_ratio) / (pos_ratio + 1e-6), dtype=torch.float32).to(device).clamp(0.5, 10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer
    optimizer = torch.optim.AdamW(model.par_head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    logger.info(f"=== VLM-PAR v3 (38 Independent CAs) ===")
    logger.info(f"  Train: {len(train_set)}, Test: {len(test_set)}, Attrs: {NUM_ATTRS}")
    logger.info(f"  LR: {args.lr}, Epochs: {args.epochs}, Batch: {args.batch_size}")

    best_mA = 0
    patience = 15
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.par_head.train()
        model.siglip.eval()
        total_loss, n = 0, 0

        for batch_idx, (imgs, labs) in enumerate(train_loader):
            imgs, labs = imgs.to(device), labs.to(device)
            logits = model(imgs)
            loss = criterion(logits, labs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.par_head.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1

            if batch_idx % 100 == 0:
                logger.info(f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] loss={loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / n

        # Evaluate
        mA, group_mA = evaluate(model, test_loader, device)
        grp = ' | '.join(f"{k}={v:.1f}" for k, v in group_mA.items())
        logger.info(f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | mA={mA:.2f}% | {grp}")

        if mA > best_mA:
            best_mA = mA
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.par_head.state_dict(),
                'mA': mA,
                'group_mA': group_mA,
            }, os.path.join(args.save_dir, 'vlmpar_best.pth'))
            logger.info(f"  ** Best mA={mA:.2f}% (saved)")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    logger.info(f"Done. Best mA={best_mA:.2f}%")


if __name__ == '__main__':
    main()
