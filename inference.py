#!/usr/bin/env python3
"""
VLM-PAR v3 Inference — 84 Attributes (Paper Structure: Q=Image, K/V=Text)

Usage:
    python inference.py --image person.jpg
    python inference.py --image person.jpg --checkpoint checkpoints/vlmpar_v3/vlmpar_v3_best.pth
"""

import argparse
import torch
import numpy as np
from PIL import Image
from vlmpar_model import VLMPARv3Wrapper, ATTR_NAMES, ATTR_GROUPS, _parse_attributes


def main():
    parser = argparse.ArgumentParser(description='VLM-PAR v3 Inference (84 attrs)')
    parser.add_argument('--image', type=str, required=True, help='Path to pedestrian image')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained .pth file')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    print("Loading VLM-PAR v3 (84 attrs, Q=Image, K/V=Text)...")
    model = VLMPARv3Wrapper(device=args.device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.par_head.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded: {args.checkpoint} (mA={checkpoint.get('mA', '?')}%, F1={checkpoint.get('f1', '?')}%)")
    else:
        print("Warning: No checkpoint loaded. Output will be random (untrained model).")

    model.eval()

    # Load and preprocess image
    img = Image.open(args.image).convert('RGB')
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    img_tensor = transform(img).unsqueeze(0).to(args.device)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Parse structured attributes
    attrs = _parse_attributes(probs, args.threshold)

    # Display
    print(f"\n{'='*50}")
    print(f"Image: {args.image}")
    print(f"{'='*50}")
    print(f"  Gender:      {attrs['gender']}")
    print(f"  Age:         {attrs['age']}")
    print(f"  Body type:   {attrs['body_type']}")
    print(f"  Hat:         {attrs['hat']}")
    print(f"  Glasses:     {attrs['glasses']}")
    print(f"  Upper:       {attrs['upper_type']} ({attrs.get('upper_type_detail', '')})")
    print(f"  Upper color: {attrs['upper_color']}")
    print(f"  Lower:       {attrs['lower_type']} ({attrs.get('lower_type_detail', '')})")
    print(f"  Lower color: {attrs['lower_color']}")
    print(f"  Shoes:       {attrs['shoes_type']}, {attrs['shoes_color']}")
    print(f"  Direction:   {attrs['direction']}")
    print(f"  Backpack:    {attrs['backpack']}")
    print(f"  Shoulder bag:{attrs['shoulder_bag']}")
    print(f"  Hand bag:    {attrs['hand_bag']}")

    # All 84 attributes with probabilities
    print(f"\n{'='*50}")
    print(f"All {len(ATTR_NAMES)} attributes:")
    print(f"{'='*50}")
    for group_name, indices in ATTR_GROUPS.items():
        print(f"\n  [{group_name}]")
        for idx in indices:
            p = probs[idx]
            marker = "*" if p > args.threshold else " "
            print(f"    {marker} {ATTR_NAMES[idx]:25s} {p:.3f}")


if __name__ == '__main__':
    main()
