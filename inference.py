#!/usr/bin/env python3
"""
VLM-PAR Inference Example

Usage:
    python inference.py --image person.jpg
    python inference.py --image person.jpg --checkpoint checkpoints/vlmpar_best.pth
"""

import argparse
import torch
import numpy as np
from PIL import Image
from vlmpar_model import VLMPARWrapper, PA100K_ATTRS


ATTR_DISPLAY = {
    'female': ('Gender', 'Female', 'Male'),
    'age_over_60': ('Age', 'Over 60', None),
    'age_18_60': ('Age', '18-60', None),
    'age_less_18': ('Age', 'Under 18', None),
    'hat': ('Hat', 'Yes', 'No'),
    'glasses': ('Glasses', 'Yes', 'No'),
    'short_sleeve': ('Upper', 'Short sleeve', None),
    'long_sleeve': ('Upper', 'Long sleeve', None),
    'trousers': ('Lower', 'Trousers', None),
    'shorts': ('Lower', 'Shorts', None),
    'skirt_dress': ('Lower', 'Skirt/Dress', None),
    'backpack': ('Bag', 'Backpack', None),
    'shoulder_bag': ('Bag', 'Shoulder bag', None),
    'hand_bag': ('Bag', 'Handbag', None),
}


def main():
    parser = argparse.ArgumentParser(description='VLM-PAR Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to pedestrian image')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained .pth file')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    # Load model
    print("Loading VLM-PAR model...")
    model = VLMPARWrapper(device=args.device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.par_head.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {args.checkpoint} (mA={checkpoint.get('mA', '?')}%)")
    else:
        print("Warning: No checkpoint loaded. Output will be random (untrained model).")

    model.eval()

    # Load and preprocess image
    img = Image.open(args.image).convert('RGB')
    img_tensor = model.preprocess(img).unsqueeze(0).to(args.device)

    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Display results
    print(f"\n{'='*40}")
    print(f"Image: {args.image}")
    print(f"{'='*40}")

    for attr, prob in zip(PA100K_ATTRS, probs):
        if attr in ATTR_DISPLAY:
            category, pos_label, neg_label = ATTR_DISPLAY[attr]
            if prob > args.threshold:
                print(f"  [{category:8s}] {pos_label:15s} ({prob:.1%})")
            elif neg_label and prob <= args.threshold:
                # Only show negative for binary attrs like gender
                if attr == 'female':
                    print(f"  [{category:8s}] {neg_label:15s} ({1-prob:.1%})")

    # All attributes (verbose)
    print(f"\n{'='*40}")
    print("All attributes:")
    print(f"{'='*40}")
    for attr, prob in zip(PA100K_ATTRS, probs):
        marker = "*" if prob > args.threshold else " "
        print(f"  {marker} {attr:20s} {prob:.3f}")


if __name__ == '__main__':
    main()
