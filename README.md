# VLM-PAR

[English](README.md) | [한국어](README_ko.md)

### Unofficial Implementation of *"VLM-PAR: A Vision Language Model for Pedestrian Attribute Recognition"*

A **PyTorch implementation** of the VLM-PAR architecture described in [arXiv:2512.22217](https://arxiv.org/abs/2512.22217) (Sellam et al., Dec 2025). The original authors have not released their code — this is an implementation built from the paper's architecture description.

> **Note**: This is NOT based on [OpenPAR](https://github.com/Event-AHU/OpenPAR) or PromptPAR. It is an implementation using only the paper as reference.

## Approach

VLM-PAR treats pedestrian attribute recognition as a **vision-language alignment** problem. It uses a pretrained Vision-Language Model ([SigLIP 2](https://arxiv.org/abs/2502.14786)) as a frozen feature extractor, and learns **per-attribute independent Cross-Attention** modules to map visual features to attribute predictions.

Key insight: **a frozen VLM already understands clothing, accessories, and body characteristics** — we only need to teach it how to read those features for specific attributes.

### Architecture

```
Image (224×224)
    ↓
SigLIP 2 ViT-B/16 (frozen, not trained)
    ↓ 14×14 = 196 patch tokens [B, 196, 768]
    ↓
38 Independent Cross-Attention Modules (trained)
    ├── CA #1:  "Female?"      → query attends to patches → 0.92 → Female
    ├── CA #2:  "Hat?"         → focuses on head region   → 0.87 → Wearing
    ├── CA #3:  "Glasses?"     → focuses on face region   → 0.12 → Not wearing
    ├── CA #4:  "T-shirt?"     → focuses on upper body    → 0.83 → Yes (short sleeve)
    ├── ...
    ├── CA #13: "Upper black?" → focuses on torso color   → 0.08 → No
    ├── CA #14: "Upper white?" → focuses on torso color   → 0.91 → Yes
    ├── ...
    └── CA #38: "Lower mixed?" → focuses on lower body    → 0.05 → No
    ↓
Result: Male, Hat, White short-sleeve, Black long-pants
```

Each attribute has its own **learnable query token** that learns to attend to the relevant image patches through Cross-Attention.

### Based On

| Foundation | Role | Reference |
|------------|------|-----------|
| **VLM-PAR** | Paper — architecture design | [arXiv:2512.22217](https://arxiv.org/abs/2512.22217) |
| **SigLIP 2** | Vision encoder — frozen ViT-B/16 | [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) |
| **open_clip** | Library — model loading, preprocessing | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) |
| **RAP v2** | Dataset — 41K surveillance camera images, 92 attributes | [dangweili/RAP](https://github.com/dangweili/RAP) |

### Paper vs Our Implementation

The core algorithm follows the paper as closely as possible. Two changes were made to adapt for **real-world person feature search**:

| Aspect | Paper | Ours | Why |
|--------|-------|------|-----|
| Cross-Attention | Per-attribute independent | **Per-attribute independent** (same) | Following the paper's design |
| Attributes | 26 (PA-100K) | **38** (RAP v2 selected) | PA-100K has **no color attributes**. Clothing color (upper 12 colors, lower 8 colors) is important for practical use. Selected 38 attributes from RAP v2's 92 for person identification. |
| Dataset | PA-100K (100K images, street photos) | **RAP v2** (41K images, **CCTV**) | PA-100K consists of eye-level pedestrian photos. Our system operates on **elevated CCTV cameras**. RAP v2 was captured from actual CCTV, matching the deployment environment. |
| Text init | SigLIP text encoder | **SigLIP text encoder** (same) | Following paper — initializes query tokens with text like "a person wearing a hat" to provide semantic prior knowledge. |
| Code | Not released | **This repository** | Paper (arXiv:2512.22217) does not provide source code. Independently implemented based on the architecture description. |

### Why This Design

**Goal**: Classify pedestrian attributes (clothing type, color, accessories) from surveillance camera images with high accuracy.

**Why VLM-PAR**: The paper showed that a frozen SigLIP 2 backbone already understands clothing attributes — only lightweight attention heads need training. Fast training (~6 hours), small VRAM (~1.3GB), high accuracy (mA 88%).

**Why RAP v2 over PA-100K**: PA-100K has 26 attributes but **zero color attributes**. Clothing color is important for practical pedestrian attribute recognition. RAP v2 provides 12 upper-body colors + 8 lower-body colors, and its images come from real surveillance cameras.

## Results

### RAP v2 Test Set — mA 88.14%

| Group | Attributes | mA |
|-------|-----------|-----|
| Gender | female | **95.9%** |
| Head | hat, glasses | **87.3%** |
| Upper type | shirt, sweater, vest, t-shirt, cotton, jacket, suit, tight, short-sleeve | 85.0% |
| Upper color | black, white, gray, red, green, blue, yellow, brown, purple, pink, orange, mixed | **89.8%** |
| Lower type | long-trousers, skirt, short-skirt, dress, jeans, tight-trousers | **91.6%** |
| Lower color | black, white, gray, red, green, blue, yellow, mixed | **85.9%** |

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Download RAP v2 Dataset

Request from the [RAP dataset page](https://www.rapdataset.com/rapv2.html) or use the [OpenPAR Dropbox mirror](https://www.dropbox.com/scl/fo/boipdmufnsnsvmfdle5um/AMbwWDNnlBWnVbnxxv4VcFM).

### 3. Train

```bash
python train.py \
  --data-dir /path/to/RAPv2 \
  --epochs 50 \
  --batch-size 16 \
  --lr 3e-4 \
  --device cuda:0 \
  --save-dir checkpoints/vlmpar
```

~6 hours on a single GPU. Only the 38 Cross-Attention heads (~186M params) are trained — SigLIP 2 stays frozen.

### 4. Inference

```python
import torch
from vlmpar_model import VLMPARWrapper, ATTR_NAMES

model = VLMPARWrapper(device='cuda:0')
model.par_head.load_state_dict(torch.load('vlmpar_best.pth')['model_state_dict'])

results = model.classify(tensor)
print(results[0])
# {'gender': 'male', 'hat': True, 'upper_type': 'short_sleeve',
#  'upper_color': 'white', 'lower_type': 'long_pants', 'lower_color': 'black'}
```

## 38 Attributes

Selected for **person feature reference**:

| Category | Count | Attributes |
|----------|-------|-----------|
| Gender | 1 | female |
| Head | 2 | hat, glasses |
| Upper type | 9 | shirt, sweater, vest, t-shirt, cotton, jacket, suit, tight, short-sleeve |
| Upper color | 12 | black, white, gray, red, green, blue, yellow, brown, purple, pink, orange, mixed |
| Lower type | 6 | long-trousers, skirt, short-skirt, dress, jeans, tight-trousers |
| Lower color | 8 | black, white, gray, red, green, blue, yellow, mixed |

## Model Specs

| | |
|---|---|
| Architecture | SigLIP 2 ViT-B/16 (frozen) + 38 Independent Cross-Attention |
| Trainable params | ~186M |
| Total params | ~272M (86M frozen SigLIP) |
| Input | 224 × 224 RGB |
| Output | 38 probabilities (sigmoid) |
| Best mA | **88.14%** (RAP v2) |
| Framework | PyTorch + open_clip |

## License

| Component | License |
|-----------|---------|
| This code | [MIT](LICENSE) |
| SigLIP 2 | Apache 2.0 + CC BY 4.0 |
| RAP v2 | Research use only |

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{sellam2025vlmpar,
  title={VLM-PAR: A Vision Language Model for Pedestrian Attribute Recognition},
  author={Sellam, Abdellah Zakaria and Bekhouche, Salah Eddine and Dornaika, Fadi
          and Distante, Cosimo and Hadid, Abdenour},
  journal={arXiv preprint arXiv:2512.22217},
  year={2025}
}
```

## Acknowledgements

- [VLM-PAR](https://arxiv.org/abs/2512.22217) — Sellam et al.
- [SigLIP 2](https://arxiv.org/abs/2502.14786) — Google Research
- [open_clip](https://github.com/mlfoundations/open_clip) — LAION
- [RAP Dataset](https://github.com/dangweili/RAP) — Li et al.
