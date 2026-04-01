# VLM-PAR

[English](README.md) | [한국어](README_ko.md)

### Unofficial Implementation of *"VLM-PAR: A Vision Language Model for Pedestrian Attribute Recognition"*

A **PyTorch implementation** of the VLM-PAR architecture described in [arXiv:2512.22217](https://arxiv.org/abs/2512.22217) (Sellam et al., Dec 2025). The original authors have not released their code — this is an implementation built from the paper's architecture description.

> **Note**: This is NOT based on [OpenPAR](https://github.com/Event-AHU/OpenPAR) or PromptPAR. It is an implementation using only the paper as reference.

## Approach

VLM-PAR treats pedestrian attribute recognition as a **vision-language alignment** problem using the paper's original Cross-Attention direction: **Q=Image, K/V=Text**.

A pretrained Vision-Language Model ([SigLIP 2](https://arxiv.org/abs/2502.14786)) serves as a frozen feature extractor. For each attribute, image patch tokens query against cached text embeddings through independent Cross-Attention modules — allowing each attribute to attend to its relevant image regions.

Key insight: **each image patch asks "how relevant am I to this attribute description?"** — hat patches attend to "a person wearing a hat", torso patches attend to "a person wearing black upper clothing".

### Architecture (Paper Section 3.4)

```
Image (224x224)
    |
SigLIP 2 ViT-B/16 (frozen, not trained)
    | 14x14 = 196 patch tokens [B, 196, 768]
    |
x84 Independent Cross-Attention Modules (trained)
    |
    |   Q = Image patch tokens [B, 196, 768]    <-- Image asks
    |   K = Text embedding [T, 768]              <-- Attribute description (cached)
    |   V = Text embedding [T, 768]              <-- Attribute description (cached)
    |
    |   head_h = softmax(Q . K^T / sqrt(96)) . V    (8 heads, d_k=96)
    |   h = LayerNorm(MultiHead + Residual)
    |   h = LayerNorm(FFN(h) + h)
    |   GAP -> Linear(768->1) -> logit
    |
    +-- CA #1:  "Female?"            -> attends to full body   -> 0.92
    +-- CA #13: "Hat?"               -> attends to head region -> 0.87
    +-- CA #55: "Upper black?"       -> attends to torso color -> 0.08
    +-- CA #56: "Upper white?"       -> attends to torso color -> 0.91
    +-- CA #67: "Lower black?"       -> attends to legs color  -> 0.85
    +-- ...
    +-- CA #84: "Shoes mixed color?" -> attends to feet        -> 0.05
    |
Result: Male, Hat, White short-sleeve, Black long-pants, Black sport shoes
```

Text embeddings are computed **once at server startup** and cached as K/V buffers — zero text encoder cost at inference time.

### Based On

| Foundation | Role | Reference |
|------------|------|-----------|
| **VLM-PAR** | Paper — architecture design | [arXiv:2512.22217](https://arxiv.org/abs/2512.22217) |
| **SigLIP 2** | Vision encoder — frozen ViT-B/16 | [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) |
| **open_clip** | Library — model loading, preprocessing | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) |
| **RAP v2** | Dataset — 41K surveillance camera images, 92 attributes | [dangweili/RAP](https://github.com/dangweili/RAP) |

### Paper vs Our Implementation

| Aspect | Paper | Ours | Why |
|--------|-------|------|-----|
| Cross-Attention direction | Q=Image, K/V=Text | **Q=Image, K/V=Text** (same) | Faithful to paper Section 3.4. Each patch queries against text description. |
| Text encoder at inference | Runs every time | **Cached once** (cost=0) | 84 attribute prompts are fixed. Cache text embeddings at startup. |
| Attributes | 26 (PA-100K) | **84** (RAP v2, excl. actions) | PA-100K has **no color attributes**. We use RAP v2's full attribute set minus 8 action attributes for practical surveillance use. |
| Dataset | PA-100K (100K, street photos) | **RAP v2** (41K, **CCTV**) | RAP v2 was captured from actual CCTV cameras, matching deployment environment. |
| Loss function | CE + Focal + Label Smoothing | **Focal + Label Smoothing + pos_weight** | Following paper Section 3.5 for class imbalance handling. |

## 84 Attributes

RAP v2 full 92 attributes minus 8 action attributes = **84 attributes**:

| Category | Count | Attributes |
|----------|:-----:|-----------|
| **Personal** | 9 | gender, age(3), body type(3), role(2) |
| **Head/Accessories** | 6 | bald, long hair, black hair, hat, glasses, muffler |
| **Upper type** | 9 | shirt, sweater, vest, t-shirt, cotton, jacket, suit, tight, short-sleeve |
| **Lower type** | 6 | long-trousers, skirt, short-skirt, dress, jeans, tight-trousers |
| **Shoes type** | 5 | leather, sport, boots, cloth, casual |
| **Attachments** | 8 | backpack, shoulder bag, handbag, box, plastic bag, paper bag, trunk, other |
| **Direction** | 4 | front, back, left, right |
| **Occlusion** | 8 | left, right, up, down, environment, attachment, person, other |
| **Upper color** | 12 | black, white, gray, red, green, blue, yellow, brown, purple, pink, orange, mixed |
| **Lower color** | 8 | black, white, gray, red, green, blue, yellow, mixed |
| **Shoes color** | 9 | black, white, gray, red, green, blue, yellow, brown, mixed |

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
  --save-dir checkpoints/vlmpar_v3
```

### 4. Inference

```python
import torch
from vlmpar_model import VLMPARv3Wrapper, _parse_attributes

model = VLMPARv3Wrapper(device='cuda:0')
checkpoint = torch.load('checkpoints/vlmpar_v3/vlmpar_v3_best.pth', weights_only=False)
model.par_head.load_state_dict(checkpoint['model_state_dict'])

results = model.classify(tensor)
print(results[0])
# {'gender': 'male', 'age': 'young', 'hat': True, 'glasses': False,
#  'upper_type': 'short_sleeve', 'upper_color': 'black',
#  'lower_type': 'short_pants', 'lower_color': 'black',
#  'shoes_type': 'sport', 'shoes_color': 'white',
#  'backpack': True, 'direction': 'front'}
```

Or use the CLI:

```bash
python inference.py --image person.jpg --checkpoint checkpoints/vlmpar_v3/vlmpar_v3_best.pth
```

## Model Specs

| | |
|---|---|
| Architecture | SigLIP 2 ViT-B/16 (frozen) + 84 Independent Cross-Attention |
| CA direction | **Q=Image patches, K/V=Text embeddings** (paper faithful) |
| Trainable params | ~130M (84 CA modules) |
| Total params | ~216M (86M frozen SigLIP) |
| Input | 224 x 224 RGB |
| Output | 84 probabilities (sigmoid) |
| Loss | Focal Loss (gamma=2.0) + Label Smoothing (eps=0.05) + pos_weight |
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
