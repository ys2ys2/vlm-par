#!/usr/bin/env python3
"""
VLM-PAR v3: 논문 원본 구조 — Q=Image, K/V=Text, 84속성 독립 Cross-Attention

arXiv:2512.22217 (VLM-PAR) 논문 충실 재현:
- SigLIP 2 ViT-B/16 (frozen)
- 속성별 독립 Cross-Attention 84개
  - Query = Image patch tokens [B, 196, 768]
  - Key/Value = Text token embeddings [T, 768] (캐시됨)
- 속성당 Classification Head (768 → 1)
- 추론 시 Text Encoder 불필요 (서버 시작 시 1회 인코딩 → 캐시)

84속성 (RAP v2 전체 92개 - 행동 8개):
- 성별 (1), 나이 (3), 체형 (3), 역할 (2)
- 머리/액세서리 (6): 대머리, 장발, 흑발, 모자, 안경, 머플러
- 상의 유형 (9), 하의 유형 (6), 신발 유형 (5)
- 소지품 (8), 방향 (4), 가림 (8)
- 상의 색상 (12), 하의 색상 (8), 신발 색상 (9)

VRAM: ~1.2GB (SigLIP frozen fp16 + 84 CA fp32)
License: MIT (코드) + Apache 2.0 / CC BY 4.0 (SigLIP 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================
# 84속성 정의 (RAP v2 전체 - 행동 8개)
# ============================================================

ATTR_NAMES = [
    # --- 성별·나이·체형·역할 (9) ---
    'female',                                           # 0
    'age_less16', 'age_17_30', 'age_31_45',            # 1-3
    'body_fat', 'body_normal', 'body_thin',            # 4-6
    'customer', 'clerk',                                # 7-8
    # --- 머리·액세서리 (6) ---
    'hs_bald', 'hs_long_hair', 'hs_black_hair',       # 9-11
    'hs_hat', 'hs_glasses', 'hs_muffler',             # 12-14
    # --- 상의 유형 (9) ---
    'ub_shirt', 'ub_sweater', 'ub_vest', 'ub_tshirt', # 15-18
    'ub_cotton', 'ub_jacket', 'ub_suit',               # 19-21
    'ub_tight', 'ub_short_sleeve',                     # 22-23
    # --- 하의 유형 (6) ---
    'lb_long_trousers', 'lb_skirt', 'lb_short_skirt', # 24-26
    'lb_dress', 'lb_jeans', 'lb_tight_trousers',      # 27-29
    # --- 신발 유형 (5) ---
    'shoes_leather', 'shoes_sport', 'shoes_boots',     # 30-32
    'shoes_cloth', 'shoes_casual',                     # 33-34
    # --- 소지품 (8) ---
    'attach_backpack', 'attach_shoulder_bag',          # 35-36
    'attach_hand_bag', 'attach_box',                   # 37-38
    'attach_plastic_bag', 'attach_paper_bag',          # 39-40
    'attach_hand_trunk', 'attach_other',               # 41-42
    # --- 방향 (4) ---
    'face_front', 'face_back', 'face_left', 'face_right',  # 43-46
    # --- 가림 (8) ---
    'occ_left', 'occ_right', 'occ_up', 'occ_down',    # 47-50
    'occ_environment', 'occ_attachment',               # 51-52
    'occ_person', 'occ_other',                         # 53-54
    # --- 상의 색상 (12) ---
    'up_black', 'up_white', 'up_gray', 'up_red',      # 55-58
    'up_green', 'up_blue', 'up_yellow', 'up_brown',   # 59-62
    'up_purple', 'up_pink', 'up_orange', 'up_mixture', # 63-66
    # --- 하의 색상 (8) ---
    'low_black', 'low_white', 'low_gray', 'low_red',  # 67-70
    'low_green', 'low_blue', 'low_yellow', 'low_mixture',  # 71-74
    # --- 신발 색상 (9) ---
    'shoes_black', 'shoes_white', 'shoes_gray',        # 75-77
    'shoes_red', 'shoes_green', 'shoes_blue',          # 78-80
    'shoes_yellow', 'shoes_brown', 'shoes_mixture',    # 81-83
]
NUM_ATTRS = len(ATTR_NAMES)
assert NUM_ATTRS == 84, f"Expected 84 attrs, got {NUM_ATTRS}"

# RAP v2 annotation 인덱스 → 84속성 매핑 (행동 43-50 제외)
RAP_INDICES = [
    # 성별·나이·체형·역할 (0-8)
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    # 머리·액세서리 (9-14)
    9, 10, 11, 12, 13, 14,
    # 상의 유형 (15-23)
    15, 16, 17, 18, 19, 20, 21, 22, 23,
    # 하의 유형 (24-29)
    24, 25, 26, 27, 28, 29,
    # 신발 유형 (30-34)
    30, 31, 32, 33, 34,
    # 소지품 (35-42)
    35, 36, 37, 38, 39, 40, 41, 42,
    # 방향 (51-54) — 행동(43-50) 건너뜀
    51, 52, 53, 54,
    # 가림 (55-62)
    55, 56, 57, 58, 59, 60, 61, 62,
    # 상의 색상 (63-74)
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    # 하의 색상 (75-82)
    75, 76, 77, 78, 79, 80, 81, 82,
    # 신발 색상 (83-91)
    83, 84, 85, 86, 87, 88, 89, 90, 91,
]
assert len(RAP_INDICES) == 84, f"Expected 84 indices, got {len(RAP_INDICES)}"

# 속성 그룹 (평가·파이프라인용)
ATTR_GROUPS = {
    'gender':       [0],
    'age':          [1, 2, 3],
    'body':         [4, 5, 6],
    'role':         [7, 8],
    'head':         list(range(9, 15)),
    'upper_type':   list(range(15, 24)),
    'lower_type':   list(range(24, 30)),
    'shoes_type':   list(range(30, 35)),
    'attachment':   list(range(35, 43)),
    'direction':    list(range(43, 47)),
    'occlusion':    list(range(47, 55)),
    'upper_color':  list(range(55, 67)),
    'lower_color':  list(range(67, 75)),
    'shoes_color':  list(range(75, 84)),
}

# 텍스트 프롬프트 (SigLIP text encoder용, 84개)
TEXT_PROMPTS = [
    # 성별·나이·체형·역할 (9)
    "a female person",
    "a child or teenager under 16 years old",
    "a young person aged 17 to 30",
    "a middle-aged person aged 31 to 45",
    "a fat or overweight person",
    "a person with normal body type",
    "a thin or slim person",
    "a customer or shopper",
    "a clerk or shop employee",
    # 머리·액세서리 (6)
    "a bald person",
    "a person with long hair",
    "a person with black hair",
    "a person wearing a hat or cap",
    "a person wearing glasses or sunglasses",
    "a person wearing a muffler or scarf",
    # 상의 유형 (9)
    "a person wearing a shirt",
    "a person wearing a sweater",
    "a person wearing a vest",
    "a person wearing a t-shirt",
    "a person wearing cotton clothing",
    "a person wearing a jacket or coat",
    "a person wearing a suit",
    "a person wearing tight upper clothing",
    "a person wearing short sleeves",
    # 하의 유형 (6)
    "a person wearing long trousers or pants",
    "a person wearing a skirt",
    "a person wearing a short skirt or shorts",
    "a person wearing a dress",
    "a person wearing jeans",
    "a person wearing tight trousers or leggings",
    # 신발 유형 (5)
    "a person wearing leather shoes",
    "a person wearing sport shoes or sneakers",
    "a person wearing boots",
    "a person wearing cloth shoes or sandals",
    "a person wearing casual shoes",
    # 소지품 (8)
    "a person carrying a backpack",
    "a person carrying a single shoulder bag",
    "a person carrying a handbag",
    "a person carrying a box",
    "a person carrying a plastic bag",
    "a person carrying a paper bag",
    "a person carrying a hand trunk or suitcase",
    "a person carrying other items",
    # 방향 (4)
    "a person facing front toward the camera",
    "a person facing away from the camera showing their back",
    "a person facing left",
    "a person facing right",
    # 가림 (8)
    "a person partially occluded on the left side",
    "a person partially occluded on the right side",
    "a person partially occluded on the upper body",
    "a person partially occluded on the lower body",
    "a person occluded by environment or objects",
    "a person occluded by their own attachments",
    "a person occluded by another person",
    "a person with other types of occlusion",
    # 상의 색상 (12)
    "a person wearing black upper clothing",
    "a person wearing white upper clothing",
    "a person wearing gray upper clothing",
    "a person wearing red upper clothing",
    "a person wearing green upper clothing",
    "a person wearing blue upper clothing",
    "a person wearing yellow upper clothing",
    "a person wearing brown upper clothing",
    "a person wearing purple upper clothing",
    "a person wearing pink upper clothing",
    "a person wearing orange upper clothing",
    "a person wearing mixed color upper clothing",
    # 하의 색상 (8)
    "a person wearing black lower clothing",
    "a person wearing white lower clothing",
    "a person wearing gray lower clothing",
    "a person wearing red lower clothing",
    "a person wearing green lower clothing",
    "a person wearing blue lower clothing",
    "a person wearing yellow lower clothing",
    "a person wearing mixed color lower clothing",
    # 신발 색상 (9)
    "a person wearing black shoes",
    "a person wearing white shoes",
    "a person wearing gray shoes",
    "a person wearing red shoes",
    "a person wearing green shoes",
    "a person wearing blue shoes",
    "a person wearing yellow shoes",
    "a person wearing brown shoes",
    "a person wearing mixed color shoes",
]
assert len(TEXT_PROMPTS) == 84, f"Expected 84 prompts, got {len(TEXT_PROMPTS)}"


# ============================================================
# 논문 원본 구조: 속성별 독립 Cross-Attention (Q=Image, K/V=Text)
# ============================================================

class PaperCrossAttention(nn.Module):
    """논문 원본 Cross-Attention (arXiv:2512.22217 Section 3.4)

    Q = Image patch tokens [B, N, 768]
    K = Text token embeddings [T, 768] (캐시됨)
    V = Text token embeddings [T, 768] (캐시됨)
    """

    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

        self.register_buffer('text_kv', None)

    def set_text_embedding(self, text_tokens: torch.Tensor):
        """Text embedding 캐시 설정 — [T, dim]"""
        self.text_kv = text_tokens.detach()

    def forward(self, img_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_tokens: [B, N, 768]
        Returns:
            logit: [B, 1]
        """
        B = img_tokens.shape[0]

        if self.text_kv is not None:
            kv = self.text_kv.unsqueeze(0).expand(B, -1, -1)
        else:
            kv = img_tokens

        attn_out, _ = self.cross_attn(img_tokens, kv, kv)
        h = self.norm1(attn_out + img_tokens)
        h = self.norm2(self.ffn(h) + h)
        pooled = h.mean(dim=1)
        return self.head(pooled)


class VLM_PAR_v3(nn.Module):
    """VLM-PAR v3: 84속성 독립 Cross-Attention (논문 원본 Q=Image, K/V=Text)"""

    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attns = nn.ModuleList([
            PaperCrossAttention(dim, num_heads, dropout)
            for _ in range(NUM_ATTRS)
        ])

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, 196, 768]
        Returns:
            logits: [B, 84]
        """
        logits = [ca(patch_tokens) for ca in self.cross_attns]
        return torch.cat(logits, dim=1)

    def set_text_embeddings(self, text_embeddings: List[torch.Tensor]):
        for i, ca in enumerate(self.cross_attns):
            ca.set_text_embedding(text_embeddings[i])

    def init_from_text(self, siglip_model, tokenizer, device='cuda'):
        """SigLIP text encoder로 K/V 캐시 초기화"""
        try:
            text_embeddings = []
            for prompt in TEXT_PROMPTS:
                tokens = tokenizer([prompt]).to(device)
                with torch.no_grad():
                    text_encoder = siglip_model.text
                    x = text_encoder(tokens)
                    if hasattr(x, 'shape') and x.dim() == 2:
                        text_embeddings.append(x.squeeze(0).unsqueeze(0))
                    else:
                        text_embeddings.append(x.squeeze(0))

            self.set_text_embeddings(text_embeddings)
            logger.info(f"Text K/V cache initialized: {len(text_embeddings)} attributes")
            return True
        except Exception as e:
            logger.warning(f"Text init failed: {e}, trying encode_text fallback")
            return self._init_from_encode_text(siglip_model, tokenizer, device)

    def _init_from_encode_text(self, siglip_model, tokenizer, device='cuda'):
        try:
            text_embeddings = []
            for prompt in TEXT_PROMPTS:
                tokens = tokenizer([prompt]).to(device)
                with torch.no_grad():
                    feat = siglip_model.encode_text(tokens)
                    feat = F.normalize(feat, dim=-1)
                text_embeddings.append(feat.squeeze(0).unsqueeze(0))

            self.set_text_embeddings(text_embeddings)
            logger.info(f"Text K/V cache initialized (sentence embedding fallback)")
            return True
        except Exception as e:
            logger.warning(f"Text init completely failed: {e}")
            return False


# ============================================================
# SigLIP 2 + VLM-PAR v3 통합 래퍼
# ============================================================

class VLMPARv3Wrapper(nn.Module):
    """SigLIP 2 (frozen) + VLM-PAR v3 (84 독립 CA, 논문 구조)

    VRAM:
    - SigLIP ViT-B/16 (fp16, frozen): ~170MB
    - 84 Cross-Attention (fp32, trainable): ~1GB
    - 총합: ~1.2GB
    """

    def __init__(self, device: str = 'cuda:0'):
        super().__init__()
        self.device = device

        logger.info("Loading SigLIP 2 ViT-B/16 (frozen, fp16)...")
        siglip, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16-SigLIP2', pretrained='webli'
        )
        self.siglip = siglip.to(device).half().eval()
        for p in self.siglip.parameters():
            p.requires_grad = False

        self.par_head = VLM_PAR_v3(dim=768, num_heads=8, dropout=0.1).to(device)

        trainable = sum(p.numel() for p in self.par_head.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.siglip.parameters())
        logger.info(f"VLM-PAR v3: {trainable:,} trainable + {total:,} frozen = "
                    f"{NUM_ATTRS} independent CAs")

        try:
            tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP2')
            self.par_head.init_from_text(self.siglip, tokenizer, device)
        except Exception as e:
            logger.warning(f"Text init failed: {e}")

    def extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """SigLIP 2 패치 토큰 추출 → [B, 196, 768] (fp32)"""
        with torch.no_grad():
            images_fp16 = images.half()
            visual = self.siglip.visual
            trunk = visual.trunk
            x = trunk.patch_embed(images_fp16)
            x = trunk._pos_embed(x)
            if hasattr(trunk, 'patch_drop'):
                x = trunk.patch_drop(x)
            if hasattr(trunk, 'norm_pre'):
                x = trunk.norm_pre(x)
            for blk in trunk.blocks:
                x = blk(x)
            x = trunk.norm(x)
        return x.float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.extract_patch_tokens(images)
        return self.par_head(patch_tokens)

    def classify(self, images: torch.Tensor, threshold: float = 0.5) -> List[Dict]:
        """추론: 84속성 → 파이프라인 호환 Dict"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for b in range(probs.shape[0]):
            p = probs[b]
            results.append(_parse_attributes(p, threshold))

        return results


# ============================================================
# 속성 파싱 (파이프라인 호환)
# ============================================================

def _parse_attributes(p: np.ndarray, threshold: float = 0.5) -> Dict:
    """84개 확률을 구조화된 속성으로 변환"""
    result = {}

    # --- 성별 ---
    result['gender'] = 'female' if p[0] > threshold else 'male'

    # --- 나이 (argmax) ---
    age_probs = p[1:4]
    age_idx = int(age_probs.argmax())
    result['age'] = ['child', 'young', 'middle_aged'][age_idx]

    # --- 체형 (argmax) ---
    body_probs = p[4:7]
    body_idx = int(body_probs.argmax())
    result['body_type'] = ['fat', 'normal', 'thin'][body_idx]

    # --- 역할 ---
    result['is_clerk'] = bool(p[8] > threshold)

    # --- 머리·액세서리 ---
    result['bald'] = bool(p[9] > threshold)
    result['long_hair'] = bool(p[10] > threshold)
    result['black_hair'] = bool(p[11] > threshold)
    result['hat'] = bool(p[12] > threshold)
    result['glasses'] = bool(p[13] > threshold)
    result['muffler'] = bool(p[14] > threshold)

    # --- 상의 유형 (9종 → short_sleeve/long_sleeve) ---
    upper_type_probs = p[15:24]
    upper_types = ['shirt', 'sweater', 'vest', 'tshirt', 'cotton',
                   'jacket', 'suit', 'tight', 'short_sleeve']
    upper_type_raw = upper_types[int(upper_type_probs.argmax())]
    result['upper_type'] = (
        'short_sleeve' if upper_type_raw in ('tshirt', 'vest', 'short_sleeve')
        else 'long_sleeve'
    )
    result['upper_type_detail'] = upper_type_raw

    # --- 하의 유형 (6종 → long_pants/short_pants/skirt) ---
    lower_type_probs = p[24:30]
    lower_types = ['long_pants', 'skirt', 'short_pants', 'dress', 'jeans', 'tight_trousers']
    lower_type_raw = lower_types[int(lower_type_probs.argmax())]
    lower_map = {
        'long_pants': 'long_pants', 'jeans': 'long_pants',
        'tight_trousers': 'long_pants',
        'skirt': 'skirt', 'short_pants': 'short_pants', 'dress': 'skirt',
    }
    result['lower_type'] = lower_map.get(lower_type_raw, 'long_pants')
    result['lower_type_detail'] = lower_type_raw

    # --- 신발 유형 (argmax) ---
    shoes_type_probs = p[30:35]
    shoes_types = ['leather', 'sport', 'boots', 'cloth', 'casual']
    result['shoes_type'] = shoes_types[int(shoes_type_probs.argmax())]

    # --- 소지품 ---
    result['backpack'] = bool(p[35] > threshold)
    result['shoulder_bag'] = bool(p[36] > threshold)
    result['hand_bag'] = bool(p[37] > threshold)
    result['has_box'] = bool(p[38] > threshold)
    result['plastic_bag'] = bool(p[39] > threshold)
    result['paper_bag'] = bool(p[40] > threshold)
    result['hand_trunk'] = bool(p[41] > threshold)
    result['other_attach'] = bool(p[42] > threshold)

    # --- 방향 (argmax) ---
    dir_probs = p[43:47]
    dir_idx = int(dir_probs.argmax())
    result['direction'] = ['front', 'back', 'left', 'right'][dir_idx]

    # --- 가림 (이진) ---
    result['occluded'] = bool(any(p[47:55] > threshold))

    # --- 상의 색상 (argmax, 12색) ---
    upper_color_probs = p[55:67]
    upper_colors = ['black', 'white', 'gray', 'red', 'green', 'blue',
                    'yellow', 'brown', 'purple', 'pink', 'orange', 'mixture']
    result['upper_color'] = upper_colors[int(upper_color_probs.argmax())]

    # --- 하의 색상 (argmax, 8색) ---
    lower_color_probs = p[67:75]
    lower_colors = ['black', 'white', 'gray', 'red', 'green',
                    'blue', 'yellow', 'mixture']
    result['lower_color'] = lower_colors[int(lower_color_probs.argmax())]

    # --- 신발 색상 (argmax, 9색) ---
    shoes_color_probs = p[75:84]
    shoes_colors = ['black', 'white', 'gray', 'red', 'green',
                    'blue', 'yellow', 'brown', 'mixture']
    result['shoes_color'] = shoes_colors[int(shoes_color_probs.argmax())]

    return result
