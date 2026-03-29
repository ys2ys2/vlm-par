#!/usr/bin/env python3
"""
VLM-PAR v3: 논문 원본 구조 — 속성별 독립 Cross-Attention

arXiv:2512.22217 원본 설계:
- SigLIP 2 ViT-B/16 (frozen)
- 속성별 독립 Cross-Attention (38개)
- 속성당 이진 분류 head

38속성 (실종자 검색 필수):
- 성별 (1), 모자 (1), 안경 (1)
- 상의 유형 9종, 상의 색상 12색
- 하의 유형 6종, 하의 색상 8색

VRAM: ~1.3GB (batch 16)
License: MIT (코드) + Apache 2.0 / CC BY 4.0 (SigLIP 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import logging
import numpy as np
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================
# 38속성 정의
# ============================================================

ATTR_NAMES = [
    # 성별 (1)
    'female',
    # 머리 (2)
    'hat', 'glasses',
    # 상의 유형 (9)
    'ub_shirt', 'ub_sweater', 'ub_vest', 'ub_tshirt', 'ub_cotton',
    'ub_jacket', 'ub_suit', 'ub_tight', 'ub_short_sleeve',
    # 상의 색상 (12)
    'up_black', 'up_white', 'up_gray', 'up_red', 'up_green', 'up_blue',
    'up_yellow', 'up_brown', 'up_purple', 'up_pink', 'up_orange', 'up_mixture',
    # 하의 유형 (6)
    'lb_long_trousers', 'lb_skirt', 'lb_short_skirt', 'lb_dress', 'lb_jeans', 'lb_tight_trousers',
    # 하의 색상 (8)
    'low_black', 'low_white', 'low_gray', 'low_red', 'low_green',
    'low_blue', 'low_yellow', 'low_mixture',
]
NUM_ATTRS = len(ATTR_NAMES)
assert NUM_ATTRS == 38

# RAP v2 annotation 인덱스
RAP_INDICES = [
    0,        # Female
    12, 13,   # Hat, Glasses
    15, 16, 17, 18, 19, 20, 21, 22, 23,  # Upper type (9)
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,  # Upper color (12)
    24, 25, 26, 27, 28, 29,  # Lower type (6)
    75, 76, 77, 78, 79, 80, 81, 82,  # Lower color (8)
]
assert len(RAP_INDICES) == 38

# 텍스트 프롬프트 (SigLIP text encoder 초기화용)
TEXT_PROMPTS = [
    "a female person",
    "a person wearing a hat or cap",
    "a person wearing glasses",
    "a person wearing a shirt",
    "a person wearing a sweater",
    "a person wearing a vest",
    "a person wearing a t-shirt",
    "a person wearing cotton clothing",
    "a person wearing a jacket",
    "a person wearing a suit",
    "a person wearing tight upper clothing",
    "a person wearing short sleeves",
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
    "a person wearing long trousers",
    "a person wearing a skirt",
    "a person wearing a short skirt or shorts",
    "a person wearing a dress",
    "a person wearing jeans",
    "a person wearing tight trousers or leggings",
    "a person wearing black lower clothing",
    "a person wearing white lower clothing",
    "a person wearing gray lower clothing",
    "a person wearing red lower clothing",
    "a person wearing green lower clothing",
    "a person wearing blue lower clothing",
    "a person wearing yellow lower clothing",
    "a person wearing mixed color lower clothing",
]
assert len(TEXT_PROMPTS) == 38


# ============================================================
# 속성별 독립 Cross-Attention (논문 원본)
# ============================================================

class AttributeCrossAttention(nn.Module):
    """단일 속성을 위한 독립 Cross-Attention 블록

    Query: 학습 가능한 속성 토큰 [1, dim]
    Key/Value: 이미지 패치 토큰 [B, 196, dim]
    Output: 이진 분류 logit [B, 1]
    """

    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 1)

    def forward(self, img_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_tokens: [B, 196, dim]
        Returns:
            logit: [B, 1]
        """
        B = img_tokens.shape[0]
        q = self.query_token.expand(B, -1, -1)  # [B, 1, dim]

        # Cross-Attention: query attends to image patches
        attn_out, _ = self.cross_attn(q, img_tokens, img_tokens)
        x = self.norm1(attn_out + q)  # residual

        # FFN
        x = self.norm2(self.ffn(x) + x)  # residual

        # Classification
        return self.head(x.squeeze(1))  # [B, 1]


class VLM_PAR(nn.Module):
    """VLM-PAR v3: 38개 속성별 독립 Cross-Attention

    논문 원본 구조. 각 속성이 고유한 query token + CA를 가짐.
    """

    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attr_cas = nn.ModuleList([
            AttributeCrossAttention(dim, num_heads, dropout)
            for _ in range(NUM_ATTRS)
        ])

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: [B, 196, 768]
        Returns:
            logits: [B, 38]
        """
        logits = [ca(patch_tokens) for ca in self.attr_cas]  # list of [B, 1]
        return torch.cat(logits, dim=1)  # [B, 38]

    def init_queries_from_text(self, siglip_model, tokenizer, device='cuda'):
        """SigLIP text encoder로 각 속성의 query token 초기화"""
        try:
            tokens = tokenizer(TEXT_PROMPTS).to(device)
            with torch.no_grad():
                text_features = siglip_model.encode_text(tokens)  # [38, 768]
                text_features = F.normalize(text_features, dim=-1)

            for i, ca in enumerate(self.attr_cas):
                ca.query_token.data = text_features[i:i+1].unsqueeze(0)  # [1, 1, 768]

            logger.info(f"All 38 query tokens initialized from SigLIP text encoder")
            return True
        except Exception as e:
            logger.warning(f"Text init failed: {e}, using random init")
            return False


# ============================================================
# Wrapper
# ============================================================

class VLMPARWrapper(nn.Module):
    """SigLIP 2 (frozen) + VLM-PAR v3 (38 독립 CA)"""

    def __init__(self, device: str = 'cuda:0'):
        super().__init__()
        self.device = device

        # SigLIP 2 backbone (frozen)
        logger.info("Loading SigLIP 2 ViT-B/16 (frozen)...")
        siglip, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-16-SigLIP2', pretrained='webli'
        )
        self.siglip = siglip.to(device).eval()
        for p in self.siglip.parameters():
            p.requires_grad = False

        # PAR head
        self.par_head = VLM_PAR(dim=768, num_heads=8, dropout=0.1).to(device)

        trainable = sum(p.numel() for p in self.par_head.parameters() if p.requires_grad)
        logger.info(f"VLM-PAR v3: {trainable:,} trainable params (38 independent CAs)")

        # Text init
        try:
            tokenizer = open_clip.get_tokenizer('ViT-B-16-SigLIP2')
            self.par_head.init_queries_from_text(self.siglip, tokenizer, device)
        except Exception as e:
            logger.warning(f"Text init failed: {e}")

    def extract_patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """SigLIP 2에서 패치 토큰 추출"""
        with torch.no_grad():
            visual = self.siglip.visual
            trunk = visual.trunk
            x = trunk.patch_embed(images)
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
        """추론: 38속성 → 구조화된 Dict"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for b in range(probs.shape[0]):
            p = probs[b]

            # 성별
            gender = 'female' if p[0] > threshold else 'male'
            # 모자, 안경
            hat = bool(p[1] > threshold)
            glasses = bool(p[2] > threshold)

            # 상의 유형 (argmax 3~11)
            upper_type_probs = p[3:12]
            upper_types = ['shirt', 'sweater', 'vest', 'tshirt', 'cotton', 'jacket', 'suit', 'tight', 'short_sleeve']
            upper_type_raw = upper_types[int(upper_type_probs.argmax())]
            upper_type = 'short_sleeve' if upper_type_raw in ('tshirt', 'vest', 'short_sleeve') else 'long_sleeve'

            # 상의 색상 (argmax 12~23)
            upper_color_probs = p[12:24]
            upper_colors = ['black', 'white', 'gray', 'red', 'green', 'blue', 'yellow', 'brown', 'purple', 'pink', 'orange', 'mixture']
            upper_color = upper_colors[int(upper_color_probs.argmax())]

            # 하의 유형 (argmax 24~29)
            lower_type_probs = p[24:30]
            lower_types = ['long_pants', 'skirt', 'short_pants', 'dress', 'jeans', 'tight_trousers']
            lower_type_raw = lower_types[int(lower_type_probs.argmax())]
            lower_map = {'long_pants': 'long_pants', 'jeans': 'long_pants', 'tight_trousers': 'long_pants',
                         'skirt': 'skirt', 'short_pants': 'short_pants', 'dress': 'skirt'}
            lower_type = lower_map.get(lower_type_raw, 'long_pants')

            # 하의 색상 (argmax 30~37)
            lower_color_probs = p[30:38]
            lower_colors = ['black', 'white', 'gray', 'red', 'green', 'blue', 'yellow', 'mixture']
            lower_color = lower_colors[int(lower_color_probs.argmax())]

            results.append({
                'gender': gender,
                'hat': hat,
                'glasses': glasses,
                'upper_type': upper_type,
                'upper_type_detail': upper_type_raw,
                'upper_color': upper_color,
                'lower_type': lower_type,
                'lower_type_detail': lower_type_raw,
                'lower_color': lower_color,
                'raw_probs': {name: float(p[i]) for i, name in enumerate(ATTR_NAMES)},
            })

        return results
