# VLM-PAR

[English](README.md) | [한국어](README_ko.md)

### *"VLM-PAR: A Vision Language Model for Pedestrian Attribute Recognition"* 비공식 구현

[arXiv:2512.22217](https://arxiv.org/abs/2512.22217) (Sellam et al., 2025.12) 논문의 **PyTorch 구현체**입니다. 원 논문의 코드가 공개되지 않아 논문의 아키텍처 설명을 기반으로 구현했습니다.

> **참고**: [OpenPAR](https://github.com/Event-AHU/OpenPAR)나 PromptPAR를 기반으로 하지 않았습니다. 논문만을 참고한 구현입니다.

## 접근 방식

VLM-PAR는 보행자 속성 인식을 **비전-언어 정렬(vision-language alignment)** 문제로 풉니다. 논문 원본의 Cross-Attention 방향을 충실히 재현합니다: **Q=이미지, K/V=텍스트**.

사전학습된 비전-언어 모델([SigLIP 2](https://arxiv.org/abs/2502.14786))을 동결된 특징 추출기로 사용하고, 각 속성에 대해 이미지 패치 토큰이 캐시된 텍스트 임베딩을 질의하는 독립 Cross-Attention 모듈을 학습합니다.

핵심 : **각 이미지 패치가 "이 속성 설명과 내가 얼마나 관련 있는가?"를 질문한다** — 모자 패치는 "a person wearing a hat"에, 상체 패치는 "a person wearing black upper clothing"에 집중합니다.

### 아키텍처 (논문 Section 3.4)

```
이미지 (224x224)
    |
SigLIP 2 ViT-B/16 (동결, 학습 안 함)
    | 14x14 = 196개 패치 토큰 [B, 196, 768]
    |
x84 독립 Cross-Attention 모듈 (학습)
    |
    |   Q = 이미지 패치 토큰 [B, 196, 768]    <-- 이미지가 질문
    |   K = 텍스트 임베딩 [T, 768]              <-- 속성 설명 (캐시됨)
    |   V = 텍스트 임베딩 [T, 768]              <-- 속성 설명 (캐시됨)
    |
    |   head_h = softmax(Q . K^T / sqrt(96)) . V    (8 heads, d_k=96)
    |   h = LayerNorm(MultiHead + Residual)
    |   h = LayerNorm(FFN(h) + h)
    |   GAP -> Linear(768->1) -> logit
    |
    +-- CA #1:  "여성인가?"         -> 전신 패치에 attend   -> 0.92
    +-- CA #13: "모자?"             -> 머리 패치에 집중     -> 0.87
    +-- CA #55: "상의 검정?"        -> 상체 색상에 집중     -> 0.08
    +-- CA #56: "상의 흰색?"        -> 상체 색상에 집중     -> 0.91
    +-- CA #67: "하의 검정?"        -> 하체 색상에 집중     -> 0.85
    +-- ...
    +-- CA #84: "신발 혼합색?"      -> 발 부근에 집중       -> 0.05
    |
결과: 남성, 모자 착용, 흰색 반팔, 검정 긴바지, 검정 운동화
```

텍스트 임베딩은 **서버 시작 시 1회만 계산**되어 K/V 버퍼로 캐시됩니다 — 추론 시 텍스트 인코더 비용 제로.

### 기반 기술

| 기반 | 역할 | 참조 |
|------|------|------|
| **VLM-PAR** | 논문 — 아키텍처 설계 | [arXiv:2512.22217](https://arxiv.org/abs/2512.22217) |
| **SigLIP 2** | 비전 인코더 — 동결 ViT-B/16 | [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) |
| **open_clip** | 라이브러리 — 모델 로딩, 전처리 | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) |
| **RAP v2** | 데이터셋 — 감시카메라 이미지 41K장, 92속성 | [dangweili/RAP](https://github.com/dangweili/RAP) |

### 논문 대비 구현

| 항목 | 논문 | 본 구현 | 변경 이유 |
|------|------|---------|-----------|
| Cross-Attention 방향 | Q=이미지, K/V=텍스트 | **Q=이미지, K/V=텍스트** (동일) | 논문 Section 3.4 충실 재현. 각 패치가 텍스트 설명을 질의. |
| 추론 시 텍스트 인코더 | 매번 실행 | **1회 캐시** (비용=0) | 84개 속성 프롬프트는 고정. 시작 시 텍스트 임베딩 캐시. |
| 속성 수 | 26개 (PA-100K) | **84개** (RAP v2, 행동 제외) | PA-100K에는 **색상 속성이 없음**. 실제 감시카메라 환경에서 필요한 전체 속성(색상, 소지품, 방향, 가림 등) 포함. |
| 데이터셋 | PA-100K (10만장, 거리 사진) | **RAP v2** (4.1만장, **CCTV**) | RAP v2는 실제 CCTV 카메라로 촬영되어 배포 환경과 일치. |
| 손실 함수 | CE + Focal + Label Smoothing | **Focal + Label Smoothing + pos_weight** | 논문 Section 3.5 구현. 극심한 클래스 불균형 대응. |

## 84개 속성

RAP v2 전체 92개 속성에서 행동 8개를 제외한 **84개 속성**:

| 카테고리 | 수 | 속성 |
|---------|:---:|------|
| **인적 사항** | 9 | 성별, 나이(3), 체형(3), 역할(2) |
| **머리/액세서리** | 6 | 대머리, 긴머리, 검은머리, 모자, 안경, 머플러 |
| **상의 유형** | 9 | 셔츠, 스웨터, 조끼, 티셔츠, 면, 자켓, 정장, 타이트, 반팔 |
| **하의 유형** | 6 | 긴바지, 치마, 짧은치마, 원피스, 청바지, 타이트바지 |
| **신발 유형** | 5 | 가죽구두, 운동화, 부츠, 천신발, 캐주얼 |
| **소지품** | 8 | 배낭, 숄더백, 핸드백, 박스, 비닐봉지, 종이백, 캐리어, 기타 |
| **방향** | 4 | 정면, 후면, 좌측, 우측 |
| **가림** | 8 | 좌/우/상/하, 환경/소지품/사람/기타 |
| **상의 색상** | 12 | 검정, 흰색, 회색, 빨강, 초록, 파랑, 노랑, 갈색, 보라, 분홍, 주황, 혼합 |
| **하의 색상** | 8 | 검정, 흰색, 회색, 빨강, 초록, 파랑, 노랑, 혼합 |
| **신발 색상** | 9 | 검정, 흰색, 회색, 빨강, 초록, 파랑, 노랑, 갈색, 혼합 |

## 빠른 시작

### 1. 설치

```bash
pip install -r requirements.txt
```

### 2. RAP v2 데이터셋 다운로드

[RAP 데이터셋 페이지](https://www.rapdataset.com/rapv2.html)에서 신청하거나 [OpenPAR Dropbox 미러](https://www.dropbox.com/scl/fo/boipdmufnsnsvmfdle5um/AMbwWDNnlBWnVbnxxv4VcFM)에서 다운로드.

### 3. 학습

```bash
python train.py \
  --data-dir /path/to/RAPv2 \
  --epochs 50 \
  --batch-size 16 \
  --lr 3e-4 \
  --device cuda:0 \
  --save-dir checkpoints/vlmpar_v3
```

### 4. 추론

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

또는 CLI:

```bash
python inference.py --image person.jpg --checkpoint checkpoints/vlmpar_v3/vlmpar_v3_best.pth
```

## 모델 사양

| | |
|---|---|
| 아키텍처 | SigLIP 2 ViT-B/16 (동결) + 84개 독립 Cross-Attention |
| CA 방향 | **Q=이미지 패치, K/V=텍스트 임베딩** (논문 충실) |
| 학습 파라미터 | ~130M (84 CA 모듈) |
| 전체 파라미터 | ~216M (86M 동결 SigLIP) |
| 입력 | 224 x 224 RGB |
| 출력 | 84개 확률값 (sigmoid) |
| 손실 함수 | Focal Loss (gamma=2.0) + Label Smoothing (eps=0.05) + pos_weight |
| 프레임워크 | PyTorch + open_clip |

## 라이선스

| 구성요소 | 라이선스 |
|----------|---------|
| 본 코드 | [MIT](LICENSE) |
| SigLIP 2 | Apache 2.0 + CC BY 4.0 |
| RAP v2 | 연구 목적 전용 |

## 인용

본 구현을 사용하시면 원 논문을 인용해주세요:

```bibtex
@article{sellam2025vlmpar,
  title={VLM-PAR: A Vision Language Model for Pedestrian Attribute Recognition},
  author={Sellam, Abdellah Zakaria and Bekhouche, Salah Eddine and Dornaika, Fadi
          and Distante, Cosimo and Hadid, Abdenour},
  journal={arXiv preprint arXiv:2512.22217},
  year={2025}
}
```

## 감사의 글

- [VLM-PAR](https://arxiv.org/abs/2512.22217) — Sellam et al.
- [SigLIP 2](https://arxiv.org/abs/2502.14786) — Google Research
- [open_clip](https://github.com/mlfoundations/open_clip) — LAION
- [RAP Dataset](https://github.com/dangweili/RAP) — Li et al.
