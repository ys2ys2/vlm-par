# VLM-PAR

[English](README.md) | [한국어](README_ko.md)

### *"VLM-PAR: A Vision Language Model for Pedestrian Attribute Recognition"* 비공식 구현

[arXiv:2512.22217](https://arxiv.org/abs/2512.22217) (Sellam et al., 2025.12) 논문의 **PyTorch 구현체**입니다. 원 논문의 코드가 공개되지 않아 논문의 아키텍처 설명을 기반으로 구현했습니다.

> **참고**: [OpenPAR](https://github.com/Event-AHU/OpenPAR)나 PromptPAR를 기반으로 하지 않았습니다. 논문만을 참고한 구현입니다.

## 접근 방식

VLM-PAR는 보행자 속성 인식을 **비전-언어 정렬(vision-language alignment)** 문제로 풉니다. 사전학습된 비전-언어 모델([SigLIP 2](https://arxiv.org/abs/2502.14786))을 동결된 특징 추출기로 사용하고, **속성별 독립 Cross-Attention** 모듈만 학습하여 시각적 특징을 속성 예측에 매핑합니다.

핵심 : **동결된 VLM은 이미 의류, 액세서리, 신체 특성을 이해한다** — 특정 속성을 읽는 방법만 가르치면 됩니다.

### 아키텍처

```
이미지 (224×224)
    ↓
SigLIP 2 ViT-B/16 (동결, 학습 안 함)
    ↓ 14×14 = 196개 패치 토큰 [B, 196, 768]
    ↓
38개 독립 Cross-Attention 모듈 (학습)
    ├── CA #1:  "여성인가?"     → query가 196패치를 살펴봄 → 0.92 → 여성
    ├── CA #2:  "모자?"         → 머리 부근 패치에 집중   → 0.87 → 착용
    ├── CA #3:  "안경?"         → 얼굴 부근 패치에 집중   → 0.12 → 미착용
    ├── CA #4:  "티셔츠?"       → 상체 패치에 집중        → 0.83 → 맞음 (반팔)
    ├── ...
    ├── CA #13: "상의 검정?"    → 상체 색상 패치에 집중   → 0.08 → 아님
    ├── CA #14: "상의 흰색?"    → 상체 색상 패치에 집중   → 0.91 → 맞음
    ├── ...
    └── CA #38: "하의 혼합색?"  → 하체 패치에 집중        → 0.05 → 아님
    ↓
결과: 남성, 모자 착용, 흰색 반팔, 검정 긴바지
```

각 속성이 **고유한 query token**을 가지고, Cross-Attention으로 이미지의 관련 영역에 자동으로 집중하도록 학습됩니다.

### 기반 기술

| 기반 | 역할 | 참조 |
|------|------|------|
| **VLM-PAR** | 논문 — 아키텍처 설계 | [arXiv:2512.22217](https://arxiv.org/abs/2512.22217) |
| **SigLIP 2** | 비전 인코더 — 동결 ViT-B/16 | [arXiv:2502.14786](https://arxiv.org/abs/2502.14786) |
| **open_clip** | 라이브러리 — 모델 로딩, 전처리 | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) |
| **RAP v2** | 데이터셋 — 감시카메라 이미지 41K장, 92속성 | [dangweili/RAP](https://github.com/dangweili/RAP) |

### 논문 대비 구현

핵심 알고리즘은 논문과 최대한 같게 구현했습니다. **실제 사용자 특징 검색** 환경에 맞추기 위해 두 가지를 변경했습니다:

| 항목 | 논문 | 본 구현 | 변경 이유 |
|------|------|---------|-----------|
| Cross-Attention | 속성별 독립 | **속성별 독립** (동일) | 논문 설계 중심 |
| 속성 수 | 26개 (PA-100K) | **38개** (RAP v2 선별) | PA-100K에는 **색상 속성이 없음**. 실용적으로 의류 색상(상의 12색, 하의 8색)은 중요. RAP v2의 92개 속성 중 인물 식별에 필요한 38개를 선별. |
| 데이터셋 | PA-100K (10만장, 거리 사진) | **RAP v2** (4.1만장, **CCTV**) | PA-100K는 눈높이 보행자 사진. 우리 시스템은 **높은 각도의 CCTV 카메라**에서 동작. RAP v2는 실제 CCTV로 촬영되어 배포 환경과 일치. |
| 텍스트 초기화 | SigLIP 텍스트 인코더 | **SigLIP 텍스트 인코더** (동일) | 논문 방식 유지 — "a person wearing a hat" 같은 텍스트로 query token을 초기화하여 의미적 사전지식 제공. |
| 코드 | 미공개 | **본 저장소** | 논문(arXiv:2512.22217)이 소스코드를 공개하지 않아, 논문의 아키텍처 설명을 기반으로 독립 구현. |

### 이렇게 설계한 이유

**목표**: 감시카메라 이미지에서 보행자 속성(의류 유형, 색상, 액세서리)을 높은 정확도로 분류.

**VLM-PAR를 선택한 이유**: 논문에서 동결된 SigLIP 2 백본이 이미 의류 속성을 이해하고 있다는 것을 보여줌 — 경량 attention head만 학습하면 됨. 빠른 학습(~6시간), 작은 VRAM(~1.3GB), 높은 정확도(mA 88%).

**PA-100K 대신 RAP v2를 선택한 이유**: PA-100K는 26개 속성이지만 **색상 속성이 0개**. 실용적으로 의류 색상 정보는 중요. RAP v2는 상의 12색 + 하의 8색을 제공하며, 실제 감시카메라 촬영 이미지라 CCTV 환경에 적합.

## 학습 결과

### RAP v2 테스트셋 — mA 88.14%

| 그룹 | 속성 | mA |
|------|------|-----|
| 성별 | 여성 | **95.9%** |
| 머리 | 모자, 안경 | **87.3%** |
| 상의 유형 | 셔츠, 스웨터, 조끼, 티셔츠, 면, 자켓, 정장, 타이트, 반팔 | 85.0% |
| 상의 색상 | 검정, 흰색, 회색, 빨강, 초록, 파랑, 노랑, 갈색, 보라, 분홍, 주황, 혼합 | **89.8%** |
| 하의 유형 | 긴바지, 치마, 짧은치마, 원피스, 청바지, 타이트바지 | **91.6%** |
| 하의 색상 | 검정, 흰색, 회색, 빨강, 초록, 파랑, 노랑, 혼합 | **85.9%** |

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
  --save-dir checkpoints/vlmpar
```

단일 GPU 기준 약 6시간. 38개 Cross-Attention 헤드(~186M 파라미터)만 학습하며 SigLIP 2 백본은 동결.

### 4. 추론

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

## 38개 속성

**특징 참조용**으로 선별

| 카테고리 | 수 | 속성 |
|---------|-----|------|
| 성별 | 1 | 여성 |
| 머리 | 2 | 모자, 안경 |
| 상의 유형 | 9 | 셔츠, 스웨터, 조끼, 티셔츠, 면, 자켓, 정장, 타이트, 반팔 |
| 상의 색상 | 12 | 검정, 흰색, 회색, 빨강, 초록, 파랑, 노랑, 갈색, 보라, 분홍, 주황, 혼합 |
| 하의 유형 | 6 | 긴바지, 치마, 짧은치마, 원피스, 청바지, 타이트바지 |
| 하의 색상 | 8 | 검정, 흰색, 회색, 빨강, 초록, 파랑, 노랑, 혼합 |

## 모델 사양

| | |
|---|---|
| 아키텍처 | SigLIP 2 ViT-B/16 (동결) + 38개 독립 Cross-Attention |
| 학습 파라미터 | ~186M |
| 전체 파라미터 | ~272M (86M 동결 SigLIP) |
| 입력 | 224 × 224 RGB |
| 출력 | 38개 확률값 (sigmoid) |
| 최고 mA | **88.14%** (RAP v2) |
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
