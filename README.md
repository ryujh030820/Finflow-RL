# BIPD: Bio-Inspired Portfolio Defense

**강화학습 기반 면역 시스템 모델을 활용한 주식 포트폴리오 리스크 관리 시스템**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

---

## 📋 프로젝트 개요

**BIPD (Bio-Inspired Portfolio Defense)**는 생체 면역 시스템의 메커니즘을 모방한 강화학습 기반 주식 포트폴리오 관리 시스템입니다.

이 프로젝트는 **B-Cell**과 **T-Cell** 에이전트를 활용하여:

-   📈 **포트폴리오 수익률 극대화**
-   🛡️ **리스크 안정적 제어**
-   🧠 **면역 시스템 기반 적응형 학습**
-   📊 **실시간 시장 변화 대응**

을 목표로 합니다.

---

## 🏗️ 프로젝트 구조

```
BIPD/
├── 📁 agents/                    # 면역 시스템 에이전트
│   ├── base.py                   # 기본 에이전트 클래스
│   ├── bcell.py                  # B-Cell 에이전트 (항체 생성)
│   ├── tcell.py                  # T-Cell 에이전트 (면역 조절)
│   └── memory.py                 # 면역 기억 시스템
│
├── 📁 core/                      # 핵심 시스템 모듈
│   ├── backtester.py             # 백테스팅 엔진
│   ├── hierarchical.py           # 계층적 의사결정 시스템
│   ├── reward.py                 # 보상 함수 설계
│   └── system.py                 # 면역 시스템 통합 관리
│
├── 📁 xai/                       # 설명 가능한 AI
│   ├── analyzer.py               # 의사결정 분석기
│   ├── dashboard.py              # 대시보드 생성
│   └── visualization.py          # 시각화 도구
│
├── 📁 data/                      # 데이터 저장소
│   ├── csv_output/               # CSV 형태 데이터
│   ├── *.pkl                     # 피클 형태 시장 데이터
│   └── pkl_to_csv_converter.py   # 데이터 변환기
│
├── 📁 results/                   # 실험 결과
│   └── analysis_*/               # 분석 결과별 디렉토리
│       ├── *.json                # 분석 결과 데이터
│       ├── *.html                # 대시보드 파일
│       ├── *.png                 # 시각화 이미지
│       └── models/               # 학습된 모델
│
├── 📁 models/                    # 모델 저장소
├── main.py                       # 메인 실행 파일
├── constant.py                   # 상수 및 설정
└── README.md                     # 프로젝트 문서
```

---

## 🚀 주요 기능

### 🧬 면역 시스템 모델링

-   **B-Cell 에이전트**: 시장 패턴 인식 및 항체(투자 전략) 생성
-   **T-Cell 에이전트**: 면역 반응 조절 및 리스크 관리
-   **면역 기억**: 과거 시장 경험을 통한 학습 및 적응

### 📊 고급 분석 도구

-   **실시간 백테스팅**: 과거 데이터 기반 성능 검증
-   **계층적 의사결정**: 다층 구조의 투자 결정 시스템
-   **XAI 대시보드**: 투자 결정 과정의 투명한 시각화

### 🎯 리스크 관리

-   **동적 포트폴리오 조정**: 시장 변화에 따른 실시간 리밸런싱
-   **다중 자산 클래스**: 주식, 채권, 상품 등 다양한 자산 지원
-   **리스크 메트릭**: VaR, CVaR, 샤프 비율 등 고급 리스크 지표

---

## 📝 사용법

### 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 데이터 디렉토리 생성
python constant.py
```

### 2. 기본 실행

```bash
# 메인 시스템 실행
python main.py
```

### 3. 커스텀 설정

```python
# main.py에서 설정 수정
symbols = ["AAPL", "MSFT", "GOOGL", ...]  # 투자 대상 종목
train_start = "2008-01-02"                # 훈련 시작일
test_start = "2021-01-01"                 # 테스트 시작일
```

---

## 📊 결과 분석

실행 후 `results/` 디렉토리에서 다음 결과를 확인할 수 있습니다:

-   **📈 성능 지표**: 수익률, 샤프 비율, 최대 낙폭
-   **🎨 시각화**: 포트폴리오 성과 차트, 면역 시스템 상태
-   **📋 대시보드**: HTML 형태의 인터랙티브 분석 도구
-   **💾 모델**: 학습된 면역 시스템 상태

---

## 🔬 기술적 특징

### 강화학습 알고리즘

-   **PPO (Proximal Policy Optimization)**: 안정적인 정책 학습
-   **A3C (Asynchronous Advantage Actor-Critic)**: 비동기 학습
-   **Experience Replay**: 효율적인 경험 활용

### 면역 시스템 메타포

-   **항원 인식**: 시장 패턴 및 이상 징후 탐지
-   **항체 생성**: 적응형 투자 전략 개발
-   **면역 기억**: 장기적 시장 경험 축적
-   **면역 조절**: 과도한 반응 방지 및 균형 유지

---

## ⚠️ 주의사항

> [!NOTE]
> 본 시스템은 연구 및 교육 목적으로 개발되었습니다. 실제 투자에 사용하기 전에 충분한 검증과 백테스팅을 수행하시기 바랍니다. 과거 성과가 미래 수익을 보장하지 않으며, 투자 손실의 위험이 있습니다.

### 시스템 요구사항

-   **Python**: 3.8 이상
-   **메모리**: 최소 8GB RAM 권장
-   **GPU**: CUDA 지원 GPU (선택사항, 학습 속도 향상)

### 데이터 요구사항

-   **시장 데이터**: 최소 5년 이상의 일별 OHLCV 데이터
-   **데이터 품질**: 결측값 및 이상치 처리 필요

---

## 📄 라이센스

이 프로젝트는 **MIT 라이센스** 하에 배포됩니다.

```
MIT License

Copyright (c) 2025 BIPD Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
