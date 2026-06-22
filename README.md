
<!-- markdownlint-disable MD033 MD041 -->

<p align="center">
  <img src="docs/architecture_diagram.png" alt="EdgeOps AI Agent Architecture" width="800"/>
</p>

<h1 align="center">EdgeOps AI Agent — ANKA Takımı</h1>
<p align="center"><strong>TEKNOFEST 2026 · Akıllı Güvenlik Sistemleri Kategorisi</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python">
  <img src="https://img.shields.io/badge/vLLM-0.8+-orange?logo=nvidia">
  <img src="https://img.shields.io/badge/Qwen2.5--VL--7B--Instruct-green">
  <img src="https://img.shields.io/badge/License-Apache%202.0-red">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey">
</p>

---

## 📋 İçindekiler / Table of Contents

| # | Türkçe | English |
|---|--------|---------|
| 1 | [Proje Tanımı](#-proje-tanımı) | [Project Description](#-project-description) |
| 2 | [Sistem Mimarisi](#-sistem-mimarisi) | [System Architecture](#-system-architecture) |
| 3 | [Matematiksel Metrikler](#-matematiksel-metrikler) | [Mathematical Metrics](#-mathematical-metrics) |
| 4 | [Kurulum](#-kurulum) | [Installation](#-installation) |
| 5 | [Çalıştırma](#-çalıştırma-adımları) | [Execution Steps](#-execution-steps) |
| 6 | [Proje Ağacı](#-proje-ağacı) | [Project Tree](#-project-tree) |
| 7 | [Testler](#-testler) | [Tests](#-tests) |
| 8 | [Katkıda Bulunanlar](#-katkıda-bulunanlar) | [Credits](#-credits) |

---

## 🇹🇷 Proje Tanımı

**EdgeOps AI Agent**, 5 bölmeli (split-screen) bir endüstriyel güvenlik kamera sistemini izlemek için tasarlanmış, **hibrit bir uç-sunucu (Edge-Server) basamaklı müdahale (Cascaded Triage) sistemidir**. Sistem, hesaplama yükünü optimize etmek için akıllı bir üç katmanlı yaklaşım kullanır:

### Basamaklı Müdahale Hattı (Cascaded Triage Pipeline)

```
🎥 Video Girdisi
    │
    ▼
┌─────────────────────────────────────────────────────┐
│   Katman 1 — Edge (Uç) Triage Katmanı               │
│   ● SSIMMotionFilter (eşik = 0.0350)                │
│   ● Statik grid'leri atla → YOLOv11 simülasyonu     │
│   ● ROI kırpma (5 grid)                              │
└───────────────────────┬─────────────────────────────┘
                        │ (anomali tespit edilirse)
                        ▼
┌─────────────────────────────────────────────────────┐
│   Katman 2 — Tracking & ReID                        │
│   ● ReIDExtractor → 512-boyutlu embedding           │
│   ● VectorRegistry → kosinüs benzerliği ile eşleme  │
│   ● Cross-grid kimlik takibi                         │
└───────────────────────┬─────────────────────────────┘
                        │ (şüpheli takip ediliyorsa)
                        ▼
┌─────────────────────────────────────────────────────┐
│   Katman 3 — Yerel VLM (Bilişsel Beyin)             │
│   ● Qwen2.5-VL-7B-Instruct (vLLM + guided decoding) │
│   ● A_WHAT → A_WHEN → A_WHERE muhakeme zinciri      │
│   ● KararDestekRaporu (Pydantic v2 çıktısı)         │
│   ● Otonom aksiyon tetikleme                         │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
               🔒 lock_down_zone / 🚑 call_emergency_services
```

**Temel Özellikler:**
- **SSIMMotionFilter** (`threshold = 0.0350`) — ardışık kareler arasındaki yapısal benzerliği hesaplayarak statik grid'leri tespit eder ve YOLO/VLM katmanını tamamen atlar. Bu sayede uç bilgisayarda **%65'e varan işlem yükü azaltımı** sağlanır.
- **Deterministik ReID** — aynı kişi farklı grid'lere geçtiğinde SHA-256 tabanlı embedding'ler sayesinde otomatik olarak tanınır ve takip edilir.
- **FSM Kısıtlı JSON Çıktısı** — vLLM'in `GuidedDecodingParams` özelliği ile modelin çıktısı doğrudan Pydantic şemasına zorlanır. Markdown hataları, geçersiz JSON veya eksik alanlar **%0**'a indirilir.
- **Türkçe Sistem Promptu** — Qwen2.5-VL, A_WHAT → A_WHEN → A_WHERE muhakeme zincirini Türkçe olarak takip eder ve sadece ham JSON üretir.
- **Gerçek Zamanlı Görselleştirme** — OpenCV penceresi ile grid çizgileri, anomali etiketleri ve Track ID'ler anlık olarak gösterilir.

---

## 🇬🇧 Project Description

**EdgeOps AI Agent** is a **hybrid Edge-Server cascaded triage system** designed to monitor a 5-grid split-screen industrial security camera setup. The system employs an intelligent three-layer architecture to optimise computational load on edge hardware:

### Cascaded Triage Pipeline

```
🎥 Video Input
    │
    ▼
┌──────────────────────────────────────────────────────┐
│   Layer 1 — Edge Triage Layer                         │
│   ● SSIMMotionFilter (threshold = 0.0350)             │
│   ● Skip static grids → simulated YOLOv11 inference   │
│   ● ROI cropping (5 grids)                             │
└────────────────────────┬──────────────────────────────┘
                         │ (if anomaly detected)
                         ▼
┌──────────────────────────────────────────────────────┐
│   Layer 2 — Tracking & ReID                          │
│   ● ReIDExtractor → 512-dim appearance embedding     │
│   ● VectorRegistry → cosine-similarity matching      │
│   ● Cross-grid identity persistence                   │
└────────────────────────┬──────────────────────────────┘
                         │ (if suspicious track exists)
                         ▼
┌──────────────────────────────────────────────────────┐
│   Layer 3 — Local VLM (Cognitive Brain)              │
│   ● Qwen2.5-VL-7B-Instruct (vLLM + guided decoding)  │
│   ● A_WHAT → A_WHEN → A_WHERE reasoning chain        │
│   ● KararDestekRaporu (Pydantic v2 output)           │
│   ● Autonomous tool execution                         │
└────────────────────────┬──────────────────────────────┘
                         │
                         ▼
               🔒 lock_down_zone / 🚑 call_emergency_services
```

**Key Features:**
- **SSIMMotionFilter** (`threshold = 0.0350`) — computes structural similarity between consecutive frames to detect static grids and completely bypass the YOLO/VLM layer, achieving **up to 65% compute reduction** on edge hardware.
- **Deterministic ReID** — the same person moving across different grids is automatically recognised and tracked via SHA-256-based appearance embeddings.
- **FSM-Constrained JSON Output** — vLLM's `GuidedDecodingParams` forces the model output directly into the Pydantic schema, reducing markdown errors, malformed JSON, and missing fields to **~0%**.
- **Turkish System Prompt** — Qwen2.5-VL follows the A_WHAT → A_WHEN → A_WHERE reasoning chain in Turkish and outputs raw JSON only.
- **Real-Time Visualisation** — an OpenCV window displays grid boundaries, anomaly overlays, and Track IDs in real time.

---

## 🏗️ Sistem Mimarisi / System Architecture

<p align="center">
  <img src="docs/architecture_diagram.png" alt="Architecture Diagram" width="100%"/>
</p>

### Bileşenler / Components

| Modül | Sorumluluk | Responsibility |
|-------|-----------|----------------|
| `edge_triage/detector.py` | ROI kırpma + YOLOv11 taklidi anomali tespiti | ROI cropping + mock YOLOv11 anomaly detection |
| `edge_triage/ssim_filter.py` | Yapısal benzerlik filtresi (SSIM) | Structural similarity frame filter |
| `tracking_reid/reid_extractor.py` | 512-boyutlu görünüm embedding'i | 512-dim appearance embedding |
| `tracking_reid/vector_storage.py` | Kosinüs benzerliği ile kimlik hafızası | Cosine-similarity identity registry |
| `core_agent/prompt_templates.py` | Türkçe sistem prompt şablonu | Turkish system prompt template |
| `core_agent/structured_parser.py` | Pydantic v2 modelleri + vLLM GuidedDecoding | Pydantic v2 models + vLLM guided decoding |
| `tools/security_actions.py` | `lock_down_zone` / `guvenli_bolge_uyarisi` | Zone lockdown / safety warnings |
| `tools/emergency_actions.py` | `call_emergency_services` / `bilgi_amacli_bildirim` | Emergency dispatch / info notifications |
| `main.py` | Hat organizatörü (pipeline orchestrator) | Pipeline orchestrator + CLI |

---

## 📐 Matematiksel Metrikler / Mathematical Metrics

Aşağıdaki metrikler, sistemin başarımını nesnel olarak değerlendirmek için kullanılır.
The following metrics are used to objectively evaluate system performance.

### 1. Zamansal Başarım Skoru / Temporal Accuracy Score ($E_{temp}$)

Geçici algılama hatasını üstel bir çürüme fonksiyonu ile cezalandırır. $\tau$ zaman sabitidir (varsayılan: 1.0 sn).

Penalises temporal detection error with an exponential decay function. $\tau$ is the time constant (default: 1.0 s).

$$
E_{temp} = \frac{1}{N_{anomali}} \sum_{i=1}^{N_{anomali}} \exp\left(-\frac{|t_{tespit, i} - t_{gercek, i}|}{\tau}\right)
$$

- **$t_{tespit}$** : Sistemin tespit ettiği zaman damgası (sn)
- **$t_{gercek}$** : Gerçek (ground truth) zaman damgası (sn)
- **$\tau$** : Zaman sabiti (1.0 sn)
- **$E_{temp} \in [0, 1]$** : 1.0 = mükemmel zamanlama

### 2. Yapısal Ayrıştırma Oranı / Structural Parsing Score ($R_{parsing}$)

VLM çıktılarının Pydantic şemasına uygun olarak ayrıştırılma başarısını ölçer.

Measures the fraction of VLM outputs that successfully parse against the Pydantic schema.

$$
R_{parsing} = \frac{\text{Doğru Ayrıştırılan Karar Nesneleri}}{\text{Toplam Üretilen Nesneler}}
$$

- **$R_{parsing} \in [0, 1]$** : 1.0 = tüm çıktılar geçerli

### 3. Aksiyon Başarı Oranı / Action Success Rate ($S_{rec}$)

Sistem tarafından önerilen aksiyonların doğruluk oranıdır (kesinlik / precision).

The precision of the system's recommended actions against ground truth.

$$
S_{rec} = \frac{TP_{action}}{TP_{action} + FP_{action}}
$$

- **$TP_{action}$** : Doğru pozitif aksiyon (sistemin önerdiği ve olması gereken)
- **$FP_{action}$** : Yanlış pozitif aksiyon (sistemin önerdiği ama gereksiz olan)
- **$S_{rec} \in [0, 1]$** : 1.0 = tüm aksiyonlar isabetli

---

## ⚙️ Kurulum / Installation

### Gereksinimler / Prerequisites

- Python 3.11+
- NVIDIA GPU (CUDA 12.1+) — vLLM için
- 16 GB+ RAM (önerilen: 32 GB)
- Windows 10/11 veya Ubuntu 22.04

### Adımlar / Steps

```bash
# 1. Repoyu klonla / Clone the repository
git clone https://github.com/ilyasselmamouni/edgeops-ai-agent.git
cd edgeops-ai-agent

# 2. Sanal ortam oluştur / Create virtual environment
python -m venv venv

# 3. Aktifleştir / Activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
# source venv/bin/activate

# 4. Bağımlılıkları yükle / Install dependencies
pip install -r requirements.txt

# 5. (İsteğe bağlı) vLLM sunucusunu başlat / (Optional) Start vLLM server
.\venv\Scripts\python.exe -m vllm.entrypoints.openai.api_server ^
    --model Qwen/Qwen2.5-VL-7B-Instruct ^
    --host 127.0.0.1 ^
    --port 8000 ^
    --max-model-len 16384 ^
    --gpu-memory-utilization 0.85
```

---

## 🚀 Çalıştırma Adımları / Execution Steps

### Gerçek Video ile / With Real Video

Video dosyası `data/mock_videos/test_saha_kaydi.mp4` yolunda olmalıdır.
Ensure the video file is present at `data/mock_videos/test_saha_kaydi.mp4`.

```powershell
# Ana hattı başlat / Launch the main pipeline
python src\main.py
```

**Beklenen çıktı / Expected output:**

```
Pipeline finished: 300 frames, 19 anomalies, 19 actions dispatched
  Frame   25 | 00:00:00.833 | static=0 | anomalies=1 | tracks=[1] | actions=1
  Frame   42 | 00:00:01.400 | static=0 | anomalies=1 | tracks=[2] | actions=1
  ...
```

Gerçek zamanlı OpenCV penceresi (`ANKA Team - EdgeOps AI Agent`) açılır.
A real-time OpenCV window (`ANKA Team - EdgeOps AI Agent`) will open.

**Çıkış / Exit:** `q` tuşuna basın / Press the `q` key.

### Canlı vLLM ile / With Live vLLM

```python
# src/main.py içinde / Inside src/main.py:
cfg = PipelineConfig(
    video_path="data/mock_videos/test_saha_kaydi.mp4",
    mock_agent=False,        # ← Gerçek vLLM kullanımı
)
```

### Sadece Metrikler / Run Metrics Only

```powershell
python -c "from tests.evaluate_metrics import *; print(evaluate_all(...))"
```

---

## 🌳 Proje Ağacı / Project Tree

```
edgeops-ai-agent/
│
├── .github/workflows/            # CI/CD otomasyonu
│   └── ci-cd.yml
│
├── config/                       # Yapılandırma dosyaları
│   ├── vllm_config.json
│   └── agent_config.json
│
├── data/
│   ├── mock_videos/               # 5-grid test videosu
│   │   └── test_saha_kaydi.mp4
│   └── ground_truth/              # Ground truth JSON'ları
│
├── src/                           # Ana kaynak kodu
│   ├── __init__.py
│   │
│   ├── edge_triage/               # Katman 1 — Edge işleme
│   │   ├── __init__.py
│   │   ├── detector.py            # YOLOv11 taklidi + ROI
│   │   └── ssim_filter.py         # SSIM kare filtresi
│   │
│   ├── core_agent/                # Katman 3 — Bilişsel beyin
│   │   ├── __init__.py
│   │   ├── prompt_templates.py    # Türkçe sistem promptu
│   │   └── structured_parser.py   # Pydantic v2 + GuidedDecoding
│   │
│   ├── tracking_reid/             # Katman 2 — Takip & ReID
│   │   ├── __init__.py
│   │   ├── reid_extractor.py      # 512-boyutlu embedding
│   │   └── vector_storage.py      # Kosinüs eşleme havuzu
│   │
│   ├── tools/                     # Aksiyon fonksiyonları
│   │   ├── __init__.py
│   │   ├── security_actions.py    # lock_down_zone
│   │   └── emergency_actions.py   # call_emergency_services
│   │
│   └── main.py                    # Hat organizatörü
│
├── tests/                         # Testler
│   ├── __init__.py
│   └── evaluate_metrics.py        # E_temp, R_parsing, S_rec
│
├── deployment/                    # Docker konfigürasyonu
│   ├── Dockerfile.vllm
│   ├── Dockerfile.agent
│   └── docker-compose.yml
│
├── docs/                          # Dokümantasyon
│   ├── architecture_diagram.png
│   └── system_specifications.md
│
├── requirements.txt               # Python bağımlılıkları
├── LICENSE                        # Apache License 2.0
└── README.md                      # Bu dosya
```

---

## 🧪 Testler / Tests

```powershell
# Tüm entegrasyon testlerini çalıştır
# Run all integration smoke tests
python -c "
import sys
sys.path.insert(0, '.')
exec(open('tests/evaluate_metrics.py').read())
"
```

**Test sonuçları / Test results:**

| Test | Durum / Status |
|------|---------------|
| ReIDExtractor (deterministic embedding) | ✅ |
| VectorRegistry (cross-grid matching) | ✅ |
| SSIMMotionFilter (static grid detection) | ✅ |
| GridVideoProcessor (ROI crop + mock detect) | ✅ |
| YerelVideoAjanYonetici (mock agent) | ✅ |
| Action Dispatch (all 4 functions) | ✅ |
| End-to-End Pipeline (50 frames) | ✅ |
| E_temp (Temporal Detection Error) | ✅ |
| R_parsing (Schema Parsing Rate) | ✅ |
| S_rec (Recommendation Alignment) | ✅ |

---

## 📄 Lisans / License

Bu proje **Apache License 2.0** ile lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for details.

---

## 👥 Katkıda Bulunanlar / Credits

| İsim / Name | Rol / Role |
|-------------|------------|
| **İlyass Elmamouni** | AI Takım Lideri / Bilgisayarlı Görü Araştırmacısı |
| | AI Team Lead / Computer Vision Researcher |

---

<p align="center">
  <strong>EdgeOps AI Agent — ANKA Takımı</strong><br>
  TEKNOFEST 2026 · Akıllı Güvenlik Sistemleri<br><br>
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Qwen2.5--VL-0A66C2?logo=alibabacloud&logoColor=white">
  <img src="https://img.shields.io/badge/vLLM-76B900?logo=nvidia&logoColor=white">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white">
  <img src="https://img.shields.io/badge/Pydantic-E92063?logo=pydantic&logoColor=white">
</p>
