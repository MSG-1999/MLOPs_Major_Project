<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,25&height=200&section=header&text=DreamForge&fontSize=72&fontAlignY=35&desc=End-to-End%20Text-to-Image%20Generation%20Studio&descAlignY=60&descSize=20&fontColor=ffffff&animation=fadeIn" width="100%"/>

<br/>

```
██████╗ ██████╗ ███████╗ █████╗ ███╗   ███╗███████╗ ██████╗ ██████╗  ██████╗ ███████╗
██╔══██╗██╔══██╗██╔════╝██╔══██╗████╗ ████║██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
██║  ██║██████╔╝█████╗  ███████║██╔████╔██║█████╗  ██║   ██║██████╔╝██║  ███╗█████╗  
██║  ██║██╔══██╗██╔══╝  ██╔══██║██║╚██╔╝██║██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝  
██████╔╝██║  ██║███████╗██║  ██║██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
```

<br/>

> *Production-grade generative AI — LoRA fine-tuning · ASHA HPO · Magnitude Pruning · Full MLOps stack*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Stable%20Diffusion](https://img.shields.io/badge/Stable_Diffusion-v1.5-8B5CF6?style=for-the-badge&logo=buffer&logoColor=white)](https://huggingface.co/stable-diffusion-v1-5)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Diffusers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Optuna](https://img.shields.io/badge/Optuna-ASHA_HPO-3DC1D3?style=for-the-badge)](https://optuna.org)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![IIT Jodhpur](https://img.shields.io/badge/IIT-Jodhpur-FF6B35?style=for-the-badge)](https://iitj.ac.in)

<br/>

---

### 🏛 Built at Indian Institute of Technology Jodhpur
**Aditya Pratap Singh** `M25CSA002` &nbsp;·&nbsp; **Gadiya Mahek Shankesh** `M25CSA011`
*M.Tech Artificial Intelligence · 2024*

---

</div>

<br/>

## ⚡ Results at a Glance

<div align="center">

| | Metric | Result | Detail |
|:---:|:---|:---:|:---|
| 🎨 | **LoRA Fine-Tuning** (r = 16) | **−27.1% loss** | vs. zero-shot SD v1.5 baseline |
| ⚡ | **ASHA Hyperparameter Search** | **4.8× faster** | vs. exhaustive grid search |
| ✂️ | **Magnitude Pruning** (20% sparsity) | **−18% memory** | with < 1.3% quality impact |
| 🔬 | **Trainable Parameters** | **3.4 M / 860 M** | < 0.4% of the UNet backbone |
| 📊 | **MLOps Overhead** | **< 0.3 s/gen** | Negligible in production |
| 🗂️ | **Dataset** | **3,500+ images** | 7 Indian festival categories |

</div>

<br/>

---

## 📖 Table of Contents

<div align="center">

[Overview](#-overview) · [Architecture](#-system-architecture) · [Quick Start](#-quick-start) · [LoRA Fine-Tuning](#-lora-fine-tuning) · [ASHA HPO](#-asha-hyperparameter-optimisation) · [Pruning](#-post-training-pruning) · [MLOps Stack](#-mlops-stack) · [Results](#-experimental-results) · [Project Structure](#-project-structure) · [Configuration](#-configuration-reference) · [Roadmap](#-roadmap) · [Citation](#-citation)

</div>

<br/>

---

## 🔭 Overview

Large-scale latent diffusion models excel at broad visual synthesis — but they systematically **under-represent culturally specific imagery**. Generic prompts to Stable Diffusion v1.5 fail to reproduce:

- 🪔 The warm glow of **Diwali diyas** and rangoli patterns
- 🎨 The chromatic chaos of **Holi** colour explosions  
- 🐘 The grandeur of **Ganesh Chaturthi** processions

**DreamForge** solves this through a complete, reproducible, production-ready MLOps pipeline:

```
┌──────────────┐     ┌────────────────┐     ┌──────────┐     ┌─────────┐     ┌───────────┐     ┌─────────────┐
│   Dataset    │────▶│ LoRA Fine-Tune │────▶│ ASHA HPO │────▶│ Pruning │────▶│ Inference │────▶│  Monitoring │
│              │     │                │     │          │     │         │     │           │     │             │
│ IndianFests  │     │  PEFT on UNet  │     │  Optuna  │     │ L1-norm │     │ Streamlit │     │ MLflow +    │
│  3,500+ imgs │     │  attention     │     │  + ASHA  │     │ pruning │     │ UI + API  │     │ Prometheus  │
└──────────────┘     └────────────────┘     └──────────┘     └─────────┘     └───────────┘     └─────────────┘
```

<br/>

---

## 🏗 System Architecture

DreamForge is decomposed into **four clean tiers**:

```
╔═══════════════════════════════════════════════════════════════════════╗
║                         FRONTEND TIER                                 ║
║   Streamlit UI (port 8503)  ·  Plotly Dashboards                     ║
║   Generation Controls  ·  Session Gallery  ·  MLOps Monitoring Tab   ║
╠═══════════════════════════════════════════════════════════════════════╣
║                         PIPELINE TIER                                 ║
║   ModelManager (Singleton + threading.Lock + scheduler hot-swap)      ║
║   InferenceEngine  ·  PromptProcessor  ·  ImageProcessor             ║
║   BatchProcessor (PriorityQueue + async worker)                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║                        MONITORING TIER                                ║
║   MetricsCollector (Prometheus, port 8012)                            ║
║   MLflowTracker (runs, params, images, artifacts)                     ║
║   DriftDetector (sliding-window statistical alerts)                   ║
╠═══════════════════════════════════════════════════════════════════════╣
║                         TRAINING TIER                                 ║
║   LoRATrainer (PEFT · AMP · cosine LR · noise offset)                ║
║   hpo_objective (Optuna + ASHA SuccessiveHalvingPruner)               ║
║   data.py  ·  prepare_mini_data.py                                    ║
╚═══════════════════════════════════════════════════════════════════════╝
```

### 🧩 ModelManager

- **Singleton pattern** with `threading.Lock` — guaranteed single model load per process
- Device resolution: `CUDA → Apple MPS → CPU` with `float32` coercion on CPU
- Three-level fallback during `from_pretrained` for models lacking `fp16` safetensors
- **Hot-swap schedulers** at runtime: `DDIM · Euler-A · DPM++ 2M · PNDM · LMS · DPM++ SDE`

### ⚙️ InferenceEngine

- Accepts `GenerationConfig` dataclass → always returns `GenerationResult` (success or failure)
- Explicit CUDA OOM handling with `torch.cuda.empty_cache()` + actionable error messages
- The UI layer **never receives an unhandled exception**

<br/>

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+ · CUDA 12.1+ recommended
pip install -r requirements.txt
```

### 🐳 Docker Compose (Recommended)

```bash
git clone https://github.com/your-org/dreamforge.git
cd dreamforge
docker compose up --build
```

| Service | URL | Description |
|:---|:---|:---|
| 🖥️ Streamlit UI | `http://localhost:8503` | Main generation interface |
| 📊 MLflow UI | `http://localhost:5012` | Experiment tracking dashboard |
| 📈 Prometheus | `http://localhost:8012` | Raw metrics export endpoint |

### 🖥️ Local Development

```bash
# Step 1 — Prepare full dataset
python training/data.py

# Step 2 — (Optional) Mini-dataset: 100 images/class for rapid HPO
python training/prepare_mini_data.py

# Step 3 — Run hyperparameter search (ASHA + Optuna)
python training/hpo_objective.py --n-trials 10

# Step 4 — Fine-tune with the best discovered config
python training/train.py --config configs/best_trial.yaml

# Step 5 — Prune the LoRA adapter (recommended: 20% sparsity)
python training/prune.py --sparsity 0.20

# Step 6 — Launch the interactive UI
streamlit run app/main.py
```

<br/>

---

## 🎨 LoRA Fine-Tuning

### Dataset: IndianFestivals

Fine-tuned on [`AIMLOps-C4-G16/IndianFestivals`](https://huggingface.co/datasets/AIMLOps-C4-G16/IndianFestivals) — **3,500+ images** across 7 festival categories with richly engineered prompts:

<div align="center">

| Festival | Emoji | Prompt Themes |
|:---|:---:|:---|
| **Diwali** | 🪔 | Clay oil diyas · fireworks · rangoli patterns · golden illumination |
| **Holi** | 🎨 | Coloured powder · vibrant spring celebration · joyful crowds |
| **Eid** | 🕌 | Mosque architecture · crescent moon · traditional attire · lanterns |
| **Ganesh Chaturthi** | 🐘 | Lord Ganesha idol · colourful decorations · traditional worship |
| **Independence Day** | 🇮🇳 | National flag · patriotic parade · saffron–white–green |
| **Lohri** | ☀️ | Bonfire · folk dance · winter harvest celebration |
| **Christmas** | 🎄 | Festive lights · decorated trees · winter celebration |

</div>

> **Split:** 80% train / 10% val / 10% test — stratified by class  
> **Mini-dataset:** 100 images/class (700 total) for rapid HPO iterations

### 📐 Mathematical Formulation

LoRA injects trainable rank-decomposed matrices into UNet attention layers, leaving the 860 M-parameter backbone **completely frozen**:

```
h = W₀x + (α/r) · B·A·x

Where:
  W₀  ∈ ℝᵈˣᵏ  ── frozen pre-trained weight (never updated)
  A   ∈ ℝʳˣᵏ  ── initialised from 𝒩(0, σ²)
  B   ∈ ℝᵈˣʳ  ── initialised to zero  ← zero-delta initialisation
  r           ── rank  (controls adapter capacity)
  α           ── scaling factor  (α/r = scale = 2.0)
```

**Target layers** — 6 per UNet transformer block:

```
to_q  ·  to_k  ·  to_v  ·  to_out.0  ·  ff.net.0.proj  ·  ff.net.2
```

### 🏆 Best Trial Configuration

<div align="center">

| Parameter | Value | Notes |
|:---|:---:|:---|
| LoRA Rank `r` | **16** | Also evaluated: 4, 8 |
| LoRA Alpha `α` | **32** | Scale = α/r = 2.0 |
| Dropout | 0.05 | Applied to LoRA layers only |
| Trainable Parameters | **~3.4 M** | vs. 860 M frozen UNet |
| Learning Rate | 1.34 × 10⁻⁴ | AdamW + cosine schedule |
| Warmup Ratio | 0.05 | 5% of total training steps |
| Batch Size | 4 | Per-GPU |
| Gradient Accumulation | 2 | Effective batch = 8 |
| Mixed Precision | fp16 | `torch.cuda.amp.GradScaler` |
| Noise Offset | 0.05 | Improves dark-scene fidelity |
| Epochs | 10 | Best checkpoint saved by val loss |

</div>

### 🎯 Training Objective

Denoising score-matching loss in latent space (following Rombach et al.):

```
ℒ = 𝔼_{z₀, ε, t, c} [ ‖ ε − εθ(zₜ, t, τθ(c)) ‖² ]

Where:
  z₀ = E(x)     ── VAE-encoded latent  (VAE frozen)
  ε ~ 𝒩(0, I)   ── ground-truth noise
  τθ(c)         ── CLIP text embedding  (CLIP frozen)
  εθ            ── UNet denoiser  (only LoRA params updated)
```

> The VAE encoder, VAE decoder, and CLIP text encoder are **all frozen**. Only LoRA parameters inside the UNet are updated — fewer than 0.4% of total parameters.

<br/>

---

## ⚡ ASHA Hyperparameter Optimisation

### Algorithm

Asynchronous Successive Halving (ASHA) launches many trials concurrently and **early-stops unpromising ones**, eliminating the synchronisation barrier of standard SHA. A trial at rung `k` is promoted only if its metric ranks in the top `1/η` fraction of all trials reaching that rung.

```
Budget B, minimum resources r_min, halving rate η = 3
Rungs S = ⌊log_η(B / r_min)⌋

Trial promoted at rung k  iff  metric rank ≤ η^{-(S-k)}
```

### 🔍 Search Space

```python
r        ∈  { 4, 8, 16 }                      # LoRA rank
α        ∈  { 16, 32, 64 }                     # LoRA alpha
p_drop   ~  Uniform(0.0, 0.2)                  # Dropout
η_lr     ~  LogUniform(1e-5, 2e-4)             # Learning rate
B        ∈  { 1, 2, 4 }                        # Batch size
```

### 🔗 Training Loop Integration

```python
# Non-invasive — just two lines added to the training loop
if trial is not None:
    trial.report(train_loss, global_step)       # Report intermediate metric
    if trial.should_prune():                    # Check against ASHA rungs
        raise optuna.TrialPruned()              # Early-stop this trial
```

### 📊 Top-5 Trial Results

<div align="center">

| Trial | `r` | `α` | `η_lr` | `B` | Val Loss | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **7** ⭐ | **16** | **32** | **1.34×10⁻⁴** | **4** | **0.1823** | ✅ Completed |
| 3 | 16 | 64 | 9.8×10⁻⁵ | 2 | 0.1941 | ✅ Completed |
| 9 | 8 | 32 | 1.12×10⁻⁴ | 4 | 0.2017 | ✅ Completed |
| 1 | 4 | 16 | 5.6×10⁻⁵ | 2 | 0.2209 | ✅ Completed |
| 5 | 4 | 32 | 2.3×10⁻⁵ | 1 | 0.2451 | ❌ Pruned (rung 1) |

</div>

> ⭐ **Trial 7** becomes the seed for the full 10-epoch fine-tune.  
> ASHA pruned **6 of 10 trials** at the first rung — completing the search in **≈ 2.8 hours** vs. **≈ 13.5 hours** for exhaustive grid search: a **4.8× wall-clock speedup**.

<br/>

---

## ✂️ Post-Training Pruning

DreamForge applies **ℓ₁-norm magnitude pruning** exclusively to LoRA adapter matrices A and B. The 860 M-parameter frozen backbone is **never touched**.

### Implementation

```python
import torch.nn.utils.prune as prune

def prune_lora_weights(unet, sparsity: float = 0.20):
    """Apply global L1-norm magnitude pruning to all LoRA adapter layers."""
    lora_layers = [
        (m, 'weight') for n, m in unet.named_modules()
        if 'lora' in n.lower() and hasattr(m, 'weight')
    ]

    prune.global_unstructured(
        lora_layers,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    # Make masks permanent before saving checkpoint
    for module, param in lora_layers:
        prune.remove(module, param)

    return unet
```

### 📉 Sparsity–Quality Trade-off

<div align="center">

| Sparsity | Val Loss | Adapter Size Reduction | Recommendation |
|:---:|:---:|:---:|:---:|
| 0% | 0.1823 | — | Baseline |
| 10% | 0.1829 | −9% | Conservative |
| **20%** | **0.1847** | **−18%** | ✅ **Recommended** |
| 30% | 0.1891 | −26% | Acceptable |
| 50% | 0.2103 | −40% | ⚠️ Degraded |
| 70% | 0.2614 | −57% | ❌ Not recommended |

</div>

> **20% sparsity is the sweet spot** — 18% memory reduction (26 MB → ~21 MB adapter) with only a **1.3% relative increase** in validation loss. Pruning beyond 30% begins to noticeably degrade qualitative image quality.

<br/>

---

## 📊 MLOps Stack

### 🔥 Prometheus Metrics `→ port 8012`

<div align="center">

| Metric | Type | Description |
|:---|:---:|:---|
| `sd_generations_total` | `Counter` | Cumulative count · labelled by `status` + `scheduler` |
| `sd_generation_duration_seconds` | `Histogram` | Latency distribution · 10 buckets (5–300 s) |
| `sd_steps_per_second` | `Gauge` | Current inference throughput |
| `sd_active_generations` | `Gauge` | Concurrent in-flight requests |
| `sd_gpu_memory_used_bytes` | `Gauge` | CUDA allocated memory |

</div>

**Alert Rules:**

| Alert | Condition | Severity |
|:---|:---|:---:|
| `DreamForgeAppDown` | Exporter unreachable > 1 min | 🔴 Critical |
| `HighGenerationErrorRate` | Error rate > 10% over 5 min | 🟡 Warning |
| `HighGenerationLatency` | Avg latency > 60 s over 5 min | 🟡 Warning |

### 📋 MLflow Experiment Tracking `→ port 5012`

Every generation and training run is logged as an MLflow run under named experiments:

```
Parameters  ──  prompt length · steps · guidance scale · scheduler · seed · LoRA mode
Metrics     ──  latency · steps/s · pixel count · validation loss
Artifacts   ──  generated images · prompt text files
```

### 📡 Statistical Drift Detection

```
Δ_lat = ( L̃_recent − L̃_baseline ) / L̃_baseline × 100%

  Δ_lat ≥ 25%  ──▶  ⚠️  Warning alert
  Δ_lat ≥ 75%  ──▶  🚨  Critical alert

  Δ_err = (e_recent − e_baseline) × 100  ──▶  Error rate drift alert
```

Checks run in **< 2 ms** on a 100-sample sliding window — zero impact on generation latency.

### 🗂️ Model Registry

File-backed version tracking with four stage labels serialised to `registry.json`:

```
 experimental  ──▶  staging  ──▶  production  ──▶  archived
      │                │               │                │
  Initial run     Validated       Live serving       Retired
   logged        candidate          model            versions
```

Each entry stores: HuggingFace model ID · MLflow run ID · evaluation metrics · provenance tags.

<br/>

---

## 🔬 Experimental Results

<div align="center">

**Hardware:** NVIDIA RTX 3090 (24 GB VRAM) · Intel Core i9-12900K · 64 GB DDR5 · Ubuntu 22.04  
**Software:** PyTorch 2.4.0 · CUDA 12.1 · HuggingFace Diffusers 0.30.3 · PEFT 0.12.0

</div>

### Fine-Tuning Results

<div align="center">

| Model | Val Loss | Test Loss | Trainable Params | Train Time |
|:---|:---:|:---:|:---:|:---:|
| SD v1.5 (zero-shot) | 0.2501 | 0.2548 | 0 | — |
| LoRA r = 4 | 0.2209 | 0.2241 | 0.9 M | 1.2 h |
| LoRA r = 8 | 0.2017 | 0.2039 | 1.7 M | 1.6 h |
| **LoRA r = 16 (ours)** ⭐ | **0.1823** | **0.1856** | **3.4 M** | **2.3 h** |

</div>

> **27.1% validation loss reduction** over the zero-shot baseline using only **0.39%** of UNet parameters.

### Scheduler Benchmark (512×512, 20 steps)

<div align="center">

| Scheduler | Latency (s) | Steps/s | Best For |
|:---|:---:|:---:|:---|
| **Euler A** ⚡ | **4.6** | **4.3** | Creative variety — fastest |
| PNDM | 4.7 | 4.3 | Stable baseline |
| DDIM | 4.8 | 4.2 | General use · deterministic |
| DPM++ 2M | 5.1 | 3.9 | Highest quality per step |
| LMS | 5.3 | 3.8 | Smooth results |
| DPM++ SDE | 6.2 | 3.2 | Stochastic quality |

</div>

### MLOps Overhead

<div align="center">

| Component | Overhead | Notes |
|:---|:---:|:---|
| MLflow logging | ~0.3 s/generation | 6% overhead at 5 s baseline |
| Prometheus update | < 1 ms | Truly negligible |
| Drift detection | < 2 ms | 100-sample sliding window |

</div>

<br/>

---

## 📁 Project Structure

```
dreamforge/
│
├── 📂 app/
│   ├── main.py                  ← Streamlit orchestration entry point
│   ├── sidebar.py               ← Generation parameter controls
│   └── metrics_dashboard.py     ← Plotly real-time monitoring dashboard
│
├── 📂 pipeline/
│   ├── model_manager.py         ← Singleton loader + scheduler hot-swap
│   ├── inference_engine.py      ← GenerationConfig → GenerationResult
│   ├── prompt_processor.py      ← Validation + prompt enhancement
│   ├── image_processor.py       ← Metadata, disk I/O, grid generation
│   └── batch_processor.py       ← PriorityQueue + async worker thread
│
├── 📂 monitoring/
│   ├── metrics_collector.py     ← Prometheus Counter/Histogram/Gauge
│   ├── mlflow_tracker.py        ← Run logging + artifact management
│   └── drift_detector.py        ← Sliding-window statistical alerts
│
├── 📂 training/
│   ├── train.py                 ← LoRATrainer (PEFT · AMP · cosine LR)
│   ├── hpo_objective.py         ← Optuna + ASHA SuccessiveHalvingPruner
│   ├── prune.py                 ← Magnitude-based L1 unstructured pruning
│   ├── data.py                  ← Full dataset preparation pipeline
│   └── prepare_mini_data.py     ← 100 image/class mini-dataset sampler
│
├── 📂 configs/
│   ├── best_trial.yaml          ← Best ASHA trial hyperparameters
│   └── prometheus_alerts.yaml   ← Alert rule definitions
│
├── registry.json                ← Human-readable model registry
├── docker-compose.yml           ← Full production stack deployment
├── Dockerfile
└── requirements.txt
```

<br/>

---

## ⚙️ Configuration Reference

```yaml
# configs/best_trial.yaml

lora:
  rank: 16                        # r — adapter capacity
  alpha: 32                       # α — scale = α/r = 2.0
  dropout: 0.05
  target_modules:
    - to_q
    - to_k
    - to_v
    - to_out.0
    - ff.net.0.proj
    - ff.net.2

training:
  learning_rate: 1.34e-4          # AdamW optimizer
  warmup_ratio: 0.05              # 5% of total steps
  batch_size: 4                   # per-GPU
  gradient_accumulation_steps: 2  # effective batch = 8
  mixed_precision: fp16           # torch.cuda.amp
  noise_offset: 0.05              # dark-scene fidelity
  epochs: 10

pruning:
  sparsity: 0.20                  # recommended operating point
  method: l1_unstructured
  target: lora_only               # backbone never touched

monitoring:
  mlflow_port: 5012
  prometheus_port: 8012
  drift_window_size: 100
  drift_latency_warning_pct: 25
  drift_latency_critical_pct: 75
```

<br/>

---

## 🗺️ Roadmap

- [ ] 🎭 **Extend festival dataset** — add Navratri, Durga Puja, Onam, Pongal
- [ ] 🖼️ **SDXL base model** — evaluate Stable Diffusion XL for higher-resolution generation
- [ ] 🗄️ **MLflow Model Registry backend** — replace `registry.json` with proper MLflow integration
- [ ] 🔍 **Automated augmentation filtering** — flag images where horizontal flip breaks cultural semantics
- [ ] 🗜️ **INT8 / INT4 quantisation** — post-training quantisation for edge deployment
- [ ] 🎛️ **ControlNet integration** — structural conditioning for layout-aware generation
- [ ] 🌐 **FastAPI REST backend** — headless inference API for programmatic access
- [ ] 📱 **ONNX export** — cross-platform inference without PyTorch dependency

<br/>

---

## 📚 Citation

If you use DreamForge in your research, please cite:

```bibtex
@article{singh2024dreamforge,
  title        = {DreamForge: An End-to-End Text-to-Image Generation Studio},
  author       = {Singh, Aditya Pratap and Shankesh, Gadiya Mahek},
  institution  = {Indian Institute of Technology Jodhpur},
  program      = {M.Tech Artificial Intelligence},
  year         = {2024}
}
```

**Core dependencies — please also cite:**

```bibtex
@inproceedings{rombach2022ldm,
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik
               and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle = {Proceedings of the IEEE/CVF CVPR},
  pages     = {10684--10695},
  year      = {2022}
}

@inproceedings{hu2022lora,
  title     = {{LoRA}: Low-Rank Adaptation of Large Language Models},
  author    = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip
               and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean
               and Wang, Lu and Chen, Weizhu},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}

@inproceedings{li2020asha,
  title     = {A System for Massively Parallel Hyperparameter Tuning},
  author    = {Li, Liam and Jamieson, Kevin and Rostamizadeh, Afshin
               and Gonina, Ekaterina and Ben-tzur, Jonathan and Hardt, Moritz
               and Recht, Benjamin and Talwalkar, Ameet},
  booktitle = {Proceedings of Machine Learning and Systems},
  year      = {2020}
}

@inproceedings{akiba2019optuna,
  title     = {Optuna: A Next-generation Hyperparameter Optimization Framework},
  author    = {Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko
               and Ohta, Takeru and Koyama, Masanori},
  booktitle = {Proceedings of the 25th ACM SIGKDD},
  pages     = {2623--2631},
  year      = {2019}
}
```

<br/>

---

## 🙏 Acknowledgements

We thank the **AIMLOps-C4-G16 team** for releasing the IndianFestivals dataset on Hugging Face, the **HuggingFace Diffusers** and **PEFT** teams for their outstanding open-source libraries, and the **Optuna** and **Prometheus** communities for their exceptional tooling.

<br/>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,20,25&height=120&section=footer&animation=fadeIn" width="100%"/>

**DreamForge** · Built with ❤️ at IIT Jodhpur · M.Tech Artificial Intelligence

*Aditya Pratap Singh · Gadiya Mahek Shankesh*

<br/>

*"From festival prompts to photorealistic pixels — DreamForge bridges the gap."*

</div>
