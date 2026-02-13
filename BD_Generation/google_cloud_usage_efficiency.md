# Google Cloud Credits & Compute Planning Guide

This document explains **how to plan and efficiently use Google Cloud credits** for the Bubble Diagram (BD) diffusion project

---

## 1. What Google Cloud Credits Actually Pay For

Google Cloud credits act like a balance that offsets **billable usage** on eligible services. In this project, credits will mainly be consumed by:

### 1.1 Compute
- **GPU time (largest cost)** – training the diffusion model
- **CPU + RAM** – data preprocessing, debugging, lightweight experiments

### 1.2 Storage
- **Persistent Disks** attached to VMs (billed even when VM is stopped)
- **Cloud Storage buckets** for datasets, checkpoints, generated samples

### 1.3 Networking
- Ingress (uploading data) is usually free
- **Egress (downloading large files or cross-region traffic)** can cost credits

Credits do *not* disappear automatically when a VM crashes or becomes idle — **you pay while resources exist**, not while code runs.

---

## 2. How to Inspect GitHub Repos for Compute Requirements

Before running anything on the cloud, every repo should be audited with the same checklist.

### 2.1 Quick Repo Audit Checklist

When opening a repo:

1. **Read the README for hardware notes**
   - Look for phrases like `A100`, `24GB GPU`, `multi-GPU`, `tested on`

2. **Identify model size controls**
   - Transformer depth
   - Hidden dimension
   - Sequence length
   - Batch size

3. **Find the training entry point**
   - `main.py`, `train.py`, Hydra configs, shell scripts

4. **Check dataset size and format**
   - Number of samples
   - Fixed vs variable size

5. **Look for non-Python dependencies**
   - C++ builds
   - RDKit / graph-tool / CUDA-specific installs

If a repo requires compilation or large dependencies, **never set it up first on a GPU VM**.

---

## 3. Repos Used in This Project and Their Cost Implications

### 3.1 MDLM (Core Framework)

**Role:** Main diffusion framework and transformer backbone.

**What matters for cost:**
- Designed for long text sequences (e.g. 1024 tokens)
- Your project uses **SEQ_LEN tokens total** (36 for RPLAN with N_MAX=8; up to 105 for ResPlan with N_MAX=14)

**Implication:**
- Memory and compute requirements are *much lower* than typical MDLM use
- A **single mid-range GPU** is sufficient

**Cost strategy:**
- Debug and modify MDLM on CPU
- Move to GPU only once data + training loop are stable

---

### 3.2 Graph2Plan (Primary Dataset)

**Role:** Main data source (~80K bubble diagrams).

**Compute profile:**
- Data download + `.mat` loading
- Tokenization into fixed-length sequences

**Implication:**
- Entirely **CPU-bound**
- No GPU required

**Efficiency rule:**
- Convert once
- Save as `.pt` / `.npz`
- Never re-parse `.mat` files inside training loops

---

### 3.3 ResPlan (Supplementary Dataset, will use it later in the project, not now)

**Role:** Optional dataset expansion (~17K plans).

**Compute profile:**
- Geometry processing (Shapely, GeoPandas)
- Graph traversal and centroid math

**Implication:**
- CPU-heavy but still cheaper than GPU training

**Efficiency rule:**
- Run conversion once
- Cache converted graphs
- Do not recompute geometry per epoch

---

### 3.4 DiGress (Reference Only, we don't need to use anything about its repo now)

**Role:** Graph diffusion reference and evaluation ideas.

**Cost warning:**
- Heavy dependencies (RDKit, graph-tool, compiled C++)
- Environment setup can waste paid compute time

**Recommendation:**
- Read code
- Reimplement needed metrics in lightweight Python

---

### 3.5 MaskPLAN (Reference Only, we don't need to use anything about its repo now)

**Role:** Masked training ideas.

**Cost warning:**
- Explicitly requires **≥24GB GPU** and large RAM

**Recommendation:**
- Do not run
- Use only for conceptual inspiration

---

## 4. CPU vs GPU Usage Plan

### Phase 1: CPU Only (No Credits Burn)
- Download datasets
- Convert Graph2Plan / ResPlan to unified format
- Implement tokenization
- Build dataset class
- Run tiny forward passes

### Phase 2: GPU Debug Runs (Short Sessions)
- Single GPU
- Small batch size
- Few thousand steps
- Verify loss decreases and samples make sense

### Phase 3: Full Training (Budgeted)
- Single GPU
- Checkpoint frequently
- Optionally use Spot / Preemptible VMs

---

## 5. How Credits Are Commonly Wasted (Avoid These)

- Leaving a GPU VM running overnight
- Installing dependencies on a GPU VM
- Parsing raw datasets inside training loops
- Re-running dataset conversions
- Keeping unused persistent disks

## 6. Budget Control Best Practices

- Create a **Billing Budget** equal to your total credits
- Enable alerts at 25%, 50%, 75%, 90%
- Regularly inspect Billing → Reports
- Toggle "include promotional credits" to see real burn rate

---

## 7. Minimal-Cost Strategy Summary

- CPU for everything until the model trains end-to-end
- One GPU, not many
- Cache all data
- Short GPU sessions early
- Long runs only when confident

This approach keeps the project **well within limited Google Cloud credits** while still allowing full experimentation.
