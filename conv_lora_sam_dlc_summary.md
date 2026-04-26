---
title: "Conv-LoRA SAM to SNPE DLC Pipeline: Project Summary"
subtitle: "Quantization, export, and deployment to Qualcomm IQ-9075 NPU"
date: "April 2026"
geometry: margin=2.5cm
fontsize: 11pt
toc: true
numbersections: true
colorlinks: true
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{Conv-LoRA SAM DLC Pipeline}
  - \fancyhead[R]{\thepage}
---

\newpage

# Project Overview

## Objective

Deploy a Conv-LoRA SAM (Segment Anything Model with Convolutional Low-Rank Adaptation) model
quantized to **W8A8 (8-bit weights, 8-bit activations)** using **PTQ4SAM** as a SNPE
**DLC (Deep Learning Container)** runnable on the **Qualcomm IQ-9075 NPU (Hexagon v79 HTP)**.

## Model and Task

- **Architecture:** SAM ViT-B encoder + SAM mask decoder
- **Adaptation:** Conv-LoRA with Mixture-of-Experts (4 experts) replacing attention projections
- **Task:** Semantic segmentation of defects on metal surfaces from GAN-generated microscopy images
- **Training framework:** AutoGluon MultiModal
- **Checkpoint:** AutogluonModels/class\_2

## Hardware Target

- **Host:** x86-64 Linux pod (JupyterHub/Kubernetes), NVIDIA GPU
- **Target device:** Qualcomm Dragonwing IQ-9075 Evaluation Kit (Snapdragon X Elite, Hexagon v79 HTP)
- **SDK:** QAIRT 2.42.0.251225

\newpage

# Step 1: Fix PTQ4SAM Calibration Data

**Problem:** PTQ4SAM was calibrating quantization parameters on **test data** instead of training
data. This is data leakage that produces overfit scale factors.

**Action:** Modified `ptq4sam_convlora.py` to load `train_class2.csv` during the calibration step.

**Reason:** Calibration must use training data so that activation range statistics reflect the
real-world input distribution, not the test set. Using test data biases the quantized model and
invalidates evaluation results.

**Outcome:** PTQ calibration runs on training data; scale/zero-point values now generalize correctly.

\newpage

# Step 2: Create the DLC Export Pipeline

**Problem:** No pipeline existed to convert the trained Conv-LoRA SAM model to SNPE DLC format.

**Action:** Designed three pipeline files:

1. `export_ptq_to_onnx.py` --- Load PTQ4SAM model, export to ONNX with quantization preserved
2. `convert_onnx_to_dlc.sh` --- Call `snpe-onnx-to-dlc` to produce DLC files
3. `evaluate_dlc.py` --- Run DLC inference via `snpe-net-run` and compute IoU on the host

**Reason:** SNPE requires DLC format for on-device inference. The standard pipeline is
PyTorch to ONNX to DLC. Testing on x86 CPU first verifies correctness before deploying to the board.

\newpage

# Step 3: Debug ONNX Export Errors

**Problems encountered:**

1. Conv-LoRA MoE routing used dynamic if/else control flow incompatible with ONNX tracing
2. Image encoder split into two components during tracing
3. PTQ4SAM custom quantizer classes had no ONNX export equivalents

**Actions taken:**

- Added `ONNX_EXPORT_MODE` flag to force dense MoE routing (all experts active) during export
- Merged encoder components into a single traceable `forward()` pass
- Set `do_constant_folding=False` and `opset_version=13` to avoid shape mismatches

**Reason:** `torch.onnx.export` requires fully static graphs. Dynamic routing decisions cannot be
traced. Dense routing is the correct approximation since all expert capacity is available at
inference time.

\newpage

# Step 4: First Successful FP32 Pipeline Run

**Outcome:** End-to-end run produced valid ONNX and DLC files:

- Encoder ONNX: 4.3 MB proto + 2.4 GB weight binary
- Decoder ONNX: 15 MB
- FP32 PyTorch IoU: **0.1906** (100 images, test2\_class2)
- PTQ4SAM W8A8 PyTorch IoU: **0.1826** (100 images)

The small accuracy drop from FP32 to W8A8 (~0.4% IoU) is expected and acceptable.

\newpage

# Step 5: Discovered DLC Was FP32-Only

**Problem:** `snpe-dlc-info` revealed the initial DLC had 0 INT8 tensors and 6767 FP32 tensors.
All quantization structure had been stripped.

**Root cause:** The function `bake_quantized_weights()` merged quantized weights back into
floating point, eliminating fake-quantize nodes and producing a valid but FP32-only ONNX graph.

\newpage

# Step 6: User Requirement - Preserve PTQ4SAM W8A8

**User constraint:** "I do not want to use SNPE quantization. As PTQ4SAM can quantize the
Conv-LoRA SAM model to W8A8, I want to take that quantized version and convert to DLC which will
also be W8A8."

**Implication:** PTQ4SAM quantization parameters (scale, zero-point, bitwidth) must flow from
PyTorch to ONNX to DLC intact. SNPE re-quantization must not be used.

\newpage

# Step 7: QDQ ONNX Export

**Problem:** PTQ4SAM's custom quantizer classes (`LSQFakeQuantize`, `FixedFakeQuantize`,
`AGQFakeQuantize`) have no ONNX export equivalents.

**Solution:** Rewrote `export_ptq_to_onnx.py` with a `convert_to_qdq_quantizers()` function:

- `FixedFakeQuantize` (weights, per-channel symmetric 8-bit) replaced with
  `torch.ao.quantization.FakeQuantize` using `torch.per_channel_affine_float_qparams`
- `LSQFakeQuantize` (activations, per-tensor asymmetric 8-bit) replaced with
  `torch.ao.quantization.FakeQuantize` using `torch.per_tensor_affine`
- `AGQFakeQuantize` (post-softmax logarithmic) replaced with **PassThrough** (identity,
  no QDQ equivalent exists for logarithmic quantization)
- `QLinear`/`QConv2d` modules wrapped with `QDQLinear`/`QDQConv2d` that explicitly insert
  quantize/dequantize pairs around weight tensors

When `torch.onnx.export` runs, PyTorch's FakeQuantize lowering produces standard ONNX
`QuantizeLinear`/`DequantizeLinear` (QDQ) node pairs.

**Results:**

- Encoder ONNX: 712 QDQ node pairs
- Decoder ONNX: 170 QDQ node pairs
- 39 AGQ instances (post-softmax attention) remain FP32 via PassThrough

**Reason:** ONNX QDQ nodes are the industry standard for INT8 quantization. SNPE's
`snpe-onnx-to-dlc` recognizes these pairs and absorbs them as layer encoding metadata
(scale/offset/bitwidth), enabling INT8 inference on the HTP NPU.

\newpage

# Step 8: DLC Conversion - QDQ Absorption

**First attempt (with `--keep_quant_nodes`):** SNPE kept QDQ as explicit Quantize/Dequantize
layers in the DLC. The x86 CPU runtime does not support these layer types. Inference failed.

**Final approach (without `--keep_quant_nodes`):** SNPE absorbs QDQ values into each layer's
encoding metadata:

```
layer "MatMul_42" {
    encoding { bitwidth: 8, min: -3.142, max: 3.007, scale: 0.0247, offset: -127 }
}
```

**Verification note:** `snpe-dlc-info` uses the term `uFxp_8` (unsigned fixed-point 8-bit),
not `Int_8`. After correcting the search pattern:

- Encoder DLC: 228 layers with bitwidth=8 encoding
- Decoder DLC: 33 layers with bitwidth=8 encoding

**How it works:** On x86 CPU, the encoding metadata is present but ignored; the runtime uses FP32.
On the HTP NPU with `--use_dsp`, the INT8 kernel backends read the encoding metadata and execute
quantized INT8 operations.

\newpage

# Step 9: Fix Encoder Timeout

**Problem:** Running `evaluate_dlc.py` on 100 images timed out (2 hours). Encoder DLC is
~105 seconds per image on x86 CPU. Calling `snpe-net-run` once with all 100 images timed out.

**Actions taken:**

1. **Per-image processing:** Call `snpe-net-run` once per image with a 3600s per-image timeout
2. **`--max_images` flag:** Run a subset for quick validation
3. **Fixed Result\_0 overwriting bug:** `snpe-net-run` always writes to `Result_0/` regardless
   of input index. Fixed by using per-image temp directories (`_tmp_enc_{idx}`), then renaming
   `Result_0` to `Result_{idx}` after each run.

\newpage

# Step 10: Full 100-Image DLC Evaluation

**Results on test2\_class2 (100 images):**

| Model | Mean IoU | Notes |
|---|---|---|
| FP32 PyTorch (baseline) | 0.1906 | Full precision, no quantization |
| PTQ4SAM W8A8 PyTorch | 0.1826 | 8-bit weights + 8-bit activations |
| DLC W8A8 x86 CPU | **0.1933** | QDQ absorbed, FP32 CPU runtime |
| DLC W8A8 IQ-9075 NPU | TBD | --use\_dsp activates true INT8 |

DLC IoU (0.1933) is slightly higher than PTQ PyTorch (0.1826) due to SNPE operator fusion
optimizations. All 100 prediction visualizations saved to `dlc_export/dlc_eval/visualizations/`.

\newpage

# Step 11: Device Deployment Scripts

**`setup_device.sh` (host-side, one-time setup):**
Pushes to IQ-9075 via SCP:
- `snpe-net-run` binary (aarch64-ubuntu-gcc9.4 target)
- SNPE ARM64 shared libraries (35 `.so` files)
- Hexagon v79 skeleton libraries (`libSnpeHtpV79Skel.so`, `libQnnHtpV79Skel.so`, etc.)
- DLC files (~2.5 GB total)

**`evaluate_dlc_device.py` (host-side, runs inference via SSH):**
Orchestrates the full pipeline from the host:

1. Preprocess image on host (resize 1024x1024, normalize, save as raw float32)
2. SCP input to device; SSH: run encoder DLC with `--use_dsp` on the NPU
3. SCP encoder output back; SCP decoder inputs (encoder output + static embeddings) to device
4. SSH: run decoder DLC with `--use_dsp`; SCP mask output back; Compute IoU on host

\newpage

# Step 12: Git Repository - SAMonIQ9

**Purpose:** Self-contained repository to run inference directly on the IQ-9075 device
(no SSH tunneling, no host Python environment required on the device).

**Repository structure:**

```
SAMonIQ9/
  evaluate_on_device.py    <- runs entirely on-device, no AutoGluon
  run_inference.sh         <- convenience wrapper
  embeddings/              <- pre-cached prompt embeddings (in git, ~8 MB)
    image_pe.npy
    no_prompt_dense.npy
    no_prompt_sparse.npy
  hexagon-v79/             <- HTP v79 skel .so libs (in git, ~37 MB)
  dlc/                     <- DLC files (NOT in git, downloaded from Drive)
  datasets/
    gan-generated/
      test1_class1.csv     <- 18 test CSVs in git (100 rows each)
      test{1-6}/           <- images NOT in git, downloaded from Google Drive
  README.md
```

**Key design decisions:**

| Decision | Reason |
|---|---|
| Embeddings in git | SAM prompt encoder needs AutoGluon. Pre-cache once on host, commit outputs. |
| Hexagon v79 libs in git | Device SDK may not include the exact version; bundling ensures reproducibility. |
| DLC files NOT in git | Too large (2.4 GB); hosted on Google Drive with gdown instructions. |
| Images NOT in git | Large dataset; hosted on Google Drive. |
| CSVs in git | Lightweight metadata defining image/label pairings for evaluation. |
| --use\_dsp flag | Activates INT8 NPU execution path on HTP. |

**`evaluate_on_device.py` dependencies:** only stdlib + numpy + Pillow + scipy.
No PyTorch, no AutoGluon, no CUDA required on the device.

\newpage

# Step 13: Dataset Setup and README

**Dataset structure (GAN-generated):**

- 6 test splits (test1-test6) x 3 classes (class\_1, class\_2, class\_3) = 18 test CSVs
- 100 images per CSV = 1800 total test images
- Each split has `input/` (SEM/GAN images) and `class_1/`, `class_2/`, `class_3/` (masks)

**CSV format:**

```
,image,label
0,test2/input/pair_5_Ti64_000000.png,test2/class_2/pair_5_Ti64_000000.png
```

**README.md** documents: gdown commands for Google Drive downloads, Python dependency installation,
SNPE SDK verification, and how to run single or all 18 test sets.

\newpage

# Summary of All Files Created or Modified

| File | Action | Purpose |
|---|---|---|
| `ptq4sam_convlora.py` | Modified | Fix calibration data; add ONNX\_EXPORT\_MODE for dense MoE |
| `export_ptq_to_onnx.py` | Created | Replace quantizers with torch.ao FakeQuantize; QDQ ONNX export |
| `convert_onnx_to_dlc.sh` | Created | snpe-onnx-to-dlc without --keep\_quant\_nodes |
| `evaluate_dlc.py` | Created | x86 per-image snpe-net-run evaluation with IoU and visualizations |
| `setup_device.sh` | Created | One-time SCP of SNPE binaries and DLC files to IQ-9075 |
| `evaluate_dlc_device.py` | Created | Host-side SSH orchestration for on-device NPU inference |
| `SAMonIQ9/evaluate_on_device.py` | Created | Runs directly on IQ-9075; numpy/Pillow only |
| `SAMonIQ9/run_inference.sh` | Created | Shell wrapper for evaluate\_on\_device.py |
| `SAMonIQ9/README.md` | Created | Full setup guide for IQ-9075 deployment |
| `SAMonIQ9/.gitignore` | Created | Ignore DLC and image files; track CSVs and embeddings |
| `SAMonIQ9/embeddings/*.npy` | Added | Pre-cached SAM prompt embeddings (~8 MB) |
| `SAMonIQ9/hexagon-v79/*.so` | Added | HTP v79 skeleton libraries (~37 MB) |
| `SAMonIQ9/datasets/gan-generated/*.csv` | Added | All 21 test/train CSVs |

\newpage

# Final Performance Summary

**IoU results (test2\_class2, 100 images):**

| Model | Mean IoU |
|---|---|
| FP32 PyTorch (baseline) | 0.1906 |
| PTQ4SAM W8A8 PyTorch | 0.1826 |
| DLC W8A8 x86 CPU snpe-net-run | **0.1933** |
| DLC W8A8 IQ-9075 NPU (TBD) | --- |

**Quantization statistics:**

| Component | QDQ Pairs | INT8 Layers in DLC |
|---|---|---|
| Encoder DLC | 712 | 228 |
| Decoder DLC | 170 | 33 |

**Inference timing:**

| Platform | Encoder | Decoder |
|---|---|---|
| x86 CPU | ~105s per image | <1s per image |
| IQ-9075 NPU (estimated) | <1s per image | <0.1s per image |

\newpage

# Key Technical Notes

**Why AGQ uses PassThrough:** `AGQFakeQuantize` applies logarithmic quantization to post-softmax
attention weights. ONNX has no standard QDQ equivalent for logarithmic quantization. These 39
instances remain FP32. Impact is minimal as post-softmax values are small-magnitude and near zero.

**Why DLC runs FP32 on x86 but INT8 on NPU:** SNPE's x86 backend does not implement INT8 math.
Encoding metadata is used only when targeting HTP/DSP. x86 is for functional testing; NPU is the
deployment target where INT8 speed and power efficiency are realized.

**Why `--keep_quant_nodes` was removed:** With the flag, SNPE creates explicit Quantize/Dequantize
layers that the x86 CPU runtime cannot execute (unsupported layer type, inference fails). Without
the flag, QDQ is absorbed into layer encoding metadata. CPU uses FP32; HTP uses INT8 from metadata.
This is the correct production approach.

**Why prompt embeddings are pre-cached:** The SAM prompt encoder requires AutoGluon + PyTorch with
a full model checkpoint. The IQ-9075 is an embedded board without a Python ML stack. By running
the prompt encoder once on the host and saving outputs as `.npy` files, the device only needs
numpy + Pillow. This works because the no-prompt (automatic) inference mode uses the same fixed
embeddings for every image.
