# Conv-LoRA SAM — IQ-9075 NPU Evaluation

Evaluate the quantized (PTQ4SAM W8A8) Conv-LoRA SAM segmentation model on the Qualcomm IQ-9075 Development Kit using the NPU (HTP/Hexagon v79) via the SNPE/QAIRT SDK.

---

## Repository Structure

```
SAMonIQ9/
├── evaluate_on_device.py       ← inference + IoU evaluation script
├── run_inference.sh            ← convenience wrapper shell script
├── embeddings/                 ← pre-cached SAM prompt embeddings (in git)
│   ├── image_pe.npy
│   ├── no_prompt_dense.npy
│   └── no_prompt_sparse.npy
├── hexagon-v79/                ← Hexagon v79 HTP skel .so libs (in git)
├── dlc/                        ← DLC model files (NOT in git — download below)
│   ├── convlora_sam_encoder_ptq.dlc   (~2.4 GB)
│   └── convlora_sam_decoder_ptq.dlc   (~15 MB)
└── datasets/
    └── gan-generated/          ← dataset CSVs are in git; images must be downloaded
        ├── test1_class1.csv    ← 100 image/label pairs per CSV
        ├── test1_class2.csv
        ├── ...                 (18 CSVs total: test1–6 × class1–3)
        ├── test1/              ← image folders (NOT in git — download below)
        │   ├── input/          ← test images (.png)
        │   ├── class_1/        ← ground truth masks (.png)
        │   ├── class_2/
        │   └── class_3/
        ├── test2/ ... test6/
        └── note.txt
```

---

## Step 1 — Clone the repo

```bash
git clone https://github.com/MushfiqShovon/Conv-LoRA-SAM-on-IQ-9.git
cd SAMonIQ9
```

---

## Step 2 — Download DLC model files

> **Google Drive:** https://drive.google.com/drive/folders/1t2wSW3MHnCoLxJSAzbyU9u71Jh__rOOk?usp=sharing

Download both files and place them in the `dlc/` folder:

```
dlc/convlora_sam_encoder_ptq.dlc   (~2.4 GB)
dlc/convlora_sam_decoder_ptq.dlc   (~15 MB)
```

Using `gdown` (install with `pip3 install gdown`):

```bash
pip3 install gdown
# Download the entire DLC folder to dlc/
gdown --folder "https://drive.google.com/drive/folders/1t2wSW3MHnCoLxJSAzbyU9u71Jh__rOOk?usp=sharing" -O dlc/ --remaining-ok
```

---

## Step 3 — Download the dataset

> **Google Drive:** https://drive.google.com/drive/folders/1GQvW7WDw3KkeCMZYyo2Vkx3HUH8ZynOT?usp=sharing

Download the dataset and extract it so the image folders are inside `datasets/gan-generated/`:

```bash
pip3 install gdown
# Download the entire dataset folder
gdown --folder "https://drive.google.com/drive/folders/1GQvW7WDw3KkeCMZYyo2Vkx3HUH8ZynOT?usp=sharing" \
      -O datasets/gan-generated/ --remaining-ok
```

After download, verify the structure looks like:

```
datasets/gan-generated/
├── test1/input/*.png
├── test1/class_1/*.png
├── test1/class_2/*.png
├── test1/class_3/*.png
├── test2/ ... test6/  (same sub-structure)
```

---

## Step 4 — Install Python dependencies

```bash
pip3 install numpy pillow scipy
# Optional (for precise circular ROI masking):
pip3 install opencv-python
```

---

## Step 5 — Verify SNPE SDK

The device must have SNPE/QAIRT SDK installed with `snpe-net-run` accessible.

```bash
# Check snpe-net-run is in PATH:
snpe-net-run --version

# Or if SDK is installed at a custom path, set SNPE_ROOT:
export SNPE_ROOT=/path/to/qairt/2.42.0.251225
```

---

## Step 6 — Run inference

### Single CSV (recommended first test)

```bash
# Quick test — 5 images on NPU
./run_inference.sh test2_class2 --max_images 5

# Full 100-image evaluation on NPU
./run_inference.sh test2_class2

# CPU fallback (debugging only)
./run_inference.sh test2_class2 --use_cpu
```

### All 18 test sets

```bash
for csv in test{1..6}_class{1..3}; do
    ./run_inference.sh $csv
done
```

### With explicit SNPE SDK path

```bash
SNPE_ROOT=/path/to/qairt/2.42.0.251225 ./run_inference.sh test2_class2
```

---

## Results

Results are saved to `work/results/iou_<data_name>.txt` after each run.

```
Dataset: test2_class2
Runtime: NPU/HTP (--use_dsp)
Images evaluated: 100
Mean IoU: X.XXXX
Individual IoUs: [...]
```

### Reference IoU (x86 CPU, 100 images, test2_class2)

| Model | Mean IoU |
|---|---|
| FP32 PyTorch | 0.1906 |
| PTQ4SAM W8A8 PyTorch | 0.1826 |
| DLC W8A8 (x86 CPU, snpe-net-run) | 0.1933 |
| **DLC W8A8 (IQ-9075 NPU)** | TBD |

---

## Notes

- **Hexagon v79 skel libs** (`hexagon-v79/*.so`) are included in this repo and used automatically via `ADSP_LIBRARY_PATH`. These are the HTP kernel libraries for the IQ-9075's NPU.
- The `--use_dsp` flag activates INT8 execution on the NPU. Without it, inference falls back to FP32 on CPU.
- The encoder DLC is large (~2.4 GB). Ensure you have enough free storage on the device.
- `dlc/*.dlc` and all image files are excluded from git via `.gitignore`.
