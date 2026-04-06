#!/usr/bin/env python3
"""
Conv-LoRA SAM DLC evaluation — runs DIRECTLY on the IQ-9075 device.

No SSH, no autogluon. Uses snpe-net-run from the device's own SNPE SDK
to run inference on the NPU (HTP/DSP) with --use_dsp.

Pre-cached prompt embeddings (embeddings/*.npy) are loaded from this repo
so autogluon is NOT needed on the device.

Directory layout expected:
  SAMonIQ9/
  ├── dlc/
  │   ├── convlora_sam_encoder_ptq.dlc    ← download from Google Drive (see README.md)
  │   └── convlora_sam_decoder_ptq.dlc    ← download from Google Drive (see README.md)
  ├── embeddings/
  │   ├── image_pe.npy
  │   ├── no_prompt_dense.npy
  │   └── no_prompt_sparse.npy
  ├── datasets/
  │   └── gan-generated/
  │       ├── test1/, test2/, ... test6/   ← download images from Google Drive (see README.md)
  │       │   ├── input/     ← test images (.png)
  │       │   ├── class_1/   ← ground truth masks (.png)
  │       │   ├── class_2/
  │       │   └── class_3/
  │       ├── test1_class1.csv  ← already in git (image/label column pairs)
  │       └── ...               (18 CSVs total)
  └── work/           ← runtime I/O (auto-created)

Usage:
    python3 evaluate_on_device.py \
        --data_name test2_class2 \
        [--dataset_dir datasets/gan-generated] \
        [--snpe_root /path/to/snpe_sdk]   # optional, if snpe-net-run not in PATH
        [--use_cpu]                        # fallback to CPU (no --use_dsp)
        [--max_images 5]

Dependencies (standard on Linux ARM64):
    numpy, Pillow, scipy
    Optional: opencv-python (for precise circular ROI masking)

Install if missing:
    pip3 install numpy pillow scipy
"""

import os
import sys
import argparse
import subprocess
import shutil
import logging
import time

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("evaluate_on_device")

# Repository root = directory containing this script
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =====================================================================
# Environment helpers
# =====================================================================

def get_snpe_env(snpe_root=None):
    """Build env dict for snpe-net-run. snpe_root=None means use PATH."""
    env = os.environ.copy()
    hexagon_dir = os.path.join(REPO_ROOT, "hexagon-v79")
    if os.path.isdir(hexagon_dir):
        # Prefer repo's skel libs; fall back to system
        env["ADSP_LIBRARY_PATH"] = hexagon_dir
    if snpe_root:
        env["PATH"] = os.path.join(snpe_root, "bin") + ":" + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = (
            os.path.join(snpe_root, "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
        )
    return env


def find_snpe_net_run(snpe_root=None):
    """Return path to snpe-net-run binary."""
    if snpe_root:
        # Try common sub-paths in an installed SDK
        candidates = [
            os.path.join(snpe_root, "bin", "snpe-net-run"),
            os.path.join(snpe_root, "bin", "aarch64-ubuntu-gcc9.4", "snpe-net-run"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
    return "snpe-net-run"   # assume it is in PATH


# =====================================================================
# Preprocessing
# =====================================================================

def preprocess_image(image_path):
    """Resize to 1024x1024, normalise, return NHWC float32 array."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((1024, 1024), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr[np.newaxis].astype(np.float32)   # (1, H, W, C)


# =====================================================================
# Post-processing
# =====================================================================

def bilinear_resize(arr_2d, out_h, out_w):
    """Bilinear resize using PIL (no torch needed)."""
    h, w = arr_2d.shape
    # Normalise to [0,255] uint8 for PIL
    a_min, a_max = arr_2d.min(), arr_2d.max()
    if a_max > a_min:
        norm = ((arr_2d - a_min) / (a_max - a_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(arr_2d, dtype=np.uint8)
    img = Image.fromarray(norm).resize((out_w, out_h), Image.BILINEAR)
    result = np.asarray(img, dtype=np.float32)
    # Restore original range
    result = result / 255.0 * (a_max - a_min) + a_min
    return result


def postprocess_mask(raw_mask, orig_h, orig_w):
    """
    256x256 logit → 1024x1024 → orig_size → binary.
    Matches the two-step bilinear upsampling in AutoGluon's SAM pipeline.
    """
    expected = 256 * 256
    if raw_mask.size == expected:
        mask = raw_mask.reshape(256, 256)
    elif raw_mask.size == 1 * 1 * 1 * 256 * 256:
        mask = raw_mask.reshape(256, 256)
    else:
        side = int(np.sqrt(raw_mask.size))
        mask = raw_mask.reshape(side, side)

    mask_1024 = bilinear_resize(mask, 1024, 1024)
    mask_orig = bilinear_resize(mask_1024, orig_h, orig_w)
    return (mask_orig > 0.5).astype(np.uint8)


def get_valid_region(img_arr):
    """
    Detect circular metal region for IoU masking.
    Uses cv2 if available, falls back to full-image mask.
    """
    try:
        import cv2
        mask = img_arr > 10
        mask = ndi.binary_closing(mask, structure=np.ones((5, 5)))
        mask = ndi.binary_fill_holes(mask)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cnt = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        m = np.zeros_like(mask, dtype=np.uint8)
        cv2.circle(m, (int(x), int(y)), int(radius), 1, -1)
        return m.astype(bool)
    except Exception:
        # cv2 not available or detection failed — use full image
        return np.ones(img_arr.shape, dtype=bool)


def calculate_iou(pred, gt):
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return 0.0 if union == 0 else round(float(intersection) / float(union), 4)


# =====================================================================
# SNPE inference
# =====================================================================

def run_encoder(snpe_bin, dlc_path, raw_files, enc_out_dir, use_dsp, snpe_env, timeout=600):
    """Run encoder one image at a time."""
    os.makedirs(enc_out_dir, exist_ok=True)
    total = len(raw_files)
    logger.info(f"  Running encoder on {total} images ({'NPU/DSP' if use_dsp else 'CPU'}) ...")

    for idx, raw_path in enumerate(raw_files):
        input_txt = os.path.join(enc_out_dir, f"_in_{idx}.txt")
        tmp_out   = os.path.join(enc_out_dir, f"_tmp_{idx}")
        os.makedirs(tmp_out, exist_ok=True)

        with open(input_txt, "w") as f:
            f.write(f"pixel_values:={raw_path}\n")

        cmd = [snpe_bin, "--container", dlc_path,
               "--input_list", input_txt,
               "--output_dir", tmp_out,
               "--perf_profile", "balanced"]
        if use_dsp:
            cmd.append("--use_dsp")

        t0 = time.time()
        logger.info(f"  Encoder [{idx+1}/{total}] ...")
        result = subprocess.run(cmd, env=snpe_env,
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            logger.error(f"Encoder failed on image {idx}:\n{result.stderr}")
            raise RuntimeError(f"Encoder failed (image {idx}): {result.stderr[:400]}")
        logger.info(f"  Encoder [{idx+1}/{total}] done in {time.time()-t0:.1f}s")

        # Rename Result_0 → Result_<idx>
        src = os.path.join(tmp_out, "Result_0")
        dst = os.path.join(enc_out_dir, f"Result_{idx}")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        os.rename(src, dst)
        shutil.rmtree(tmp_out)
        os.remove(input_txt)

    logger.info("Encoder done.")


def run_decoder(snpe_bin, dlc_path, n_images, enc_out_dir, static_raws, dec_out_dir, use_dsp, snpe_env, timeout=7200):
    """Run decoder for all images in one snpe-net-run call."""
    os.makedirs(dec_out_dir, exist_ok=True)
    pe_raw, sparse_raw, dense_raw = static_raws

    input_txt = os.path.join(dec_out_dir, "_dec_input.txt")
    with open(input_txt, "w") as f:
        for idx in range(n_images):
            emb = os.path.join(enc_out_dir, f"Result_{idx}", "image_embeddings.raw")
            f.write(
                f"image_embeddings:={emb} "
                f"image_pe:={pe_raw} "
                f"sparse_prompt_embeddings:={sparse_raw} "
                f"dense_prompt_embeddings:={dense_raw}\n"
            )

    cmd = [snpe_bin, "--container", dlc_path,
           "--input_list", input_txt,
           "--output_dir", dec_out_dir,
           "--perf_profile", "balanced",
           "--set_unconsumed_as_output"]
    if use_dsp:
        cmd.append("--use_dsp")

    logger.info(f"  Running decoder ({n_images} images) ...")
    t0 = time.time()
    result = subprocess.run(cmd, env=snpe_env,
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        logger.error(f"Decoder failed:\n{result.stderr}")
        raise RuntimeError(f"Decoder failed: {result.stderr[:400]}")
    logger.info(f"  Decoder done in {time.time()-t0:.1f}s")


# =====================================================================
# Main
# =====================================================================

def main(args):
    repo = REPO_ROOT
    dlc_dir    = os.path.join(repo, "dlc")
    emb_dir    = os.path.join(repo, "embeddings")
    work_dir   = os.path.join(repo, "work")
    raw_dir    = os.path.join(work_dir, "raw_images")
    enc_out    = os.path.join(work_dir, "encoder_output")
    dec_out    = os.path.join(work_dir, "decoder_output")
    results_dir = os.path.join(work_dir, "results")

    for d in [raw_dir, enc_out, dec_out, results_dir]:
        os.makedirs(d, exist_ok=True)

    snpe_bin = find_snpe_net_run(args.snpe_root)
    snpe_env = get_snpe_env(args.snpe_root)
    use_dsp  = not args.use_cpu

    # Verify snpe-net-run is available
    try:
        r = subprocess.run([snpe_bin, "--version"], env=snpe_env,
                           capture_output=True, text=True, timeout=15)
        logger.info(f"snpe-net-run: {r.stdout.strip() or r.stderr.strip()}")
    except FileNotFoundError:
        logger.error(
            f"snpe-net-run not found. Install SNPE SDK and ensure it is in PATH, "
            f"or pass --snpe_root /path/to/sdk"
        )
        sys.exit(1)

    # ---- Load test CSV ----
    dataset_dir = os.path.join(repo, args.dataset_dir)
    data_csv    = os.path.join(dataset_dir, f"{args.data_name}.csv")
    import csv
    rows = []
    with open(data_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if args.max_images > 0:
        rows = rows[:args.max_images]
    logger.info(f"Loaded {len(rows)} samples from {data_csv}")

    # Resolve image/label paths relative to dataset_dir
    # (CSVs store paths like  test2/input/image.png  relative to gan-generated/)
    def abs_path(p):
        return p if os.path.isabs(p) else os.path.join(dataset_dir, p)

    # ---- Prompt embeddings ----
    pe_raw     = os.path.join(work_dir, "image_pe.raw")
    sparse_raw = os.path.join(work_dir, "sparse_prompt.raw")
    dense_raw  = os.path.join(work_dir, "dense_prompt.raw")
    np.load(os.path.join(emb_dir, "image_pe.npy")).astype(np.float32).tofile(pe_raw)
    np.load(os.path.join(emb_dir, "no_prompt_sparse.npy")).astype(np.float32).tofile(sparse_raw)
    np.load(os.path.join(emb_dir, "no_prompt_dense.npy")).astype(np.float32).tofile(dense_raw)

    # ---- Step 1: Preprocess ----
    logger.info("=" * 55)
    logger.info("Step 1: Preprocessing images")
    logger.info("=" * 55)
    raw_files  = []
    orig_sizes = []
    image_paths = []
    for idx, row in enumerate(rows):
        img_path = abs_path(row["image"])
        img = Image.open(img_path).convert("RGB")
        orig_sizes.append((img.height, img.width))
        image_paths.append(img_path)
        pv = preprocess_image(img_path)
        raw_path = os.path.join(raw_dir, f"image_{idx:04d}.raw")
        pv.tofile(raw_path)
        raw_files.append(raw_path)
    logger.info(f"Preprocessed {len(raw_files)} images")

    # ---- Step 2: Encoder ----
    logger.info("=" * 55)
    logger.info(f"Step 2: Encoder DLC  ({'NPU/HTP' if use_dsp else 'CPU'})")
    logger.info("=" * 55)
    encoder_dlc = os.path.join(dlc_dir, "convlora_sam_encoder_ptq.dlc")
    if not os.path.exists(encoder_dlc):
        logger.error(f"Encoder DLC not found: {encoder_dlc}")
        logger.error("Copy the .dlc files into the dlc/ folder.")
        sys.exit(1)
    run_encoder(snpe_bin, encoder_dlc, raw_files, enc_out, use_dsp, snpe_env)

    # ---- Step 3: Decoder ----
    logger.info("=" * 55)
    logger.info(f"Step 3: Decoder DLC  ({'NPU/HTP' if use_dsp else 'CPU'})")
    logger.info("=" * 55)
    decoder_dlc = os.path.join(dlc_dir, "convlora_sam_decoder_ptq.dlc")
    run_decoder(snpe_bin, decoder_dlc, len(rows), enc_out,
                (pe_raw, sparse_raw, dense_raw), dec_out, use_dsp, snpe_env)

    # ---- Step 4: IoU ----
    logger.info("=" * 55)
    logger.info("Step 4: Post-processing and IoU")
    logger.info("=" * 55)
    ious = []
    for idx, row in enumerate(rows):
        mask_path = os.path.join(dec_out, f"Result_{idx}", "low_res_masks.raw")
        if not os.path.exists(mask_path):
            logger.warning(f"No mask for image {idx}: {mask_path}")
            ious.append(0.0)
            continue

        orig_h, orig_w = orig_sizes[idx]
        raw_mask    = np.fromfile(mask_path, dtype=np.float32)
        mask_binary = postprocess_mask(raw_mask, orig_h, orig_w)

        label_path  = abs_path(row["label"])
        label_arr   = np.array(Image.open(label_path).convert("L")).astype(np.uint8)
        img_gray    = np.array(Image.open(image_paths[idx]).convert("L"))
        valid_region = get_valid_region(img_gray)
        pred_masked  = np.where(valid_region, mask_binary, 0).astype(np.uint8)
        gt_masked    = np.where(valid_region, label_arr, 0)
        iou = calculate_iou(pred_masked.astype(bool), gt_masked.astype(bool))
        ious.append(iou)
        logger.info(f"  Image {idx:3d}: IoU = {iou:.4f}")

    mean_iou = float(np.mean(ious)) if ious else 0.0
    logger.info("=" * 55)
    logger.info(f"Mean IoU ({len(ious)} images): {mean_iou:.4f}")
    logger.info("=" * 55)

    # Save results
    out_txt = os.path.join(results_dir, f"iou_{args.data_name}.txt")
    with open(out_txt, "w") as f:
        f.write(f"Dataset: {args.data_name}\n")
        f.write(f"Runtime: {'NPU/HTP (--use_dsp)' if use_dsp else 'CPU'}\n")
        f.write(f"Images evaluated: {len(ious)}\n")
        f.write(f"Mean IoU: {mean_iou:.4f}\n")
        f.write(f"Individual IoUs: {ious}\n")
    logger.info(f"Results saved to {out_txt}")

    # Cleanup raw input files
    if not args.keep_work:
        shutil.rmtree(raw_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conv-LoRA SAM DLC evaluation on IQ-9075 NPU"
    )
    parser.add_argument("--data_name",  default="test2_class2",
                        help="CSV name without extension, e.g. test2_class2 (inside --dataset_dir)")
    parser.add_argument("--dataset_dir", default="datasets/gan-generated",
                        help="Path (relative to repo root) containing CSV files and image subdirs")
    parser.add_argument("--snpe_root",  default=None,
                        help="Path to SNPE SDK root (optional if snpe-net-run is in PATH)")
    parser.add_argument("--use_cpu",    action="store_true",
                        help="Use CPU runtime instead of NPU/DSP (for debugging)")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Limit number of images (0 = all)")
    parser.add_argument("--keep_work",  action="store_true",
                        help="Keep raw files in work/ after evaluation")
    args = parser.parse_args()
    main(args)
