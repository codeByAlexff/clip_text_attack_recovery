import classifier
from PIL import Image
from pathlib import Path
import ocr
from labels import labels
import masking
import csv
import gc
import torch

INPUT_DIR = Path("dataset/attacked/")
OUTPUT_CSV = Path("dataset/output/predictions.csv")

valid_extensions = {".jpg", ".jpeg", ".png"}
MAX_DIM = 768  # resize large images down to this before processing, to save memory

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
header = ["file", "true_label", "attacked_top1", "solid_top1", "opencv_top1", "lama_top1"]

files = [f for f in INPUT_DIR.glob("**/*") if f.suffix.lower() in valid_extensions]
print(f"Found {len(files)} files")

# Resume support: skip files already recorded in an existing CSV
already_done = set()
if OUTPUT_CSV.exists():
    with open(OUTPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            already_done.add(row["file"])
    print(f"Resuming — {len(already_done)} files already processed")

write_header = not OUTPUT_CSV.exists()

def resize_if_needed(img, max_dim=MAX_DIM):
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)

with open(OUTPUT_CSV, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    if write_header:
        writer.writeheader()

    for i, filepath in enumerate(files):
        if str(filepath) in already_done:
            continue

        true_label = filepath.parent.name
        print(f"[{i+1}/{len(files)}] Processing: {true_label}/{filepath.name}")

        try:
            with Image.open(filepath) as raw_img:
                img = raw_img.convert("RGB")
            img = resize_if_needed(img)

            coords = ocr.get_coords(ocr.detect_text(img))

            with torch.no_grad():
                attacked_top1 = classifier.classify_image(img, labels, 1)[0][0]

                solid_img = masking.solid_mask(img, coords)
                solid_top1 = classifier.classify_image(solid_img, labels, 1)[0][0]

                opencv_img = masking.cv_inpaint(img, coords)
                opencv_top1 = classifier.classify_image(opencv_img, labels, 1)[0][0]

                lama_img = masking.infill(img, coords)
                lama_top1 = classifier.classify_image(lama_img, labels, 1)[0][0]

            row = {
                "file": str(filepath),
                "true_label": true_label,
                "attacked_top1": attacked_top1,
                "solid_top1": solid_top1,
                "opencv_top1": opencv_top1,
                "lama_top1": lama_top1,
            }
            writer.writerow(row)
            f.flush()  # ensure it's actually written to disk immediately

            del img, solid_img, opencv_img, lama_img

        except Exception as e:
            print(f"  ERROR on {filepath}: {e}")
            continue

        if i % 5 == 0:
            gc.collect()

print(f"\nDone. Records saved to {OUTPUT_CSV}")

# --- Aggregate metrics from the CSV (works even after a resumed/partial run) ---
records = []
with open(OUTPUT_CSV, "r", newline="") as f:
    reader = csv.DictReader(f)
    records = list(reader)

total = len(records)
attacked_success = [r for r in records if r["attacked_top1"] != r["true_label"]]
num_attacked_success = len(attacked_success)

asr = (num_attacked_success / total * 100) if total > 0 else 0.0

def recovery_rate(method_key):
    if num_attacked_success == 0:
        return 0.0
    num_recovered = sum(1 for r in attacked_success if r[method_key] == r["true_label"])
    return num_recovered / num_attacked_success * 100

solid_rate = recovery_rate("solid_top1")
opencv_rate = recovery_rate("opencv_top1")
lama_rate = recovery_rate("lama_top1")

print("\n--- Summary ---")
print(f"Total images:               {total}")
print(f"Successful attacks:         {num_attacked_success}")
print(f"Attack Success Rate (ASR):  {asr:.2f}%")
print(f"Recovery Rate (Solid Mask): {solid_rate:.2f}%")
print(f"Recovery Rate (OpenCV):     {opencv_rate:.2f}%")
print(f"Recovery Rate (LaMa):       {lama_rate:.2f}%")