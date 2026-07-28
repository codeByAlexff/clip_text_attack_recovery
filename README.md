# CLIP Typographic Attack
 
Vision-language models like CLIP don't just *see* images, they also read the text inside them. 
That's a vulnerability: stamp a misleading word onto a photo, and CLIP will often classify the image as the **word** rather than the **subject**.
A photo of a dog with "cat" printed across it gets labelled *cat*.
 
This project builds a pipeline that **detects** the adversarial text, **removes**
it (three different ways), and **re-classifies** — then measures how well each
removal method restores the correct prediction.
 
---

## The Attack in One Picture
 

| Typographic Attack | OCR Detection | LaMa Masking | Correct Prediction |
|----------|-------------------|-----------|-----------|
| ![attacked](assets/input_image.jpg) | ![ocr](assets/ocr_result.jpg) | ![cleaned](assets/cleaned_image.jpg) | ![comparison](assets/comparison.png)
 
The stamped word hijacks CLIP's prediction. Once the text is detected and
removed, classification recovers.
 
---
 
## How It Works
 
The pipeline runs in four stages:
 
1. **Classify** the attacked image with CLIP (`ViT-B/32`) — this is the fooled
   prediction.
2. **Detect** the stamped text with EasyOCR, returning bounding boxes.
3. **Remove** the text inside those boxes using one of three recovery methods.
4. **Re-classify** the cleaned image — this is the recovered prediction.
```
attacked image ──► CLIP ─────────────► "cat" (fooled)
       │
       ├──► EasyOCR ──► text boxes ──► remove text ──► cleaned image
                                                            │
                                                            └──► CLIP ──► "dog" (recovered)
```
 
### Text detection
 

![OCR text detection](assets/ocr_result.jpg)
 
EasyOCR locates the stamped word and returns the bounding box that the removal
step masks out.
 
---
 
## Recovery Methods
 
The project implements three strategies of increasing sophistication, so their
effectiveness can be compared directly:
 
| Method | Type | How it fills the region |
|--------|------|-------------------------|
| **Solid mask** | Crude baseline | Paints a solid box over the text |
| **OpenCV inpaint** | Classical CV | Diffuses surrounding pixels inward (`INPAINT_TELEA`) |
| **LaMa** | Neural | Reconstructs plausible background with a trained model |
 

![Solid Mask](assets/solid_fill_comparison.png) | ![OpenCV Inpaint](assets/opencv_infill_comparison.png) | ![LaMa Infill](assets/comparison.png)
 
---
 
## Results
 

| Recovery Method | Attack Success Rate | Recovery Rate |
|-----------------|--------------------|--------------:|
| Solid mask      | XX%                | XX%           |
| OpenCV inpaint  | XX%                | XX%           |
| LaMa            | XX%                | XX%           |
 
*Evaluated on a dataset of N images across M classes, each stamped with a
misleading label from the class set.*
 
**Key finding:** _<one or two sentences on what you actually observed — e.g.
"Neural inpainting recovered the true label more reliably than classical
methods on visually complex images, but all three performed similarly when the
text sat over simple backgrounds.">_
 
---
 
## Dataset
 
Attacked images are generated programmatically: each base image is stamped with a
misleading label (drawn from the class set, excluding its true label) using a
fixed position and size for controlled results. A manifest CSV records the true
label, the attack word, and file paths for every image, providing ground truth
for evaluation.
 

![Sample attacked images](assets/dataset_samples.png)
 
---
 
## Tech Stack
 
- **CLIP** (`ViT-B/32`) — zero-shot image classification
- **EasyOCR** — text detection
- **OpenCV** — classical inpainting + mask construction
- **LaMa** (`simple-lama-inpainting`) — neural inpainting
- **PyTorch**, **Pillow**, **NumPy**
---
 
## Project Structure
 
```
├── classifier.py      # CLIP loading + classification
├── ocr.py             # EasyOCR text detection + coordinate extraction
├── masking.py         # solid-box removal
├── infill.py          # OpenCV inpainting removal
├── ai_infill.py       # LaMa neural inpainting removal
├── pipeline.py        # orchestrates classify → detect → remove → re-classify
├── stamp.py           # generates the attacked dataset + manifest
├── visualize.py       # builds the comparison output cards
├── test.py            # runs the pipeline on a single image
└── assets/            # result images used in this README
```
 
---
 
## Setup
 
```bash
# Clone
git clone https://github.com/YOUR_USERNAME/clip-typographic-attack.git
cd clip-typographic-attack
 
# Create and activate a virtual environment
python -m venv env
source env/bin/activate        # macOS/Linux
# .\env\Scripts\activate       # Windows PowerShell
 
# Install dependencies
pip install -r requirements.txt
 
# CLIP (installed from source)
pip install git+https://github.com/openai/CLIP.git
```
 
> **GPU note:** for NVIDIA GPUs, install the CUDA build of PyTorch matching your
> card before the rest (e.g. `cu128` for RTX 50-series). On Apple Silicon and
> CPU-only machines the pipeline runs as-is, just slower.
 
---
 
## Usage
 
Run the pipeline on a single image:
 
```bash
python test.py
```
 
This classifies the image, detects and removes the stamped text, re-classifies,
and writes the comparison card plus the OCR detection image.
 
Generate the attacked dataset:
 
```bash
python stamp.py
```
 
---
 
## Why This Matters
 
Typographic attacks are a simple, real vulnerability in vision-language models —
no adversarial noise or gradient access required, just a word written on an
image. Studying detection-and-removal as a defense, and comparing how much the
*quality* of removal affects recovery, is a small window into how these models
weigh text against visual content.
