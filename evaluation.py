from pathlib import Path
from turtle import write
import classifier
from labels import labels
import csv
from PIL import Image
import ocr
import masking
import gc
import numpy as np
import matplotlib.pyplot as plt

directory = Path("dataset/attacked")
prediction_output = Path("dataset/output/")

CSV_PATH = prediction_output / "predictions.csv"
FIELDNAMES = [
    "File Name",
    "True Label",
    "Attacked Top 1 Label Prediction",
    "Solid Label Prediction",
    "OpenCV Label Prediction",
    "Lama Label Prediction"
]

def process_images():

    #Write header if file does not exist
    write_header = not CSV_PATH.exists()

    with open(CSV_PATH, "a", newline="", encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for file in directory.rglob("*"):
            if file.is_file() and file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    try:
                        with Image.open(file) as img:
                            img.thumbnail((800,800))
                            img = img.convert("RGB")

                            #Set Parameters
                            true_label = file.parent.name
                            ocr_result = ocr.detect_text(img)
                            coords = ocr.get_coords(ocr_result)

                            #Initial Attacked Prediction
                            attacked_top1 = classifier.classify_image(img, labels, 1)[0][0]

                            #Masking Methods
                            solid = masking.solid_mask(img, coords)          
                            opencv = masking.cv_inpaint(img, coords)
                            lama = masking.infill(img, coords)

                            #Top 1 Prediction From Masking Methods
                            solid_label = classifier.classify_image(solid, labels, 1)[0][0]
                            opencv_label = classifier.classify_image(opencv, labels, 1)[0][0]
                            lama_label = classifier.classify_image(lama, labels, 1)[0][0]

                            writer.writerow({
                                "File Name": file.name,
                                "True Label": true_label,
                                "Attacked Top 1 Label Prediction": attacked_top1,
                                "Solid Label Prediction": solid_label,
                                "OpenCV Label Prediction": opencv_label,
                                "Lama Label Prediction": lama_label,
                            })
                        gc.collect()
                    except Exception as e:
                        print(f"Error processing {file}: {e}")

def read_predictions():
    with open(CSV_PATH, "r", newline="", encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def attack_success_rate_and_recovery(csv_list):
    #Compute Attack Success Rate (ASR)
    asr_count = 0
    successful_attacks = []
    for row in csv_list:
        if row["Attacked Top 1 Label Prediction"] != row["True Label"]:
            asr_count += 1
            successful_attacks.append(row)
    asr = asr_count / len(csv_list) * 100 if len(csv_list) > 0 else 0  # Convert to percentage
    total = len(csv_list)

    #Recovery Rate (RR)
    def recovery_rate(field):
        if not successful_attacks:
            return 0.0
        recovered = sum(1 for r in successful_attacks if r[field] == r["True Label"])
        return recovered / len(successful_attacks) * 100 if len(successful_attacks) > 0 else 0  # Convert to percentage

    solid_rr = recovery_rate('Solid Label Prediction')
    opencv_rr = recovery_rate('OpenCV Label Prediction')
    lama_rr = recovery_rate('Lama Label Prediction')

    print(f"Total Images: {total}")
    print(f"Successfull Attacks: {asr_count}")
    print(f"Attack Success Rate (ASR): {asr:.2f}%")
    print(f"Recovery Rate (Solid Mask): {solid_rr:.2f}%")
    print(f"Recovery Rate (OpenCV): {opencv_rr:.2f}%")
    print(f"Recovery Rate (Lama): {lama_rr:.2f}%")

    plt.bar(['Total', 'ASR', 'Solid', 'OpenCV', 'Lama'], [total, asr, solid_rr, opencv_rr, lama_rr])
    plt.ylim()
    plt.ylabel("Percentage (%)")
    plt.bar_label(plt.gca().containers[0])
    plt.tight_layout()
    plt.savefig("dataset/output/attack_success_and_recovery_rates.png")

    
    


if __name__ == "__main__":
    process_images()
    csv_list = read_predictions()
    attack_success_rate_and_recovery(csv_list)


    









                
    
                



