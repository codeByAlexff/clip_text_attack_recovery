import cv2
import numpy as np
import pipeline
from PIL import Image
import output_composition


def draw_boxes(image, result, output_path="masked_image.png"):
    annotated = np.array(image).copy()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
    for bbox, text, confidence in result:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        cv2.rectangle(annotated, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(annotated, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(output_path, annotated)

image = Image.open("dataset/input/input_image.jpg").convert("RGB")
result = pipeline.run(image)
output_composition.save_comparison(
    result["cleaned_image"],
    result["original_pred"],
    result["cleaned_pred"],
    "dataset/output/comparison.png"
)

print("Original:", result["original_pred"][:3])
print("Cleaned:", result["cleaned_pred"][:3])

draw_boxes(image, result["ocr_result"], "dataset/output/ocr_result.jpg")
result["cleaned_image"].save("dataset/output/cleaned_image.jpg")



