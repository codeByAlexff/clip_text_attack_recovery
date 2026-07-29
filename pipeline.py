import classifier
import ocr
import masking
from labels import labels

def run(image):
    original_pred = classifier.classify_image(image, labels)

    ocr_result = ocr.detect_text(image)
    coords = ocr.get_coords(ocr_result)

    cleaned_image = masking.infill(image, coords)

    cleaned_pred = classifier.classify_image(cleaned_image, labels)

    return {
    "original_pred": original_pred,
    "cleaned_pred": cleaned_pred,
    "ocr_result": ocr_result,
    "cleaned_image": cleaned_image,
}
