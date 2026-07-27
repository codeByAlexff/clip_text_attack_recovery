import cv2
from PIL import Image
import numpy as np
from simple_lama_inpainting import SimpleLama

simple_lama = SimpleLama()

def infill(image, coords):
    image = np.array(image).copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for top_left, bottom_right in coords:
        cv2.rectangle(mask, top_left, bottom_right, 255, -1)
    #result = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
    image_pil = Image.fromarray(image)
    mask_pil = Image.fromarray(mask)
    result = simple_lama(image_pil, mask_pil)
    return result