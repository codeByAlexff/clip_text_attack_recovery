import easyocr
import numpy as np
import cv2

reader = easyocr.Reader(['en'], gpu=False)

def detect_text(image):
    result = reader.readtext(np.array(image))
    return result


def get_coords(result):
    coords = []
    for bbox, text, confidence in result:
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        coords.append((top_left, bottom_right))
    return coords


