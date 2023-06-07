# Google Cloud Vision API
# text + image

import io
import os
import numpy as np
import platform
from PIL import ImageFont, ImageDraw, Image
##from utils import plt_imshow
import cv2

# Set environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mercurial-craft-387907-dab4d60b4a75.json"

# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
path = os.path.abspath('book01.jpg')

# Loads the image into memory, rb = 바이너리 읽기 모드
with io.open(path, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# Performs label detection on the image file
response = client.text_detection(image=image)
texts = response.text_annotations

print('Text:')
for text in texts:
    print(text.description)

# image putText    
img = cv2.imread(path)
roi_img = img.copy()
    
for text in texts:
    print('\n"{}"'.format(text.description))

    vertices = (['({},{})'.format(vertex.x, vertex.y)
                for vertex in text.bounding_poly.vertices])
    
    ocr_text = text.description
    x1 = text.bounding_poly.vertices[0].x
    y1 = text.bounding_poly.vertices[0].y
    x2 = text.bounding_poly.vertices[1].x
    y2 = text.bounding_poly.vertices[2].y
    
    cv2.rectangle(roi_img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    ##roi_img = cv2.putText(roi_img, ocr_text, x1, y1 - 30, font_size=30)
    roi_img = cv2.putText(roi_img, ocr_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


if response.error.message:
    raise Exception(
        '{}\nFor more info on error messages, check: '
        'https://cloud.google.com/apis/design/errors'.format(
            response.error.message))
    
##plt_imshow(["Original", "ROI"], [img, roi_img], figsize=(16, 10))

cv2.imshow("Original", img)
cv2.imshow("ROI", roi_img)
cv2.waitKey(0)
cv2.destroyAllWindows()