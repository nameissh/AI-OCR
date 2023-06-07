# Google Cloud Vision API
# text

import io
import os

# Set environment variable
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "mercurial-craft-387907-dab4d60b4a75.json"

# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('book01.jpg')

# Loads the image into memory, rb = 바이너리 읽기 모드
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = vision.Image(content=content)

# Performs label detection on the image file
responese = client.text_detection(image=image)
labels = responese.text_annotations

print('Text:')
for text in labels:
    print(text.description)