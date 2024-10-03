import os

source_image_path = 'data/source_images/image.jpg'
if os.path.exists(source_image_path):
    print(f"File exists: {source_image_path}")
else:
    print(f"File does not exist: {source_image_path}")