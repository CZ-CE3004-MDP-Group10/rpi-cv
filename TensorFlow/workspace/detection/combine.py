from __future__ import print_function
import os
import math
from PIL import Image


def collate_output_image():
    folder = 'images/detected_images'
    image_paths = [os.path.join(folder, f)
                   for f in os.listdir(folder) if f.endswith('.jpg')]

    number_images = len(image_paths)
    row = math.ceil(number_images / 4)
    col = math.ceil(number_images / row)

    result = Image.new("RGB", (500 * row, 500 * col))

    for index, file in enumerate(image_paths):
        path = os.path.expanduser(file)
        img = Image.open(path)
        img.thumbnail((500, 500), Image.ANTIALIAS)
        x = index // 4 * 500
        y = index % 4 * 500
        w, h = img.size
        # print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
        result.paste(img, (x, y, x + w, y + h))

    result.save('./images/output_images/final_output.jpg', 'JPEG')
    print("Output image generated :)")
