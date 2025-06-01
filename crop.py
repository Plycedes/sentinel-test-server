import cv2
import math
import os

def crop_bottom_middle(image_path, scale=41, crop_ratio=7):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    height, width = image.shape[:2]
    print("Original dimensions:", height, width)

    sc_height = height / scale
    sc_width = width / scale

    cr_height = math.floor(sc_height * crop_ratio)
    cr_width = math.floor(sc_width * crop_ratio)
    print("Crop size:", cr_height, cr_width)

    x_start = (width // 2) - (cr_width // 2)
    y_start = height - cr_height

    cropped = image[y_start:height, x_start:x_start + cr_width]

    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    new_filename = f'{name}_crp{ext}'
    save_path = os.path.join(os.path.dirname(image_path), new_filename)

    cv2.imwrite(save_path, cropped)
    os.unlink(image_path)
    print(f"Cropped image saved to: {save_path}")
    return save_path

    # cv2.imshow('Cropped', cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


crop_bottom_middle('qrs/CP1.png')

