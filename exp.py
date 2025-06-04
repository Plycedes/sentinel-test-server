import cv2
import os

def extract_qr_and_split_strip(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width, _ = image.shape
    scale_x = width / 37
    scale_y = height / 25

    qr_w = int(round(scale_x * 25))
    qr_h = int(round(scale_y * 25))
    strip_w = int(round(scale_x * 12))
    strip_h = qr_h

    border_x = int(round(scale_x * 1))
    border_y = int(round(scale_y * 1))

    # Extract 25x25 QR region
    qr_region = image[0:qr_h, 0:qr_w]

    # Extract 12x25 strip region
    x_start = qr_w + border_x
    x_end = qr_w + strip_w - border_x
    y_start = border_y
    y_end = strip_h - border_y
    strip_region = image[y_start:y_end, x_start:x_end]

    # Split strip into top and bottom (originally 13 + 12)
    top_half_h = int(round(scale_y * 13))
    bottom_half_h = int(round(scale_y * 12))

    # Crop 3 units from bottom of top half
    strip_top = strip_region[0:top_half_h - int(round(scale_y * 3)), :]

    # Crop bottom half
    strip_bottom = strip_region[top_half_h : top_half_h + bottom_half_h, :]

    directory = os.path.dirname(image_path)
    basename = os.path.splitext(os.path.basename(image_path))[0]

    qr_path = os.path.join(directory, f"{basename}_qr.png")
    noise_path = os.path.join(directory, f"{basename}_strip_top_12x11.png")
    spiral_path = os.path.join(directory, f"{basename}_strip_bottom_12x11.png")

    # Similarly also write the qr and noise pattern images and return their paths
    #cv2.imwrite(qr_path, qr_region)
    #cv2.imwrite(noise_path, strip_top)
    cv2.imwrite(spiral_path, strip_bottom)

    #return qr_path, noise_path, spiral_path
    return spiral_path
