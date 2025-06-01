import os
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as compare_ssim
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import random
import joblib
from scipy.stats import entropy
from skimage.feature import local_binary_pattern

class PHash:
    def __init__(self, hash_size=16, high_freq_factor=8):
        self.hash_size = hash_size
        self.high_freq_factor = high_freq_factor

    def _preprocess_image(self, image):
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        gray = cv2.resize(gray, (self.hash_size * self.high_freq_factor, 
                                self.hash_size * self.high_freq_factor),
                         interpolation=cv2.INTER_AREA)
        return gray

    def compute_hash(self, image):
        gray = self._preprocess_image(image)
        dct = cv2.dct(np.float32(gray))
        dct_low_freq = dct[:self.hash_size, :self.hash_size]
        median = np.median(dct_low_freq.flatten())
        hash_bits = (dct_low_freq > median).astype(np.uint8)
        return hash_bits

    # def similarity(self, hash1, hash2):
    #     distance = np.linalg.norm(hash1 - hash2)
    #     similarity_value = -(1 - distance / np.sqrt(len(hash1)))
        
    #     min_similarity = 1 
    #     max_similarity = 10 

    #     normalized_similarity = (similarity_value / 1000) * 10
        
    #     return normalized_similarity    

    def similarity(self, hash1, hash2):
        if hash1.shape != hash2.shape:
            raise ValueError("Hash shapes do not match")

        # Flatten both hashes and calculate Hamming distance
        hamming_distance = np.count_nonzero(hash1.flatten() != hash2.flatten())
        total_bits = hash1.size

        # Similarity is 1.0 for identical, 0.0 for completely different
        similarity = 1 - (hamming_distance / total_bits)
        rounded_similarity = round(similarity, 4)
        return rounded_similarity * 10



def calculate_ssim_similarity(image1, image2):
    if image1.shape != image2.shape:
        image2 = np.array(image2, dtype=np.uint8)
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    similarity_index, _ = compare_ssim(image1, image2, full=True)
    return similarity_index

def texture_similarity(image1, image2):
    glcm1 = graycomatrix(image1, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm2 = graycomatrix(image2, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    props = ['contrast', 'homogeneity', 'energy', 'correlation']
    similarity_score = 0
    for prop in props:
        score1 = graycoprops(glcm1, prop)[0, 0]
        score2 = graycoprops(glcm2, prop)[0, 0]
        similarity_score += abs(score1 - score2)

    return 1 / (1 + similarity_score) 

def pixel_intensity_distribution_similarity(image1, image2, bins=4):
    hist1, _ = np.histogram(image1, bins=bins, range=(0, 256))
    hist2, _ = np.histogram(image2, bins=bins, range=(0, 256))

    hist1 = hist1 / hist1.sum() 
    hist2 = hist2 / hist2.sum()

    similarity = np.sum(np.minimum(hist1, hist2)) 
    return similarity

def detect_qr_and_extract_noise_cordinates(image):
    if image is None:
        raise ValueError("Could not read the image")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        try:
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)

            if perimeter == 0:
                continue

            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            print(f"Contour Area: {area}, Circularity: {circularity}, Perimeter: {perimeter}")
            
            if (0.5 < circularity < 1.2 and 500 <= perimeter <= 1500) or (0.4 < circularity < 0.5 and 500 <= perimeter <= 1500):
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius) - 3
                
                square_size = int((19/29) * (2 * radius))  
                half_size = square_size // 2
                
                square_corners = [
                    (center[0] - half_size, center[1] - half_size),  
                    (center[0] + half_size, center[1] - half_size), 
                    (center[0] + half_size, center[1] + half_size),  
                    (center[0] - half_size, center[1] + half_size)   
                ]
                
                x_start = max(0, center[0] - radius)
                x_end = min(image.shape[1], center[0] + radius)
                y_start = max(0, center[1] - radius)
                y_end = min(image.shape[0], center[1] + radius)
                
                cropped_circle = image[int(y_start):int(y_end), int(x_start):int(x_end)]
                
                cropped_square_corners = [
                    (corner[0] - x_start, corner[1] - y_start) for corner in square_corners
                ]
                
                return cropped_circle, (center[0], center[1], radius), cropped_square_corners
        
        except ZeroDivisionError:
            print("ZeroDivisionError: Perimeter is zero for contour")
            continue
    
    return None, None, None


def crop_image_by_coordinates(image, coordinates):
    x, y, w, h = coordinates
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image

def generate_noise_pattern_svg(text: str, grid_size: int) -> str:
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    random.seed(seed)
    
    cell_size = 400 // grid_size
    
    grid = []
    for _ in range(grid_size):
        row = []
        for _ in range(grid_size):
            value = random.choice([True, False])
            row.append(value)
        grid.append(row)
    
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 400">\n'
    
    svg += f'  <rect x="0" y="0" width="400" height="400" fill="white" />\n'
    
    for y in range(grid_size):
        for x in range(grid_size):
            if grid[y][x]:
                svg += f'  <rect x="{x * cell_size}" y="{y * cell_size}" '
                svg += f'width="{cell_size}" height="{cell_size}" '
                svg += f'fill="black" />\n'
    
    svg += '</svg>'
    return svg

def generate_noise_pattern_array(text: str, grid_size: int) -> np.ndarray:
    print(f"Generating noise pattern for text '{text}' with grid size {grid_size}")
    seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
    random.seed(seed)
    
    grid = np.zeros((grid_size, grid_size), dtype=bool)
    for y in range(grid_size):
        for x in range(grid_size):
            grid[y, x] = random.choice([False, True])  
    
    return grid

def resize_binary_image(noise_image_extracted, grid_size):
    if len(noise_image_extracted.shape) == 3:
        if noise_image_extracted.shape[2] == 4:  # RGBA
            noise_image_extracted = np.dot(noise_image_extracted[..., :3], [0.2989, 0.5870, 0.1140])
        elif noise_image_extracted.shape[2] == 3:  # RGB
            noise_image_extracted = cv2.cvtColor(noise_image_extracted, cv2.COLOR_RGB2GRAY)
    noise_image_extracted = noise_image_extracted.astype(np.uint8)
    
    _, binary_image = cv2.threshold(noise_image_extracted, 127, 255, cv2.THRESH_BINARY)
    
    target_size = (grid_size, grid_size)
    
    resized_image = cv2.resize(binary_image, 
                              target_size, 
                              interpolation=cv2.INTER_NEAREST)
    
    _, final_binary = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)
    
    return final_binary

def calculate_image_correlation(img1, img2):
    correlation = np.corrcoef(img1.flatten(), img2.flatten())[0, 1]
    return correlation

def flann_matcher(img1, img2):
    if img1 is None or img2 is None:
        print("Invalid images provided!")
        return None

    sift = cv2.SIFT_create(
        nfeatures=2000, 
        contrastThreshold=0.005,  
        edgeThreshold=8,  
        sigma=0.9
    )

    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    if descriptors1 is None or descriptors2 is None:
        print("Could not compute descriptors!")
        return None

    # print(f"Descriptors Shape: {descriptors1.shape}, {descriptors2.shape}")

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches_img2_img2 = flann.knnMatch(descriptors2, descriptors2, k=2)
    good_matches_img2_img2 = [m for m, n in matches_img2_img2 if m.distance < 0.85 * n.distance]
    count_img2_img2 = len(good_matches_img2_img2)

    matches_img1_img2 = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches_img1_img2 = [m for m, n in matches_img1_img2 if m.distance < 0.85 * n.distance]
    count_img1_img2 = len(good_matches_img1_img2)

    match_ratio = count_img1_img2 / count_img2_img2 if count_img2_img2 > 0 else 0

    # print(f"Matches (img2, img2): {count_img2_img2}")
    # print(f"Matches (img1, img2): {count_img1_img2}")
    # print(f"Match Ratio: {match_ratio:.2f}")

    return match_ratio

def load_image_matrix(image, target_size=(160, 160)):
    """Resizes an image to target size and converts it to a pixel matrix."""
    # Ensure the image has 3 dimensions (convert grayscale to RGB)
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB

    height, width, channels = image.shape  # Now safe to unpack

    pixel_matrix = [[[int(image[i, j, c]) for c in range(channels)] for j in range(width)] for i in range(height)]
    
    return pixel_matrix, width, height, channels

def compute_mse_matrix(img1_matrix, img2_matrix, width, height, channels, weight_rgb=(0.3, 0.3, 0.4), intensity_factor=1.0):
    """Manually calculates MSE between two image pixel matrices with RGB weighting."""
    mse_total = 0
    total_pixels = width * height * channels
    
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                weight = weight_rgb[c]
                diff = (img1_matrix[i][j][c] - img2_matrix[i][j][c]) ** 2
                mse_total += weight * diff * intensity_factor

    return mse_total / total_pixels  

def preprocess_image_for_mse(image):
    """Mimic cv2.imread behavior for uniform processing of images."""
    if image.dtype == np.bool_:
        image = image.astype(np.uint8) * 255
    
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if len(image.shape) == 2:  
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    elif image.shape[2] == 4:  
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    return image

def calculate_mse_similarity(image1, image2, target_size=(160, 160), weight_rgb=(0.3, 0.3, 0.4), intensity_factor=1.0):
    img1_matrix, width, height, channels = load_image_matrix(image1, target_size)
    img2_matrix, _, _, _ = load_image_matrix(image2, target_size)
    
    mse_value = compute_mse_matrix(img1_matrix, img2_matrix, width, height, channels, weight_rgb, intensity_factor)
    return mse_value

def calculate_similarity(image1, image2, phash, image_path1, image_path2, output_folder):
    def convert_to_grayscale(img):
        if img is None:
            raise ValueError("Received an empty image")
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img.astype(np.uint8)
    
    def resize_image(img, size=(160, 160)):
        if img is None or img.size == 0:
            raise ValueError("Cannot resize an empty or invalid image")
        # img = img.astype(np.uint8)
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA if img.shape[0] > 320 else cv2.INTER_LINEAR)
    
    def resize_noise_image(img, size=(120, 120)):
        if img is None or img.size == 0:
            raise ValueError("Cannot resize an empty or invalid image")
        img = (img * 255).astype(np.uint8)
        return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)


    def resize_image_for_ber(img, size=(40, 40)):
        if img is None or img.size == 0:
            raise ValueError("Cannot resize an empty or invalid image")
        return cv2.resize(img, size)

    if image1 is None or image2 is None:
        raise ValueError("One or both images are empty")

    def adaptive_gaussian_luminosity(image):
        image = cv2.resize(image, (120, 120), interpolation=cv2.INTER_AREA)
        if len(image.shape) == 3 and image.shape[2] == 3:
            luminosity = 0.21 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.07 * image[:, :, 0]
        else:
            luminosity = image

        binary = cv2.adaptiveThreshold(
            luminosity.astype(np.uint8), 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            215, -5
        )
        # inverted_binary = cv2.bitwise_not(binary)
        
        return binary       
    
    image2_resized = image2

    image1_resized = adaptive_gaussian_luminosity(image1)
    
   

    image1_name = os.path.splitext(os.path.basename(image_path1))[0]
    image2_name = os.path.splitext(os.path.basename(image_path2))[0]
    
    unprocessed_image = cv2.resize(image1, (160, 160), interpolation=cv2.INTER_NEAREST)    

    aligned_noise_image, ecc_score = align_using_ecc(image2_resized, image1_resized)
    

    hash1 = phash.compute_hash(aligned_noise_image)
    hash2 = phash.compute_hash(image2_resized)
    phash_similarity = phash.similarity(hash1, hash2)    
    
    # flann_matches = flann_matcher(image1_resized, image2_resized)
    # correlation = calculate_image_correlation(image1_resized, image2_resized)
    # ssim_similarity = calculate_ssim_similarity(image1_resized, image2_resized)
    # mse_similarity = calculate_mse_similarity(image1_resized, image2_resized)
    # white_pixel_loss = calculate_white_pixel_loss_percentage(image2_resized, image1_resized)
    # fft_correlation = compute_fft_correlation(image1_resized, image2_resized)

    flann_matches = flann_matcher(aligned_noise_image, image2_resized)    

    if aligned_noise_image.shape != image2_resized.shape:
        image2_resized = cv2.resize(image2_resized, (aligned_noise_image.shape[1], aligned_noise_image.shape[0]))

   

    correlation = calculate_image_correlation(aligned_noise_image, image2_resized)
    ssim_similarity = calculate_ssim_similarity(aligned_noise_image, image2_resized)
    mse_similarity = calculate_mse_similarity(aligned_noise_image, image2_resized)
    white_pixel_loss = calculate_white_pixel_loss_percentage(image2_resized, aligned_noise_image)
    fft_correlation = compute_fft_correlation(aligned_noise_image, image2_resized)

    ber_image1 = resize_image_for_ber(aligned_noise_image)
    ber_image2 = resize_image_for_ber(image2_resized)
    ber_score = compute_ber(ber_image1, ber_image2)

    return ssim_similarity, phash_similarity, flann_matches, correlation, mse_similarity, white_pixel_loss, fft_correlation, ber_score, ecc_score, image1_resized, image2_resized, aligned_noise_image, unprocessed_image
    # return ssim_similarity, phash_similarity, flann_matches, correlation, mse_similarity, white_pixel_loss, fft_correlation, ber_score, ecc_score, processed_image, reference_pattern, aligned_noise_image, unprocessed_image

def crop_image_by_corners(image, coordinates):
    if image is None:
        raise ValueError("Could not read the image")

    x_min = min(point[0] for point in coordinates)
    y_min = min(point[1] for point in coordinates)
    x_max = max(point[0] for point in coordinates)
    y_max = max(point[1] for point in coordinates)

    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image
    
def calculate_white_pixel_loss_percentage(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1 is None or img2 is None:
        raise ValueError("One or both images are None.")

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    white_pixels_1 = np.count_nonzero(img1 == 255)
    white_pixels_2 = np.count_nonzero(img2 == 255)

    loss = abs(white_pixels_1 - white_pixels_2)

    if white_pixels_1 == 0:
        return 0.0 if white_pixels_2 == 0 else 100.0

    loss_percentage = (loss / white_pixels_1) * 100
    
    return round(loss_percentage, 2)


def compute_fft_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Computes the FFT correlation between two images.
    """
    # Ensure both images are grayscale and have the same dimensions
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Compute the 2D FFT of both images
    fft_img1 = np.fft.fft2(img1)
    fft_img2 = np.fft.fft2(img2)

    # Compute the magnitude spectra
    mag_img1 = np.abs(fft_img1)
    mag_img2 = np.abs(fft_img2)

    # Normalize magnitudes
    mag_img1 = (mag_img1 - np.min(mag_img1)) / (np.max(mag_img1) - np.min(mag_img1) + 1e-6)
    mag_img2 = (mag_img2 - np.min(mag_img2)) / (np.max(mag_img2) - np.min(mag_img2) + 1e-6)

    # Compute correlation coefficient
    correlation = np.corrcoef(mag_img1.flatten(), mag_img2.flatten())[0, 1]
    
    return correlation

def align_using_ecc(reference_image, target_image):
    """
    Align the target image to the reference image using ECC (Enhanced Correlation Coefficient).
    Returns the aligned image and ECC score.
    """
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY) if len(reference_image.shape) == 3 else reference_image
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY) if len(target_image.shape) == 3 else target_image

    ref_gray = np.float32(ref_gray)
    target_gray = np.float32(target_gray)

    target_gray = cv2.resize(target_gray, (ref_gray.shape[1], ref_gray.shape[0]))

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

    try:
        ecc_score, warp_matrix = cv2.findTransformECC(ref_gray, target_gray, warp_matrix, cv2.MOTION_AFFINE, criteria)
        aligned_image = cv2.warpAffine(target_image, warp_matrix, (ref_gray.shape[1], ref_gray.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        return aligned_image, round(ecc_score, 4)
    except Exception as e:
        print(f"ECC alignment failed", e)
        return target_image, None

def binarize_image(image, threshold=128):
    """
    Converts an image to binary using Otsu's thresholding.
    """
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def compute_ber(image1, image2):
    """
    Computes the Bit Error Rate (BER) between two binary images.
    BER = (Number of differing pixels / Total pixels) * 100
    """
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    
    binary1 = binarize_image(image1)
    binary2 = binarize_image(image2)
    
    differing_pixels = np.sum(binary1 != binary2)
    total_pixels = binary1.shape[0] * binary1.shape[1]
    
    ber = (differing_pixels / total_pixels) * 100
    return round(ber, 4)

def process_image_noise(image_path1, image_path2, output_folder):
    image1 = cv2.imread(image_path1)
    if image1 is None:
        return None, None, None, None, None, None
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    image2 = cv2.imread(image_path2)
    if image2 is None:
        return None, None, None, None, None, None
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        
    phash = PHash(hash_size=16, high_freq_factor=8)

    # cropped_image, circle_params, sqaure_corners = detect_qr_and_extract_noise_cordinates(image)
    
    # if (cropped_image is None or cropped_image.size == 0) and \
    #     (circle_params is None or circle_params.size == 0) and \
    #     sqaure_corners is None:
    #     return None, None, None, None, None, None
    
    # // Add your logic of extraction here

    # noise_image_extracted = crop_image_by_corners(cropped_image, sqaure_corners)
    # reference_noise_generated = generate_noise_pattern_array(text, grid_size)
    
    # temp_dir = os.path.join(os.getcwd(), "temp")
    # os.makedirs(temp_dir, exist_ok=True)
    # extracted_noise_path = os.path.join(temp_dir, f"{request_id}_extracted_noise.png")
    # cv2.imwrite(extracted_noise_path, noise_image_extracted)
    
    # noise_image_extracted = cv2.imread(extracted_noise_path)

    ssim_similarity, phash_similarity, flann_matches, correlation, mse_similarity, white_pixel_loss, fft_correlation, ber_score, ecc_score, processed_image, reference_pattern, aligned_noise_image, unprocessed_image = calculate_similarity(image1, image2, phash, image_path1, image_path2, output_folder)

    if ecc_score is None:
        ecc_score = 0

    base_name = os.path.basename(image_path1)
    name, ext = os.path.splitext(base_name)

    output = (
    f"\n{name}\n"
    f"pHash: {phash_similarity}\n"
    f"SSIM: {ssim_similarity}\n"
    f"flann: {flann_matches}\n"
    f"Cor: {correlation}\n"
    f"MSE: {mse_similarity}\n"
    f"ECC: {ecc_score}\n"
    )

    # output = (
    # # f"\n{name}\n"
    # # f"{phash_similarity:.3f}\n"
    # f"{ssim_similarity:.3f}\n"
    # # f"{flann_matches:.3f}\n"
    # # f"{correlation:.3f}\n"
    # f"{mse_similarity:.0f}\n"
    # f"{ecc_score:.3f}"
    # )
    
    print(output)
    
    # with open("results.txt", "a") as f:
    #     f.write(output + "\n")


    # if phash_similarity > 3.8:
    #     classification = 0
    # elif phash_similarity < 3.8:
    #     MODEL_PATH = os.path.join(os.getcwd(), "logistic_regression_model_now.pkl")
    #     SCALER_PATH = os.path.join(os.getcwd(), "feature_scaler_model_now.pkl")

    #     if not os.path.exists(MODEL_PATH):
    #         raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    #     if not os.path.exists(SCALER_PATH):
    #         raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

    #     model = joblib.load(MODEL_PATH)
    #     scaler = joblib.load(SCALER_PATH)

    #     feature_names = ["ssim_score", "mse_score", "flann_score", "corelation_score", "fft_correlation_score", "ber_score", "ecc_score"]

    #     feature_vector = pd.DataFrame(
    #         [[ssim_similarity, mse_similarity, flann_matches, correlation, fft_correlation, ber_score, ecc_score]],
    #         columns=feature_names
    #     )

    #     feature_vector_scaled = scaler.transform(feature_vector)
    #     classification = model.predict(feature_vector_scaled)[0]
    #     classification = 1 - classification

    # if os.path.exists(extracted_noise_path):
    #     os.remove(extracted_noise_path)
    
    # return processed_image, unprocessed_image, aligned_noise_image, reference_pattern, (ssim_similarity, phash_similarity, flann_matches, correlation, mse_similarity, white_pixel_loss, fft_correlation, ber_score, ecc_score), classification
    os.unlink(image_path1)
    return ssim_similarity

process_image_noise("consistency/P1/CP17/CP4_crp.png", "ref15/rps/RP30.png", "consistency/P1/CP17/resized" )
process_image_noise("consistency/P1/CP18/CP4_crp.png", "ref15/rps/RP30.png", "consistency/P1/CP18/resized" )
process_image_noise("consistency/P1/CP19/CP4_crp.png", "ref15/rps/RP30.png", "consistency/P1/CP19/resized" )
process_image_noise("consistency/P1/CP20/CP4_crp.png", "ref15/rps/RP30.png", "consistency/P1/CP20/resized" )
process_image_noise("consistency/P1/CP21/CP4_crp.png", "ref15/rps/RP30.png", "consistency/P1/CP21/resized" )


