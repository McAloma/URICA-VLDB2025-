from PIL import Image
import numpy as np
from collections import Counter

def calculate_distribution_difference(image1, image2):
    def image_to_binary_array(image):
        gray_image = image.convert("L")
        array = np.array(gray_image)
        return (array > 0).astype(int)

    def normalize(array):
        total = np.sum(array)
        return array / total if total > 0 else array

    def calculate_integral_difference(dist1, dist2):
        unified_points = max(len(dist1), len(dist2))
        x1 = np.linspace(0, 1, len(dist1))
        x2 = np.linspace(0, 1, len(dist2))
        unified_x = np.linspace(0, 1, unified_points)
        interp_dist1 = np.interp(unified_x, x1, dist1)
        interp_dist2 = np.interp(unified_x, x2, dist2)
        integral_diff = np.trapz(np.abs(interp_dist1 - interp_dist2), unified_x)
        return integral_diff

    def compute_distribution(mask, axis):
        non_zero_counts = np.count_nonzero(mask, axis=axis)
        return normalize(non_zero_counts)

    mask1 = image_to_binary_array(image1)
    mask2 = image_to_binary_array(image2)
    dist1_horizontal = compute_distribution(mask1, axis=0)
    dist2_horizontal = compute_distribution(mask2, axis=0)
    dist1_vertical = compute_distribution(mask1, axis=1)
    dist2_vertical = compute_distribution(mask2, axis=1)
    horizontal_diff = calculate_integral_difference(dist1_horizontal, dist2_horizontal)
    vertical_diff = calculate_integral_difference(dist1_vertical, dist2_vertical)

    return (horizontal_diff + vertical_diff) / 2

def analyze_pixel_distribution(image_path):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    pixel_counts = Counter(image_array.flatten())
    total_pixels = image_array.size
    pixel_distribution = {value: {"count": count, "ratio": count / total_pixels}
                           for value, count in pixel_counts.items()}
    return pixel_distribution


if __name__ == "__main__":
    # # read mask
    # image_path = "data/Camelyon/mask/tumor_005_evaluation_mask.png"
    # distribution = analyze_pixel_distribution(image_path)

    # print("像素值分布情况:")
    # for pixel_value, stats in sorted(distribution.items()):
    #     print(f"像素值: {pixel_value}, 数量: {stats['count']}, 占比: {stats['ratio']:.4f}")

    # compare mask
    image_path1 = "data/Camelyon/mask/tumor_005_evaluation_mask.png"
    image1 = Image.open(image_path1).convert('L')

    image_path2 = "data/Camelyon/mask/tumor_008_evaluation_mask.png"
    image2 = Image.open(image_path2).convert('L')

    score = calculate_distribution_difference(image1, image2)
    print(score)