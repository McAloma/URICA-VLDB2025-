import cv2, os, math
from PIL import Image
import numpy as np
from openslide import OpenSlide


def WSI_background_detect(wsi_image):
    gray_image = wsi_image.convert("L")
    image_np = np.array(gray_image)
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    _, binary_image = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image_pil = Image.fromarray(binary_image)

    return binary_image_pil

def load_wsi_thumbnail(slide):
    levels = slide.level_count
    width, height = slide.level_dimensions[levels-1][0], slide.level_dimensions[levels-1][1]
    thumbnail = slide.read_region((0, 0), levels-1, (width, height))   
    thumbnail = thumbnail.convert('RGB')

    background = WSI_background_detect(thumbnail)

    return thumbnail, background, levels

def get_patch_background_ratio(slide, background, infos):
    target_level = infos["level"]
    ratio = slide.level_downsamples[-1] / slide.level_downsamples[target_level]

    x, y = int(infos["position"][0]) // ratio, int(infos["position"][1]) // ratio
    w, h = int(infos["size"][0]) // ratio, int(infos["size"][1]) // ratio

    patch_background = background.crop((x, y, x + w, y + h))
    pixels = list(patch_background.getdata())

    white_pixel_ratio = pixels.count(255) / len(pixels)
    return patch_background, white_pixel_ratio

def get_region_background_ratio(slide, background, infos):
    target_level = infos["level"]
    ratio = slide.level_downsamples[-1] / slide.level_downsamples[target_level]

    x, y = int(infos["position"][0] // ratio), int(infos["position"][1] // ratio)
    w, h = int(infos["size"][0] // ratio), int(infos["size"][1] // ratio)
    angle = infos["angle"]

    canvas_size = int(math.sqrt(w**2 + h**2)) 
    canvas = Image.new("1", (canvas_size, canvas_size), 255)


    cropped = background.crop((x-canvas_size//2, y-canvas_size//2, x+canvas_size//2, y+canvas_size//2))
    canvas.paste(cropped, (0, 0))
    rotated_canvas = canvas.rotate(-angle, resample=Image.BICUBIC, expand=False)

    final_crop_left = (rotated_canvas.width - w) // 2
    final_crop_top = (rotated_canvas.height - h) // 2
    region_background = rotated_canvas.crop(
        (final_crop_left, final_crop_top, final_crop_left + w, final_crop_top + h)
    )

    pixels = list(region_background.getdata())
    white_pixel_ratio = pixels.count(255) / len(pixels)

    return region_background, white_pixel_ratio


if __name__ == "__main__":
    wsi_file_path = "data/TCGA_BRCA"
    wsi_name = "0a6b2b58-705e-43c4-a98e-618846dc696b"
    file_path = os.path.join(wsi_file_path, wsi_name)
    subfile_names = os.listdir(file_path)
    wsi_path = ""
    for filename in subfile_names:
        if filename.lower().endswith(('.svs', '.tiff')):
            wsi_path = os.path.join(file_path, filename)
    
    if wsi_path == "":
        print(f"Can not find the WSI file in {file_path}.")
    else:
        slide = OpenSlide(wsi_path)
        thumbnail, background, num_level = load_wsi_thumbnail(slide)
        print(type(thumbnail), type(background))    

        thumbnail.save("image/TCGA/thumbnail.png")
        background.save("image/TCGA/background.png")

        wsi_info = {
                "wsi_name":wsi_name,
                "position":(4000, 4000),    
                "level":1,
                "size":(2000, 2000),
            }
        
        ratio = slide.level_downsamples[-1] // slide.level_downsamples[1]

        print(ratio, slide.level_dimensions[1][0], slide.level_dimensions[1][1])

        ture_pos = (int(4000 * slide.level_downsamples[1]), int(4000 * slide.level_downsamples[1]))
        image = slide.read_region(ture_pos, 1, (2000, 2000))

        image = image.convert('RGB')
        image.save("image/TCGA/patch.png")

        patch_background, white_pixel_ratio = get_patch_background_ratio(slide, background, wsi_info)
        patch_background.save("image/TCGA/patch_background.png")
        print(f"Background Ratio: {white_pixel_ratio}")

        wsi_info["position"] = (5000, 5000)   
        wsi_info['angle'] = 45.13
        patch_background, white_pixel_ratio = get_region_background_ratio(slide, background, wsi_info)
        patch_background.save("image/TCGA/region_background.png")
        print(f"Background Ratio: {white_pixel_ratio}")