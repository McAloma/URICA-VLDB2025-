import sys, requests, os, math
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
import numpy as np
from io import BytesIO
from PIL import Image
from multiprocessing import shared_memory
from openslide import OpenSlide


class WSILoader():
    def wsi_loading_patch(self, wsi_path, position=(0,0), level=0, size=(256,256)):
        slide = OpenSlide(wsi_path)

        print(f"文件名: {wsi_path}")
        print(f"图像宽度: {slide.dimensions[0]} 像素")
        print(f"图像高度: {slide.dimensions[1]} 像素")
        print(f"级别数: {slide.level_count}")
        print(f"每级别的尺寸: {slide.level_dimensions}")
        print(f"每级别的降采样因子: {slide.level_downsamples}")

        ture_pos = tuple([int(pos*slide.level_downsamples[level]) for pos in position])
        image = slide.read_region(ture_pos, level, size)
        slide.close()
        image = image.convert('RGB')

        return image

    def wsi_loading_patches(self, wsi_path):
        """Get WSI infos and patches from WSI path"""
        slide = OpenSlide(wsi_path)
        levels, patch_size = slide.level_count, (256, 256)

        loaded_infos, loaded_images = [], []
        for level in range(1, slide.level_count):
            ratio = slide.level_downsamples[level]
            width, height = slide.level_dimensions[level][0], slide.level_dimensions[level][1]
            for w in range(0, width, patch_size[0]):
                for h in range(0, height, patch_size[0]):
                    ture_pos = (int(w * ratio), int(h * ratio))

                    infos = {
                        "wsi_name":wsi_path.split("/")[-1],
                        "position":ture_pos,    # basic on level 0
                        "level":level,
                        "size":patch_size,
                    }

                    image = slide.read_region(ture_pos, level, patch_size)
                    image = image.convert('RGB')
                    loaded_images.append(image)

        return loaded_infos, loaded_images 

    def get_wsi_shared_patches(self, wsi_path):
        """ loading patched image in share memory"""
        loaded_infos, loaded_images = self.wsi_loading_patches(wsi_path)
        img_arrays = np.array([np.array(img, dtype=np.uint8) for img in loaded_images])

        shm = shared_memory.SharedMemory(create=True, size=img_arrays.nbytes)
        shared_array = np.ndarray(img_arrays.shape, dtype=img_arrays.dtype, buffer=shm.buf)
        shared_array[:] = img_arrays[:]     # 将图像数据复制到共享内存

        return loaded_infos, shm, img_arrays.shape, img_arrays.dtype


def load_wsi_region(wsi_url, x, y, w, h, level, angle):
    """加载 region 的时候用的是中心点和角度值。"""
    x, y, w, h, level, angle = int(x), int(y), int(w), int(h), int(level), int(angle)
    length = max(w, h)
    background_url = f"{wsi_url}/{x-length}/{y-length}/{2*length}/{2*length}/{level}"

    response = requests.get(background_url)
    background = response.content
    background = Image.open(BytesIO(background)).convert("RGB")

    rotated_image = background.rotate(-angle, center=(length, length), expand=False)
    left = length - w / 2
    upper = length - h / 2
    right = length + w / 2
    lower = length + h / 2
    cropped_image = rotated_image.crop((left, upper, right, lower))
    
    return background, cropped_image

def load_region_tcga(region_info, wsi_file_path):
    wsi_dir = region_info["wsi_dir"]
    wsi_name = region_info["wsi_name"]
    wsi_path = os.path.join(wsi_file_path, wsi_dir, wsi_name)
    slide = OpenSlide(wsi_path)

    target_level = region_info["level"]
    ratio = slide.level_downsamples[target_level]

    w, h = int(region_info["size"][0]), int(region_info["size"][1])
    canvas_size = int(math.sqrt(w**2 + h**2))  # square canvas
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))

    # true position with center point
    x = int((region_info["position"][0] - canvas_size // 2) * ratio)
    y = int((region_info["position"][1] - canvas_size // 2) * ratio)
    angle = region_info["angle"]

    psuedo_region = slide.read_region((x, y), target_level, (canvas_size, canvas_size)) 
    canvas.paste(psuedo_region, (0, 0))
    rotated_canvas = canvas.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=False)

    final_crop_left = (rotated_canvas.width - w) // 2
    final_crop_top = (rotated_canvas.height - h) // 2
    region = rotated_canvas.crop(
        (final_crop_left, final_crop_top, final_crop_left + w, final_crop_top + h)
    )

    return region

def load_region_came(region_info, wsis_path):
    wsi_name = region_info['wsi_name']
    wsi_path = os.path.join(wsis_path, wsi_name)
    slide = OpenSlide(wsi_path)

    target_level = region_info["level"]
    ratio = slide.level_downsamples[target_level]

    w, h = int(region_info["size"][0]), int(region_info["size"][1])
    canvas_size = int(math.sqrt(w**2 + h**2))  # square canvas
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))

    # true position with center point
    x = int((region_info["position"][0] - canvas_size // 2) * ratio)
    y = int((region_info["position"][1] - canvas_size // 2) * ratio)
    angle = region_info["angle"]

    psuedo_region = slide.read_region((x, y), target_level, (canvas_size, canvas_size)) 
    canvas.paste(psuedo_region, (0, 0))
    rotated_canvas = canvas.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=False)

    final_crop_left = (rotated_canvas.width - w) // 2
    final_crop_top = (rotated_canvas.height - h) // 2
    region = rotated_canvas.crop(
        (final_crop_left, final_crop_top, final_crop_left + w, final_crop_top + h)
    )

    return region



if __name__ == "__main__":
    wsis_path = "data/Camelyon"
    wsi_names = [
        "tumor_001.tif",
        "tumor_002.tif",
        "tumor_003.tif",
        "tumor_004.tif",
        "tumor_005.tif",
        "tumor_006.tif",
        "tumor_007.tif",
        "tumor_008.tif",
        "tumor_009.tif",
        "tumor_010.tif",
        "tumor_011.tif",
        "tumor_012.tif",
        "tumor_013.tif",
        "tumor_014.tif",
        "tumor_015.tif",
        "tumor_016.tif",
        "tumor_017.tif",
        "tumor_018.tif",
        "tumor_019.tif",
        "tumor_020.tif",
    ]
    for name in wsi_names:
        infos = {
            "wsi_name":name,
            "position":(1000, 1000),    # basic on current level (左上角)
            "level":2,
            "size":(1000, 1000),
            "angle":0,
        }

        try:
            region = load_region_came(infos, wsis_path)
            print(f"Read region: {region}")
        except:
            print(f"Wrong WSI: {name}")