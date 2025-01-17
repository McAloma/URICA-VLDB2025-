import sys
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
import requests, cv2
from io import BytesIO
from PIL import Image
import numpy as np



def load_wsi_thumbnail(wsi_name):
    wsi_url = f"http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/{wsi_name}"
    wsi_info_url = wsi_url.replace("region", "sliceInfo")

    try:
        slide_info = eval(requests.get(wsi_info_url).content)
    except:
        print(f"Can not find the information of {wsi_info_url}.")
        return 

    if slide_info == {'error': 'No such file'}:
        print(f"Can not find usrful slide_info :{wsi_info_url}")
        return
   
    try:
        num_level = int(slide_info["openslide.level-count"])
        print(num_level)
    except:
        print(f"None useful num_level in wsi {wsi_name}")
        return
    
    width = int(slide_info[f"openslide.level[{num_level-1}].width"])
    height = int(slide_info[f"openslide.level[{num_level-1}].height"])

    thumbnail_url = wsi_url + f"/0/0/{width}/{height}/{num_level-1}"
    print(f"Thumbnail: {thumbnail_url}")

    params = {'angle': '0'}
    response = requests.get(thumbnail_url, params=params)

    img = response.content
    img = Image.open(BytesIO(img)).convert("RGB")

    return img, num_level

def WSI_background_detect(wsi_image):
    """ 用 OTSU 来确定二值化阈值 """
    gray_image = wsi_image.convert("L")
    image_np = np.array(gray_image)
    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    _, binary_image = cv2.threshold(image_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image_pil = Image.fromarray(binary_image)

    return binary_image_pil

def get_patch_background_ratio(thumbnail, num_level, wsi_info):
    info_list = wsi_info[:-4].split("_")
    background = WSI_background_detect(thumbnail)

    ratio = 2 ** (int(num_level) - int(info_list[4]) - 1)

    x, y = int(info_list[0]) // ratio, int(info_list[1]) // ratio
    w, h = int(info_list[2]) // ratio, int(info_list[3]) // ratio

    patch_background = background.crop((x, y, x + w, y + h))
    pixels = list(patch_background.getdata())

    white_pixel_ratio = pixels.count(255) / len(pixels)
    return patch_background, white_pixel_ratio


if __name__ == "__main__":
    thumbnail, num_level = load_wsi_thumbnail("267136-33.tiff")
    background = WSI_background_detect(thumbnail)
    print(type(thumbnail), type(background))
    thumbnail.save("image/thumbnail.png")
    background.save("image/background.png")

    query_img_path =  "http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/267136-33.tiff/7000/7850/256/256/3"
    wsi_info =  "7000_7850_256_256_3.png"
    patch_background, white_pixel_ratio = get_patch_background_ratio(thumbnail, num_level, wsi_info)
    patch_background.save("image/patch_background.png")
    print(f"Background Ratio: {white_pixel_ratio}")