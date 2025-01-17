import sys, os, cv2, time, torch
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from tqdm import tqdm
from PIL import Image
from openslide import OpenSlide
import numpy as np

import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class ImageAlignRetriever:
    def __init__(self, query_image, folder_path, encoder):
        self.query_image = query_image
        self.query_image_cv2 = cv2.cvtColor(np.array(query_image), cv2.COLOR_RGB2GRAY)
        self.folder_path = folder_path
        self.encoder = encoder
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.query_keypoints, self.query_descriptors = self.orb.detectAndCompute(self.query_image_cv2, None)

    def align_images(self):
        match_results = []

        for filename in os.listdir(self.folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_path = os.path.join(self.folder_path, filename)
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 以彩色模式加载图像
                keypoints, descriptors = self.orb.detectAndCompute(image, None)
                matches = self.bf.match(self.query_descriptors, descriptors)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > 4:  # 至少需要4个点进行单应性计算
                    src_pts = np.float32([self.query_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if M is not None:
                        h, w = self.query_image.size[:2]  # 确保宽高来自彩色图像
                        aligned_image_cv2 = cv2.warpPerspective(image, M, (w, h))
                        
                        # 转换为 PIL.Image 格式，调整通道顺序
                        aligned_image = Image.fromarray(cv2.cvtColor(aligned_image_cv2, cv2.COLOR_BGR2RGB))
                        
                        match_results.append((filename, aligned_image, M, matches))

        # 按照特征匹配总距离排序，选择Top-5
        match_results = sorted(match_results, key=lambda x: sum([m.distance for m in x[3]]))[:5]
        return match_results

    def calculate_iou(self, rect1, rect2):
        """计算两个矩形框的IoU值"""
        x1, y1 = max(rect1[0][0], rect2[0][0]), max(rect1[0][1], rect2[0][1])
        x2, y2 = min(rect1[1][0], rect2[1][0]), min(rect1[1][1], rect2[1][1])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (rect1[1][0] - rect1[0][0]) * (rect1[1][1] - rect1[0][1])
        area2 = (rect2[1][0] - rect2[0][0]) * (rect2[1][1] - rect2[0][1])
        union = area1 + area2 - intersection
        return intersection / union

    def calculate_similarity(self, encoded_query, encoded_target):
        """计算编码后的余弦相似度"""
        return cosine_similarity([encoded_query], [encoded_target])[0, 0]

    def extract_rotated_rect_params(self, corners):
        """从角点计算旋转矩形的参数 (x, y, w, h, theta)"""
        rect = cv2.minAreaRect(corners)
        center = rect[0]
        size = rect[1]
        angle = rect[2]
        if size[0] < size[1]:  # 保证角度范围为 [-90, 90]
            angle += 90
        return center[0], center[1], size[0], size[1], angle

    def get_top_5_matches(self):
        match_results = self.align_images()
        query_encoded = self.encoder.encode_image(self.query_image)  # 使用自定义编码器编码查询图像
        results = []

        for filename, aligned_image, M, matches in match_results:
            # 计算IoU
            h, w = self.query_image_cv2.shape
            query_rect = [[0, 0], [w - 1, h - 1]]
            dst_rect = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            aligned_corners = cv2.perspectiveTransform(dst_rect, M).reshape(-1, 2)
            aligned_rect = [[min(aligned_corners[:, 0]), min(aligned_corners[:, 1])],
                            [max(aligned_corners[:, 0]), max(aligned_corners[:, 1])]]
            iou = self.calculate_iou(query_rect, aligned_rect)

            # 计算旋转矩形参数
            x, y, w, h, theta = self.extract_rotated_rect_params(aligned_corners)

            # 计算余弦相似度
            encoded_aligned = self.encoder.encode_image(aligned_image)
            cosine_sim = self.calculate_similarity(query_encoded, encoded_aligned)

            results.append((filename, x, y, w, h, theta, iou, cosine_sim))

        # 根据IoU和余弦相似度排序
        results = sorted(results, key=lambda x: (x[6], x[7]), reverse=True)
        return results



    

def get_region_image(base_path, query_wsi_name, x, y, w, h, level, angle):
    wsi_folder_path = os.path.join(base_path, query_wsi_name)
    svs_file = None

    for filename in os.listdir(wsi_folder_path):
        if filename.lower().endswith('.svs'):
            svs_file = os.path.join(wsi_folder_path, filename)
            break

    if not svs_file:
        raise FileNotFoundError(f"No .svs file found in {wsi_folder_path}")

    slide = OpenSlide(svs_file)
    region = slide.read_region((0, 0), level, slide.level_dimensions[level])
    region = region.convert('RGB')

    if angle != 0:
        region = region.rotate(angle, expand=False)

    region = region.crop((x, y, x + w, y + h))

    return region


if __name__ == "__main__":
    base_path = "data/TCGA_BRCA"
    query_wsi_name = "0bd7ca22-a281-427c-a5ef-b59b5dd02da3"
    x, y, w, h, level, angle = 15121, 9523, 1000, 1000, 1, 85
    query_image = get_region_image(base_path, query_wsi_name, x, y, w, h, level, angle)

    from src.utils.basic.encoder import WSI_Image_UNI_Encoder, WSI_Image_test_Encoder
    # encoder = WSI_Image_UNI_Encoder()
    encoder = WSI_Image_test_Encoder()        # for test


    folder_path = "data/thumbnail_cache_sample"
    retriever = ImageAlignRetriever(query_image, folder_path, encoder)
    
    start_time = time.time()
    results = retriever.get_top_5_matches()
    end_time = time.time()

    print(results)      
    # ('0a9ea7ac-9d51-4ff7-b40b-659a57e64945.png', 383.2445373535156, 2263.652099609375, 45.100433349609375, 5.124899864196777, 79.0450668334961, 0.0, 1.0)
    print(f"Retrieval time: {end_time - start_time} seconds")