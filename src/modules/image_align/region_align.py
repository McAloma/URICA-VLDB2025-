import sys, cv2
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
import numpy as np
from PIL import Image

from src.utils.metadata_wsi.wsi_loader import load_wsi_region


def pil_to_cv(image):
    return np.array(image)[:, :, ::-1]


def align_images_with_keypoints(image1, image2):
    image1_cv = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
    image2_cv = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

    # ORB 特征检测
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1_cv, None)
    kp2, des2 = orb.detectAndCompute(image2_cv, None)

    # 使用 BFMatcher 进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配的关键点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)        # 吧 src_pts 对应到 dst_pts 上面

    # 绘制匹配的关键点（绿色）
    for m in matches:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))
        cv2.circle(image1_cv, pt1, 5, (0, 255, 0), -1)  # 绿色圆圈
        cv2.circle(image2_cv, pt2, 5, (0, 255, 0), -1)  # 绿色圆圈

    # 绘制未匹配的关键点（红色）
    matched_indices1 = set([m.queryIdx for m in matches])
    matched_indices2 = set([m.trainIdx for m in matches])
    for i, kp in enumerate(kp1):
        if i not in matched_indices1:
            pt1 = tuple(map(int, kp.pt))
            cv2.circle(image1_cv, pt1, 5, (0, 0, 255), -1)  # 红色圆圈
    for i, kp in enumerate(kp2):
        if i not in matched_indices2:
            pt2 = tuple(map(int, kp.pt))
            cv2.circle(image2_cv, pt2, 5, (0, 0, 255), -1)  # 红色圆圈

    aligned_image2_cv = cv2.warpPerspective(image1_cv, M, (image2_cv.shape[1], image2_cv.shape[0]))

    image1_with_matches = Image.fromarray(cv2.cvtColor(image1_cv, cv2.COLOR_BGR2RGB))
    image2_with_matches = Image.fromarray(cv2.cvtColor(image2_cv, cv2.COLOR_BGR2RGB))
    aligned_image2 = Image.fromarray(cv2.cvtColor(aligned_image2_cv, cv2.COLOR_BGR2RGB))

    return image1_with_matches, image2_with_matches, aligned_image2, M


def get_image2_coordinates_on_image1(image1, image2, M):
    orig_w, orig_h = image1.size[:2]
    corners = np.array([[0, 0], [orig_w, 0], [orig_w, orig_h], [0, orig_h]], dtype=np.float32)
    
    transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)

    rect = cv2.minAreaRect(transformed_corners)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    angle = rect[-1]
    center_x, center_y = rect[0]

    return {"center": (center_x, center_y), "size": rect[1], "angle": angle, "box_points": box}

def get_rotated_vec(x1, x2, theta):
    theta_rad = np.radians(theta)
    
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                  [np.sin(theta_rad), np.cos(theta_rad)]])
    
    rotated_vector = np.dot(R, np.array([x1, x2]))
    
    return rotated_vector


def align_region(query_region, retrieval_region, retrieval_info):
    _, _, _, M = align_images_with_keypoints(query_region, retrieval_region)
    target_location_info = get_image2_coordinates_on_image1(query_region, retrieval_region, M)

    x, y, w, h, angle = retrieval_info

    cx, cy = target_location_info['center']
    pre_angle = angle - target_location_info['angle']
    pre_x, pre_y = get_rotated_vec(cx-w//2, h//2-cy, angle)     # 获得region2到中心点的位置后，重新对齐

    x += pre_x
    y -= pre_y

    return x, y, target_location_info["size"][0], target_location_info["size"][1], pre_angle



if __name__ == "__main__":
    # NOTE: 假设我们使用的 query 是 1，然而我们找到了 2，我们想通过 ORB 特征 来使用 2 能够获得 1 的信息，以此来获得更精确的结果。

    # 加载 region 1 图像
    wsi_name, x1, y1, w1, h1, level1, angle1 = "241183-21.tiff", 16351, 9332, 1203, 1620, 2, 15
    wsi_url1 = f"http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/{wsi_name}"
    _, region_image1 = load_wsi_region(wsi_url1, x1, y1, w1, h1, level1, angle1)
    region_image1.save("image/match/test_region1.png")

    # 加载 region 2 图像
    wsi_name, x2, y2, w2, h2, level2, angle2 = "241183-21.tiff", 16056, 9232, 1903, 2220, 2, 60
    # wsi_name, x2, y2, w2, h2, level2, angle2 = "241183-21.tiff", 16456, 9632, 1103, 1420, 2, 60
    wsi_url2 = f"http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/{wsi_name}"
    _, region_image2 = load_wsi_region(wsi_url2, x2, y2, w2, h2, level2, angle2)
    region_image2.save("image/match/test_region2.png")

    # 对齐图像
    image1_with_matches, image2_with_matches, aligned_image2, M = align_images_with_keypoints(region_image1, region_image2)
    image1_with_matches.save("image/match/horizon_region.png")
    image2_with_matches.save("image/match/rotated_region.png")
    aligned_image2.save("image/match/aligned_image2.png")

    target_location_info = get_image2_coordinates_on_image1(region_image1, region_image2, M)
    # print("Target location info:", target_location_info)
    
    # 可视化对齐后的区域——目标区域  (注意是旋转后的，region的平移规则随着角度变化)
    rot_x, rot_y = get_rotated_vec(x1-x2, -y1+y2, -angle2)
    rect = ((w2//2+rot_x - 1, h2//2-rot_y - 1), (w1, h1), angle2-angle1)    # 2对齐1
    box = cv2.boxPoints(rect)
    box = np.intp(box)  # 将坐标转换为整数
    image_with_box = cv2.drawContours(np.array(region_image2), [box], 0, (255, 0, 0), 4)    # 蓝色

    # 可视化对齐后的区域——结果区域
    image_with_box = cv2.drawContours(image_with_box, [target_location_info["box_points"]], 0, (0, 255, 0), 2)  # 绿色
    target_region_pil = Image.fromarray(image_with_box[:, :, ::-1])
    target_region_pil.save("image/match/align_result.png")

    # 直接测试
    retrieve_info = (x2, y2, w2, h2, angle2)
    pre_x, pre_y, pre_w, pre_h, pre_angle = align_region(region_image1, region_image2, retrieve_info)

    print(pre_x, pre_y, pre_w, pre_h, pre_angle)