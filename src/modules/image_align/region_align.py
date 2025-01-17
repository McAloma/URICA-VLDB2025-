import cv2
import numpy as np
from PIL import Image


def pil_to_cv(image):
    return np.array(image)[:, :, ::-1]


def align_images_with_keypoints(image1, image2):
    image1_cv = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
    image2_cv = cv2.cvtColor(np.array(image2), cv2.COLOR_RGB2BGR)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1_cv, None)
    kp2, des2 = orb.detectAndCompute(image2_cv, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)     

    for m in matches:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))
        cv2.circle(image1_cv, pt1, 5, (0, 255, 0), -1)  
        cv2.circle(image2_cv, pt2, 5, (0, 255, 0), -1) 

    matched_indices1 = set([m.queryIdx for m in matches])
    matched_indices2 = set([m.trainIdx for m in matches])
    for i, kp in enumerate(kp1):
        if i not in matched_indices1:
            pt1 = tuple(map(int, kp.pt))
            cv2.circle(image1_cv, pt1, 5, (0, 0, 255), -1) 
    for i, kp in enumerate(kp2):
        if i not in matched_indices2:
            pt2 = tuple(map(int, kp.pt))
            cv2.circle(image2_cv, pt2, 5, (0, 0, 255), -1)  

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
    pre_x, pre_y = get_rotated_vec(cx-w//2, h//2-cy, angle)   

    x += pre_x
    y -= pre_y

    return x, y, target_location_info["size"][0], target_location_info["size"][1], pre_angle

