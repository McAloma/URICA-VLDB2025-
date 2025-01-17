import sys, math
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
import numpy as np
from itertools import combinations
from collections import Counter, defaultdict

from src.utils.evaluator.cos_sim import cos_sim_list



def get_delta(target_vec, retrieve_vec):
    cos_theta, norm_target, norm_retri = cos_sim_list(target_vec, retrieve_vec)
    delta_angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    if np.cross(target_vec, retrieve_vec) < 0:
        delta_angle = - delta_angle
    
    delta_distance = norm_retri / norm_target

    return delta_distance, delta_angle


def find_min_variance_traversal(delta_list):
    all_combinations = []
    for r in range(2, len(delta_list) + 1):
        all_combinations.extend(combinations(delta_list, r))

    min_variance = float('inf')
    best_combination = None

    for combination in all_combinations:
        first_values = [item[0] for item in combination]
        second_values = [item[1] for item in combination]

        variance_sum = np.var(first_values) + np.var(second_values)

        if variance_sum < min_variance:
            min_variance = variance_sum
            best_combination = combination

    return best_combination, min_variance


def find_min_variance_BoC(delta_list, delta_lens=0.15, delta_angle=30):
    """ Bag of Change """
    bag_of_lens_1 = defaultdict(list)
    bag_of_lens_2 = defaultdict(list)

    bag_of_angle1 = defaultdict(list)
    bag_of_angle2 = defaultdict(list)
    
    for point in delta_list:
            lens, angle = point
            angle = 180 * angle / math.pi

            index_1 = str(lens // delta_lens)
            bag_of_lens_1[index_1].append(point)

            index_2 = str((lens - (delta_lens/2)) // delta_lens)
            bag_of_lens_2[index_2].append(point)

            index1 = str(angle // delta_angle)
            bag_of_angle1[index1].append(point)

            index2 = str(((angle-(delta_angle/2)) // delta_angle) % (360 / delta_angle))
            if index2 == "5.0" and angle < 0:     # 处理循环
                point = (point[0], point[1]+2*math.pi) 
            bag_of_angle2[index2].append(point)
    
    bag_of_lens_1 = {k: v for k, v in bag_of_lens_1.items() if len(v) > 1}
    bag_of_lens_2 = {k: v for k, v in bag_of_lens_2.items() if len(v) > 1}
    bag_of_angle1 = {k: v for k, v in bag_of_angle1.items() if len(v) > 1}
    bag_of_angle2 = {k: v for k, v in bag_of_angle2.items() if len(v) > 1}

    if not bag_of_lens_1 and not bag_of_lens_2:
        return 
    if bag_of_lens_1:
        min_key_dict1 = min(bag_of_lens_1, key=lambda k: (np.var([item[1] for item in bag_of_lens_1[k]]) / (len(bag_of_lens_1[k]))))    # min of var()/len()
        target1 = np.var([item[1] for item in bag_of_lens_1[min_key_dict1]]) / (len(bag_of_lens_1[min_key_dict1]))
    if not bag_of_lens_2:
        min_result_lens = bag_of_lens_1[min_key_dict1]
    if bag_of_lens_2:
        min_key_dict2 = min(bag_of_lens_2, key=lambda k: (np.var([item[1] for item in bag_of_lens_2[k]]) / (len(bag_of_lens_2[k]))))    # min of var()/len()
        target2 = np.var([item[1] for item in bag_of_lens_2[min_key_dict2]]) / (len(bag_of_lens_2[min_key_dict2]))
    if not bag_of_lens_1:
        min_result_lens = bag_of_lens_2[min_key_dict2]
    if bag_of_lens_1 and bag_of_lens_2:
        if target1 < target2:
            min_result_lens = bag_of_lens_1[min_key_dict1]
        else:
            min_result_lens = bag_of_lens_2[min_key_dict2]

    if not bag_of_angle1 and not bag_of_angle2:
        return 
    if bag_of_angle1:
        min_key_dict1 = min(bag_of_angle1, key=lambda k: (np.var([item[0] for item in bag_of_angle1[k]]) / (len(bag_of_angle1[k])))) 
        target1 = np.var([item[0] for item in bag_of_angle1[min_key_dict1]]) / (len(bag_of_angle1[min_key_dict1]))
    if not bag_of_angle2:
        min_result_angle = bag_of_angle1[min_key_dict1]
    if bag_of_angle2:
        min_key_dict2 = min(bag_of_angle2, key=lambda k: (np.var([item[0] for item in bag_of_angle2[k]]) / (len(bag_of_angle2[k]))))
        target2 = np.var([item[0] for item in bag_of_angle2[min_key_dict2]]) / (len(bag_of_angle2[min_key_dict2]))
    if not bag_of_angle1:
        min_result_angle = bag_of_angle2[min_key_dict2]
    if bag_of_angle1 and bag_of_angle2:
        if target1 < target2:
            min_result_angle = bag_of_angle1[min_key_dict1]
        else:
            min_result_angle = bag_of_angle2[min_key_dict2]

    best_combination = min_result_lens + min_result_angle

    return best_combination


def single_anchor_evaluation(query_region, target_pos, target_name, target_level, target_results, valid_results, mode="traversal"):
    region_candidate = []
    anchor_x, anchor_y = target_pos
    for res in target_results:
        res_pos = res.payload["position"]
        retrieve_x, retrieve_y = res_pos

        delta_list = []     # 保存检索结果的距离和角度变化
        for valid_pos in valid_results:
            valid_anchor_x, valid_anchor_y = valid_pos
            valid_retrieved_results = valid_results[valid_pos]
            for valid_anchor_res in valid_retrieved_results: 
                valid_x, valid_y = valid_anchor_res.payload["position"]

                retrieve_vec = (int(valid_x) - int(retrieve_x), int(retrieve_y) - int(valid_y))     # 检索结果构成的 pointer    
                target_vec = (valid_anchor_x - anchor_x, anchor_y - valid_anchor_y)                 # query region 中构成的 pointer

                delta_distance, delta_angle = get_delta(target_vec, retrieve_vec)
                if 0.75 < delta_distance < 1.5:
                    delta_list.append((delta_distance, delta_angle))
        
        if mode == "boc":
            combination = find_min_variance_BoC(delta_list) 
        else:
            combination, _ = find_min_variance_traversal(delta_list) 

        if combination and all(i is not None and i != [] for i in combination):     # 检查 combination 不为空
            width, height = query_region.size

            avg_distence = sum([i[0] for i in combination]) / len(combination)
            avg_angle = sum([i[1] for i in combination]) / len(combination)

            delta_x, delta_y = width // 2 - anchor_x, anchor_y - height // 2
            cos_theta, sin_theta = np.cos(avg_angle), np.sin(avg_angle)

            x_prime = int((delta_x * cos_theta - delta_y * sin_theta) * avg_distence)
            y_prime = int((delta_x * sin_theta + delta_y * cos_theta) * avg_distence)

            region = (target_name, 
                      int(retrieve_x)+x_prime, 
                      int(retrieve_y)-y_prime, 
                      int(width * avg_distence), 
                      int(height * avg_distence), 
                      target_level, 
                      avg_angle * 180 / math.pi)

            region_candidate.append(region)
    
    return region_candidate