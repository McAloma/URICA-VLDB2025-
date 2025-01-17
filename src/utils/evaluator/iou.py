import sys, cv2
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")


def region_retrieval_self_IoU(query_param, region_candidate):
        q_name, q_x, q_y, q_w, q_h, q_level, q_angle = query_param
        rect_query = ((q_x, q_y), (q_w, q_h), q_angle)

        res = []
        for region, _ in region_candidate:
            r_name, r_x, r_y, r_w, r_h, r_level, r_angle = region
            if q_name != r_name or str(q_level) != str(r_level):
                res.append(0)
            else:
                rect_region = ((r_x, r_y), (r_w, r_h), r_angle)
                inter_type, inter_pts = cv2.rotatedRectangleIntersection(rect_query, rect_region)

                if inter_type > 0 and inter_pts is not None:
                    inter_area = cv2.contourArea(inter_pts)
                else:
                    inter_area = 0.0

                area1 = q_w * q_h
                area2 = r_w * r_h
                union_area = area1 + area2 - inter_area
                iou_score = inter_area / union_area

                res.append(iou_score)

        if len(res) == 0:
            return 0
        return sum(res) / len(res)