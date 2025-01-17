import time
from collections import deque, defaultdict
from src.modules.retriever.basic_patch_retriever import Image2Image_Retriever_Qdrant



class Adjacent_Region_Retriever():
    def __init__(self, basic_retriever, encoder):
        self.retriever = basic_retriever
        self.encoder = encoder

    def mesh_slides(self, image):
        width, height = image.size
        width_step = 224 - (224 - width % 224) // (width // 224)
        height_step = 224 - (224 - height % 224) // (height // 224)

        image_patches = []
        for x in range(0, width-223, width_step):
            for y in range(0, height-223, height_step):
                cropped_image = image.crop((x, y, x+224, y+224))
                image_patches.append(cropped_image)

        return image_patches
    
    def single_retrieval(self, query_image):
        results = self.retriever.retrieve(query_image, top_k=20)
        return [(result.score, result.payload) for result in results] 
    
    def mesh_slides(self, image):
        width, height = image.size
        width_step = 224 - (224 - width % 224) // (width // 224)
        height_step = 224 - (224 - height % 224) // (height // 224)

        image_patches = []
        for x in range(0, max(1, width-223), width_step):
            for y in range(0, max(1, height-223), height_step):
                cropped_image = image.crop((x, y, x+224, y+224))
                image_patches.append(cropped_image)

        return image_patches
    
    def find_most_wsi_name(self, raw_results, top_n=5):
        score_hist = defaultdict(float)
        result_hist = defaultdict(list)
        
        for result in raw_results:
            for score, payload in result:
                wsi_name = payload["wsi_name"]
                score_hist[wsi_name] += score
                result_hist[wsi_name].append((score, payload))
        
        top_targets = sorted(score_hist.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        most_target = top_targets[0][0]
        return most_target, result_hist[most_target], top_targets
    
    def find_region(self, target_results):
        def is_adjacent(rect1, rect2):
            x1, y1, w1, h1 = rect1
            x2, y2, w2, h2 = rect2

            if x1 + w1 == x2 and y1 < y2 + h2 and y2 < y1 + h1:
                return True
            elif x2 + w2 == x1 and y1 < y2 + h2 and y2 < y1 + h1:
                return True
            elif y1 + h1 == y2 and x1 < x2 + w2 and x2 < x1 + w1:
                return True
            elif y2 + h2 == y1 and x1 < x2 + w2 and x2 < x1 + w1:
                return True
            else:
                return False
        
        rect_list = [
            (result[0], [
                int(result[1]['position'][0]) * (2 ** int(result[1]['level'])), 
                int(result[1]['position'][1]) * (2 ** int(result[1]['level'])), 
                int(result[1]['patch_size'][0]) * (2 ** int(result[1]['level'])),  
                int(result[1]['patch_size'][1]) * (2 ** int(result[1]['level'])), 
              ])
            for result in target_results
        ] 
            
        score_results = defaultdict(list)
        region_results = defaultdict(list)

        checked_index = []
        target_deque = deque()  
        for i in range(len(rect_list)):
            if i in checked_index:
                continue
            target_deque.append([i, rect_list[i]])
            checked_index.append(i)

            while len(target_deque) != 0:
                index, cur = target_deque.popleft()
                score1, rect1 = cur[0], cur[1]
                
                score_results[i].append(score1)
                region_results[i].append(rect1)

                for j in range(len(rect_list)):
                    if j in checked_index:
                        continue
                
                    _, rect2 = rect_list[j]
                    if is_adjacent(rect1, rect2):
                        target_deque.append([j, rect_list[j]])
                        checked_index.append(j)

        regions = []
        for key in region_results:
            cur_patches = region_results[key]
    
            result_x = min([res[0] for res in cur_patches])
            result_y = min([res[1] for res in cur_patches])
            result_w = max([res[0]+res[2] for res in cur_patches]) - result_x
            result_h = max([res[1]+res[3] for res in cur_patches]) - result_y

            target_region = [result_x, result_y, result_w, result_h]
            regions.append(target_region)

        return regions

    def redifine_region(self, target_region, ratio):
        x, y, width, height = target_region
        mid_x = x + width // 2
        mid_y = y + height // 2

        redifine_width = int((width * height * ratio) ** 0.5)   # ratio = width / heigh
        redifine_height = int(redifine_width / ratio)
        redifine_x = max(0, mid_x - redifine_width // 2)
        redifine_y = max(0, mid_y - redifine_height // 2)

        return [redifine_x, redifine_y, redifine_width, redifine_height, 0, 0]     # with level 0 and angle 0

    def retrieve(self, image):
        width, height = image.size
        image_patches = self.mesh_slides(image)
        raw_results = [self.single_retrieval(patch) for patch in image_patches]

        target_wsi_name, target_results, top_targets = self.find_most_wsi_name(raw_results)   
        region_results = self.find_region(target_results)
        
        redifine_region = [self.redifine_region(region, width/height) for region in region_results]

        return target_wsi_name, redifine_region, top_targets

    

if __name__ == "__main__":
    from src.utils.open_wsi.wsi_loader import load_region_tcga
    base_path = "data/TCGA_BRCA"
    query_info = {
        "position": [
            2010,
            1089
        ],
        "size": [
            1098,
            600
        ],
        "level": 2,
        "angle": -114,
        "wsi_dir": "03dde19f-f6f0-4b6b-a559-febbf532ca76",
        "wsi_name": "TCGA-E9-A1NC-01Z-00-DX1.20edf036-8ba6-4187-a74c-124fc39f5aa1.svs"
    }
    query_region = load_region_tcga(query_info, wsi_file_path="data/TCGA_BRCA")

    from src.utils.basic.encoder import WSI_Image_UNI_Encoder, WSI_Image_test_Encoder
    encoder = WSI_Image_test_Encoder()
    database_path = "data/vector_database_TCGA_sample"
    basic_retriever = Image2Image_Retriever_Qdrant(encoder, database_path)

    retriever = Adjacent_Region_Retriever(basic_retriever, encoder)

    start = time.time()
    wsi_name, region_candidate = retriever.retrieve(query_region)
    end = time.time()

    print(f"Total Retrieval Time: {end-start}")
    
    print(wsi_name)
    for index, (region) in enumerate(region_candidate):     
        print(f"Res: {region}.")  