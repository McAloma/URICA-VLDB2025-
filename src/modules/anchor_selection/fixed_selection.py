import sys
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")

def get_anchor_triple(w, h):
    triple_w, triple_h = w // 3, h // 3
    anchor_positions = [
        [triple_w, triple_h], 
        [triple_w*2, triple_h], 
        [triple_w, triple_h*2], 
        [triple_w*2, triple_h*2], 
    ]

    return anchor_positions