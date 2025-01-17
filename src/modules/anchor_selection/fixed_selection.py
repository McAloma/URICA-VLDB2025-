def get_anchor_triple(w, h):
    triple_w, triple_h = w // 3, h // 3
    anchor_positions = [
        [triple_w, triple_h], 
        [triple_w*2, triple_h], 
        [triple_w, triple_h*2], 
        [triple_w*2, triple_h*2], 
    ]

    return anchor_positions