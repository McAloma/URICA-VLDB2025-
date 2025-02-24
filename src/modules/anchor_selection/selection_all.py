def select_all_anchors(w, h, step=100):
    patch_w, patch_h = (224, 224)

    coords = []
    for y in range(0, h - patch_h, step):
        for x in range(0, w - patch_w, step):
            coords.append((x,y))