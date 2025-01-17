import os, torch, timm
from tqdm import tqdm
from torchvision import transforms


class WSI_Image_UNI_Encoder():
    def __init__(self, param_local_dir=None):
        self.embed_model =  timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        local_dir = "ckpts/vit_large_patch16_224.dinov2.uni_mass100k/"
        if param_local_dir:
            local_dir = param_local_dir
        self._device = self.infer_torch_device()
        self.embed_model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu", weights_only=True), strict=True)
        self.embed_model = self.embed_model.to(self._device)
        self.embed_model.eval()
        print("Loaded UNI Encoder.")

    def infer_torch_device(self):
        """Infer the input to torch.device."""
        try:
            has_cuda = torch.cuda.is_available()
        except NameError:
            import torch  # pants: no-infer-dep
            has_cuda = torch.cuda.is_available()
        if has_cuda:
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def encode_image(self, patch_image):
        patch_image = self.transform(patch_image).unsqueeze(dim=0).to(self._device)
        embedding = self.embed_model(patch_image)

        return embedding.cpu().squeeze().tolist()

    def encode_wsi_patch(self, wsi_name, dataloader, show=False):
        embeddings = []
        with torch.no_grad():
            if show:
                for images in tqdm(dataloader, desc=f"WSI name: {wsi_name}", ascii=True):
                    images = images.to(self._device)
                    embedding = self.embed_model(images)
                    embeddings.append(embedding.cpu())
            else:
                for images in dataloader:
                    images = images.to(self._device)
                    embedding = self.embed_model(images)
                    embeddings.append(embedding.cpu())

        return embeddings


class WSI_Image_test_Encoder():
    def __init__(self):
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        print("Loaded test Encoder.")

    def encode_image(self, image):
        embedding = [1.0 for _ in range(1024)]

        return embedding
    
    def encode_wsi_patch(self, wsi_name, dataloader, show=False):
        embeddings = []
        embedding_size = 1024

        with torch.no_grad():
            if show:
                for images in tqdm(dataloader, desc=f"WSI name: {wsi_name}", ascii=True):
                    batch_size = images.size(0) 
                    embedding = torch.tensor([ [1.0] * embedding_size for _ in range(batch_size)])
                    embeddings.append(embedding)
            else:
                for images in dataloader:
                    batch_size = images.size(0)
                    embedding = torch.tensor([ [1.0] * embedding_size for _ in range(batch_size)]) 
                    embeddings.append(embedding) 

        return embeddings