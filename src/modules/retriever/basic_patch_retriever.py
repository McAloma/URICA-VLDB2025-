import sys, qdrant_client, requests
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from PIL import Image
from io import BytesIO


class Image2Image_Retriever_Qdrant():
    def __init__(self, encoder, database_path=None):
        self.image_client_name = "WSI_Region_Retrieval"
        if database_path:
            database_path = database_path
        else:
            database_path = "/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/data/vector_database"  
            
        self.image_client = qdrant_client.QdrantClient(path=database_path)
        self.nums = self.image_client.count(collection_name=self.image_client_name)
        print("Number of vectors:", self.nums)

        self.image_encoder = encoder

    def retrieve(self, image, condition=None, top_k=20):
        query_embedding = self.image_encoder.encode_image(image)    # 1024

        retrieval_results = self.image_client.search(
            collection_name=self.image_client_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=condition,
        )

        return retrieval_results


if __name__ == "__main__":
    from src.utils.basic.encoder import WSI_Image_UNI_Encoder
    encoder = WSI_Image_UNI_Encoder()

    database_path = "data/vector_database_100"
    retriever = Image2Image_Retriever_Qdrant(encoder, database_path)

    query_img_path =  "http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/241183-21.tiff/6400/25344/256/256/1"
    if "http" in query_img_path:
        response = requests.get(query_img_path)
        query_image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        query_image = Image.open(query_img_path).convert("RGB")

    results = retriever.retrieve(query_image, top_k=20)
    for result in results:
        res = result.payload
        result_url = f"""http://mdi.hkust-gz.edu.cn/wsi/metaservice/api/region/openslide/{res["wsi_name"]}/{int(res["position"][0])}/{int(res["position"][1])}/{int(res["patch_size"][0])}/{int(res["patch_size"][1])}/{int(res["level"])}"""
        print(result_url)