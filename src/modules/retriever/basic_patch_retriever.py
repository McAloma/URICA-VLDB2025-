import qdrant_client


class Image2Image_Retriever_Qdrant():
    def __init__(self, encoder, database_path=None):
        self.image_client_name = "WSI_Region_Retrieval"
        if database_path:
            database_path = database_path
        else:
            database_path = "data/vector_database"  
            
        self.image_client = qdrant_client.QdrantClient(path=database_path)
        self.nums = self.image_client.count(collection_name=self.image_client_name)
        print("Number of vectors:", self.nums)

        self.image_encoder = encoder

    def retrieve(self, image, condition=None, top_k=20):
        query_embedding = self.image_encoder.encode_image(image)    

        retrieval_results = self.image_client.search(
            collection_name=self.image_client_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=condition,
        )

        return retrieval_results
