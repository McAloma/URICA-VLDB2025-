import sys, os, json, torch, random
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed


# NOTE: 直接加载训练和测试的数据集，其中每一条内容是一个四元组，分别包括 patch1 patch2 的 UNI Embedding 以及他们的中心点位置。


class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        rep = self.data_list[idx]
        raw1, raw2, info1, info2 = rep

        raw1 = torch.tensor(raw1, dtype=torch.float32)
        raw2 = torch.tensor(raw2, dtype=torch.float32)

        info1_list = info1[:-4].split("_")
        pos1 = torch.tensor((int(info1_list[0]), int(info1_list[1])), dtype=torch.float32)
        info2_list = info2[:-4].split("_")
        pos2 = torch.tensor((int(info2_list[0]), int(info2_list[1])), dtype=torch.float32)

        return raw1, raw2, pos1, pos2
    

class Represent_Data_Loader():
    def load_embeddings_and_patch_info(self, subfolder_path, n):
        embedding_file = os.path.join(subfolder_path, 'embeddings.json')
        patch_info_file = os.path.join(subfolder_path, 'patch_info.json')
        
        with open(embedding_file, 'r') as ef:
            embeddings = json.load(ef)

        with open(patch_info_file, 'r') as pf:
            patch_info = json.load(pf)

        if len(embeddings) == 0:
            return []
        
        selected_indices1 = random.choices(range(len(embeddings)), k=n)
        selected_indices2 = random.choices(range(len(embeddings)), k=n)

        for i in range(n):
            if selected_indices1[i] == selected_indices2[i]:
                if selected_indices2[i] != len(embeddings)-1:
                    selected_indices2[i] += 1
                else:
                    selected_indices2[i] -= 1
        
        return [
            (embeddings[selected_indices1[idx]], embeddings[selected_indices2[idx]], 
                patch_info[selected_indices1[idx]], patch_info[selected_indices2[idx]]) 
            for idx in range(n)]

    def load_random_embeddings_and_patch_info(self, target_folder, specified_folders, n):
        all_embeddings_info = []
        futures = []

        with ThreadPoolExecutor() as executor:
            for folder_name in specified_folders:
                folder_path = os.path.join(target_folder, folder_name)
                
                if os.path.isdir(folder_path):
                    futures.append(executor.submit(self.load_embeddings_and_patch_info, folder_path, n))
                else:
                    print(f"指定的文件夹不存在: {folder_path}")

            for future in as_completed(futures):
                try:
                    all_embeddings_info.extend(future.result())
                except ValueError as e:
                    print(e)

        return all_embeddings_info
    
    def get_dataloader(self, n=1000, batch_size=1000, specified_folder_name=None):
        target_folder = "data/metadata_embedding"

        if not specified_folder_name:
            specified_folder_name = [
                "237208-14.tiff",
                "282540-14.tiff"
            ]

        all_embeddings_info = self.load_random_embeddings_and_patch_info(target_folder, specified_folder_name, n)

        train_dataset = ListDataset(all_embeddings_info)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        return train_loader

        

if __name__ == "__main__":
    loader = Represent_Data_Loader()
    
    train_loader = loader.get_dataloader()

    for info in train_loader:
        raw1, raw2, pos1, pos2 = info
        print(pos1, pos2)