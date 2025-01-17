import sys, os, torch
sys.path.append("/hpc2hdd/home/rsu704/MDI_RAG_project/MDI_RAG_Image2Image_Research/")
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from src.model.load_train_data import Represent_Data_Loader



# NOTE: 为了适应 Patch 的 position shifting，更好的检索表征在具有语义性的基础上还需要定位性;
# 那么，假设相邻的patch之间的表征是连续呢，那么其法平面一定过原点.
# 所以，表征学习有两个关键点，一个是超平面的法向量指向原点（尽量平展）， 另一个是自回归（保持表征）
# STEP0: 本质上，模型只是一个表征模型，用FFN就好；
# STEP1: 中点向量和段点向量进行正交损失学习；（强目标项）
# STEP2: 加权的距离回归损失，用来维持相对位置。（弱目标项）（尽量满足三角不等式？）
# STEP3: 自回归损失，用来维持原始 feature 的表征力度；（正则项）

# NOTE: 从结果可视化上来看，表征虽然展现出很好的序列指标，但是可视化之后发现表征坍缩的很厉害，无法进行正常的表达，因此从全局上对表征进行统一编码的可用性不太高。


class Represent_Model(nn.Module):
    def __init__(self):
        super(Represent_Model, self).__init__()
        self.rep_param = nn.Parameter(torch.ones(1024))
        self.lens_param = nn.Parameter(torch.tensor(1.0))
    
    def represent(self, feature):
        feature = feature * self.rep_param
        return feature

    def forward(self, raw1, raw2):
        feature1 = self.represent(raw1)
        feature2 = self.represent(raw2)

        lens = (feature1 * feature2).sum(dim=1) * (torch.sigmoid(self.lens_param) + 1e-4)

        return feature1, feature2, lens
    

class TrainingRepresentModel():
    def __init__(self):
        self.best_loss = float('inf')

    def generalized_monotonicity_loss(self, prediction, distance, alpha=0.1):
        n = prediction.size(0)
        
        pred_diff_matrix = prediction.view(n, 1) - prediction.view(1, n)
        dist_diff_matrix = distance.view(n, 1) - distance.view(1, n)
        
        monotonic_loss = torch.relu(-(pred_diff_matrix * torch.sign(dist_diff_matrix))).mean()
        
        return alpha * monotonic_loss

    def loss_function(self, raw1, raw2, pos1, pos2, feature1, feature2, lens, alpha=1, beta=1):
        difference = feature1 - feature2
        sum_result = feature1 + feature2
        dot_product = (difference * sum_result).sum(dim=1)  
        OTH_loss = torch.mean(dot_product.abs())

        distances = torch.norm(pos1 - pos2, dim=1) / 224
        MSE_loss = self.generalized_monotonicity_loss(lens, distances)

        COS_loss1 = torch.mean(torch.norm(feature1 - raw1, dim=1) ** 2)
        COS_loss2 = torch.mean(torch.norm(feature2 - raw2, dim=1) ** 2)
        
        return OTH_loss, alpha * MSE_loss, beta * COS_loss1, beta * COS_loss2

    def train(self, model, dataloader, optimizer, epoch, device, alpha=1, beta=1):
        model.train()
        train_loss = 0

        loss_oth_sum = 0.0
        loss_mse_sum = 0.0
        loss_cos1_sum = 0.0
        loss_cos2_sum = 0.0

        total_batches = len(dataloader)
        
        with tqdm(dataloader, ascii=True) as pbar:
            for (raw1, raw2, pos1, pos2) in pbar:
                raw1, raw2, pos1, pos2 = raw1.to(device), raw2.to(device), pos1.to(device), pos2.to(device)
                optimizer.zero_grad()

                feature1, feature2, lens = model(raw1, raw2)

                OTH_loss, MSE_loss, COS_loss1, COS_loss2 = self.loss_function(raw1, raw2, pos1, pos2, feature1, feature2, lens, alpha, beta)

                loss = OTH_loss + MSE_loss + COS_loss1 + COS_loss2

                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                loss_oth_sum += OTH_loss.item()
                loss_mse_sum += MSE_loss.item()
                loss_cos1_sum += COS_loss1.item()
                loss_cos2_sum += COS_loss2.item()

                avg_loss_oth = loss_oth_sum / total_batches
                avg_loss_mse = loss_mse_sum / total_batches
                avg_loss_cos1 = loss_cos1_sum / total_batches
                avg_loss_cos2 = loss_cos2_sum / total_batches

                pbar.set_description(f"Epoch: {epoch} | Loss OTH: {avg_loss_oth:.4f} | Loss MSE: {avg_loss_mse:.4f} | Loss COS1: {avg_loss_cos1:.4f} | Loss COS2: {avg_loss_cos2:.4f}")    

        avg_epoch_loss = train_loss / total_batches
        if avg_epoch_loss < self.best_loss:
            self.best_loss = avg_epoch_loss
            torch.save(model.state_dict(), "ckpts/represent_model_shift.pth")



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Represent_Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    trainer = TrainingRepresentModel()
    data_loader = Represent_Data_Loader()

    target_dir = "data/metadata_embedding"
    specified_folder_name = os.listdir(target_dir)
    train_loader = data_loader.get_dataloader(n=4000, batch_size=4000, specified_folder_name=specified_folder_name)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        trainer.train(model, train_loader, optimizer, epoch, device, alpha=0.1, beta=0.01)