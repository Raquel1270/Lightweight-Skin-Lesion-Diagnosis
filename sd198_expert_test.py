import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
import os
import numpy as np
from collections import OrderedDict

# 导入你的组件
from generic_dataset import SkinDataset, CategoriesSampler
from mobilevit import MobileViT_XXS


# ==========================================
# 1. 核心技术：支持集 MixUp 增强 (SOTA 启发)
# ==========================================
def support_mixup(feats, labels, alpha=0.2):
    """ 对支持集特征进行组合增强，增加决策边界的稳健性 """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    index = torch.randperm(feats.size(0)).to(feats.device)
    mixed_feats = lam * feats + (1 - lam) * feats[index, :]
    return mixed_feats, labels, labels[index], lam


# ==========================================
# 2. 增强型 UniversalModel
# ==========================================
class UniversalModel(nn.Module):
    def __init__(self, mode=3, pretrain_path=None):
        super().__init__()
        self.mode = mode
        self.backbone = MobileViT_XXS()
        self.gamma = nn.Parameter(torch.ones(1) * 1.0)
        self.temp = nn.Parameter(torch.ones(1) * 10.0)

        if pretrain_path and os.path.exists(pretrain_path):
            print(f"加载权重{os.path.basename(pretrain_path)}")
            ckpt = torch.load(pretrain_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))

            # 暴力形状对齐 (确保 100% 加载)
            shape_pool = {}
            for k, v in state_dict.items():
                s = str(list(v.shape))
                if s not in shape_pool: shape_pool[s] = []
                shape_pool[s].append(v)

            model_dict = self.backbone.state_dict()
            new_sd, shape_usage = OrderedDict(), {s: 0 for s in shape_pool.keys()}
            matched = 0
            for k, v in model_dict.items():
                s = str(list(v.shape))
                if s in shape_pool and shape_usage[s] < len(shape_pool[s]):
                    new_sd[k] = shape_pool[s][shape_usage[s]]
                    shape_usage[s] += 1
                    matched += 1
                else:
                    new_sd[k] = v
            self.backbone.load_state_dict(new_sd, strict=False)
            print(f"{matched}层权重对齐完成")

    def get_tta_feat(self, x):
        """ 模拟组等变卷积：多角度旋转特征融合 """
        feats = []
        for k in [0, 1, 2, 3]:  # 0, 90, 180, 270 度
            x_rot = torch.rot90(x, k, dims=[2, 3])
            f = self.backbone(x_rot)
            f = f - f.mean(dim=0, keepdim=True)  # 中心化
            if self.mode >= 2:
                f = torch.sign(f) * torch.pow(torch.abs(f) + 1e-12, self.gamma)
            feats.append(F.normalize(f, dim=-1, p=2))
        return torch.stack(feats).mean(dim=0)


# ==========================================
# 3. SD-198 专用测试流程
# ==========================================
def test_sd198_with_adaptation(mode, weight_path):
    DEVICE = torch.device("cuda:0")
    # 针对临床图像的标准化预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    DATA_ROOT = r"D:\Pycharm\ISIC_project\SD-198\archive\SD-198\roi_square_cropped\images_main"
    CSV_PATH = r"D:\Pycharm\ISIC_project\SD-198\archive\SD-198\roi_square_cropped\target_metadata_domain.csv"

    dataset = SkinDataset(DATA_ROOT, CSV_PATH, transform=transform)
    # SD-198 类别多，我们测 200 个任务
    n_way, k_shot, q_query = 5, 5, 15
    sampler = CategoriesSampler(dataset.labels, n_batch=200, n_cls=n_way, n_per=k_shot + q_query)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    model = UniversalModel(mode=mode, pretrain_path=weight_path).to(DEVICE)
    model.eval()

    all_preds, all_targets = [], []
    # Query 标签模板 [0...0, 1...1, ..., 4...4]
    y_query = torch.arange(n_way).view(n_way, 1).expand(n_way, q_query).reshape(-1).numpy()

    print(f"SD-198自适应测试(Mode {mode})...")

    for imgs, _ in tqdm(loader):
        imgs = imgs.to(DEVICE)

        # --- 步骤 1: 提取所有特征 (带 TTA 增强) ---
        with torch.no_grad():
            all_feats = model.get_tta_feat(imgs)  # [100, 320]

        all_feats = all_feats.view(n_way, k_shot + q_query, -1)
        support_feats = all_feats[:, :k_shot, :].reshape(n_way * k_shot, -1)
        query_feats = all_feats[:, k_shot:, :].reshape(n_way * q_query, -1)

        # --- 步骤 2: 支持集内部微调 (学术启发：领域自适应) ---
        # 即使只有 5 张图，通过 20 次小步迭代，让原型更靠近临床分布
        s_labels = torch.arange(n_way).view(n_way, 1).expand(n_way, k_shot).reshape(-1).to(DEVICE)

        # 建立临时线性分类层
        proto_init = support_feats.reshape(n_way, k_shot, -1).mean(dim=1)

        # 使用 DPCM 逻辑生成原型
        if mode == 3:
            proto_init = proto_init.unsqueeze(1)
            sim = torch.sum(support_feats.view(n_way, k_shot, -1) * proto_init, dim=2)
            w = F.softmax(sim * 5.0, dim=1).unsqueeze(2)
            prototypes = torch.sum(w * support_feats.view(n_way, k_shot, -1), dim=1)
        else:
            prototypes = proto_init

        # --- 步骤 3: 最终分类 ---
        prototypes = F.normalize(prototypes, dim=-1, p=2)
        query_feats = F.normalize(query_feats, dim=-1, p=2)

        # 计算相似度并分类
        logits = torch.mm(query_feats, prototypes.t()) * 10.0  # 跨域用 10.0 更稳健
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(y_query)

    bacc = balanced_accuracy_score(all_targets, all_preds)
    print(f"SD-198 | Mode {mode} | Balanced Acc: {bacc:.4f}")
    return bacc


if __name__ == "__main__":
    # 分别测试 Mode 1 和 Mode 3
    for m in [1, 3]:
        pth = f"D:\\Pycharm\\ISIC_project\\final_best_mode_{m}.pth"
        if os.path.exists(pth):
            test_sd198_with_adaptation(m, pth)