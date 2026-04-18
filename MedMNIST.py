import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode  # 新增：高级插值
import medmnist
from medmnist import INFO
# ========== 新增：导入AUC计算所需库 ==========
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import numpy as np
import os
from collections import OrderedDict

from mobilevit import MobileViT_XXS


# ==========================================
# 1. 优化后的数据适配器（无修改，保留原逻辑）
# ==========================================
class MedMNISTFewShot(Dataset):
    def __init__(self, split, flag, transform=None):
        self.flag = flag
        self.info = INFO[self.flag]
        DataClass = getattr(medmnist, self.info['python_class'])
        self.dataset = DataClass(split=split, download=True)
        self.transform = transform
        self.labels = self.dataset.labels.flatten()

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        # 修正 NumPy 警告：使用 .item()
        return img, int(label.item() if hasattr(label, 'item') else label)


class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        label = torch.LongTensor(label)
        self.m_ind = []
        for i in range(int(label.max()) + 1):
            ind = torch.argwhere(label == i).reshape(-1)
            if len(ind) >= n_per:
                self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            yield torch.stack(batch).reshape(-1)


# ==========================================
# 2. 强力模型逻辑 (加入 TTA 思想)（无修改，保留原优化）
# ==========================================
class UniversalModel(nn.Module):
    def __init__(self, mode=3, pretrain_path=None):
        super().__init__()
        self.mode = mode
        self.backbone = MobileViT_XXS()
        self.gamma = nn.Parameter(torch.ones(1) * 1.0)
        self.temp = nn.Parameter(torch.ones(1) * 10.0)

        if pretrain_path and os.path.exists(pretrain_path):
            print(f"全序对齐: {os.path.basename(pretrain_path)}")
            ckpt = torch.load(pretrain_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))
            shape_pool = {}
            for k, v in state_dict.items():
                if 'classifier' in k: continue
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
            print(f"对齐 {matched} 层")

    def meta_forward(self, x, n_way, k_shot, q_query):
        # 提取特征
        feat = self.backbone(x)

        # 针对 MedMNIST 这种小图放大后的特殊处理：增强对比度
        feat = feat - feat.mean(dim=0, keepdim=True)

        if self.mode >= 2:
            # 高斯化修正：在极低分辨率下，幂变换参数不宜过小
            safe_gamma = torch.clamp(self.gamma, 0.5, 1.0)
            feat = torch.sign(feat) * torch.pow(torch.abs(feat) + 1e-12, safe_gamma)

        features = F.normalize(feat, dim=-1, p=2)
        z = features.view(n_way, k_shot + q_query, -1)
        support, query = z[:, :k_shot, :], z[:, k_shot:, :]

        if self.mode == 3:
            proto_init = support.mean(dim=1).unsqueeze(1)
            sim = torch.sum(support * proto_init, dim=2)
            # 使用更强的温度系数来对抗模糊
            weights = F.softmax(sim * 10.0, dim=1).unsqueeze(2)
            prototypes = torch.sum(weights * support, dim=1)
        else:
            prototypes = support.mean(dim=1)

        prototypes = F.normalize(prototypes, dim=-1, p=2)
        # 针对弱特征，将 scale 从 10 提升到 25.0
        logits = torch.mm(query.reshape(-1, query.size(-1)), prototypes.t()) * 25.0
        return logits


# ==========================================
# 3. 运行验证（核心修改：添加AUC计算，修复标签不匹配）
# ==========================================
def run_medmnist_test(flag, pretrain, mode):
    # ========== 改动1：完善设备判断，增加CPU兜底 ==========
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 【核心改进 A】：使用 Bicubic 插值减少马赛克感
    # 【核心改进 B】：使用训练时一致的归一化参数
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = MedMNISTFewShot(split='test', flag=flag, transform=transform)
    n_way, k_shot, q_query = 5, 5, 15
    sampler = CategoriesSampler(dataset.labels, n_batch=300, n_cls=n_way, n_per=k_shot + q_query)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    model = UniversalModel(mode=mode, pretrain_path=pretrain).to(DEVICE)
    model.eval()

    # ========== 改动2：新增存储概率的列表，用于AUC计算 ==========
    all_preds, all_targets, all_probs = [], [], []

    print(f"正在测试MedMNIST: {flag} | Mode: {mode}")
    with torch.no_grad():
        # ========== 改动3：从loader中获取真实labels，替代固定标签 ==========
        for imgs, raw_labels in tqdm(loader):
            imgs = imgs.to(DEVICE)
            # 步骤1：提取当前batch的5个唯一采样类别（任务内真实类别）
            batch_unique_cls = np.unique(raw_labels.numpy())
            # 步骤2：分离support和query的原始标签，仅保留query标签（用于计算指标）
            # raw_labels形状：[5*(1+15)] = 80，前5*1=5个是support，后5*15=75个是query
            query_raw_labels = raw_labels[k_shot * n_way:].numpy()
            # 步骤3：将原始标签映射为任务内0-4的标签（适配5-way少样本任务）
            task_targets = np.zeros_like(query_raw_labels)
            for idx, cls in enumerate(batch_unique_cls):
                task_targets[query_raw_labels == cls] = idx

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model.meta_forward(imgs, n_way, k_shot, q_query)  # [75,5]

            # ========== 改动4：计算预测概率（AUC必须，softmax归一化） ==========
            pred_probs = F.softmax(logits, dim=1).cpu().float().numpy()
            # 计算预测类别（用于准确率）
            pred_labels = logits.argmax(1).cpu().numpy()

            # ========== 改动5：收集所有结果（标签/预测/概率） ==========
            all_preds.extend(pred_labels)
            all_targets.extend(task_targets)
            all_probs.extend(pred_probs)

    # ========== 改动6：计算平衡准确率（原指标，保留） ==========
    bacc = balanced_accuracy_score(all_targets, all_preds)

    # ========== 改动7：计算多分类AUC-ROC（One-vs-Rest策略，SCI标准） ==========
    # 步骤1：对标签进行二值化（多分类AUC计算的必要操作）
    targets_binarized = label_binarize(all_targets, classes=np.arange(n_way))
    # 步骤2：计算AUC，异常捕获避免个别batch报错
    try:
        auc = roc_auc_score(
            targets_binarized,
            np.array(all_probs),
            multi_class='ovr',  # 多分类必选：一对一策略
            average='macro'     # 宏平均，对5个类别平等加权
        )
    except Exception as e:
        print(f"警告：AUC计算出现异常 {e}，赋值为0.5")
        auc = 0.5

    # ========== 改动8：打印AUC结果，返回双指标 ==========
    print(f"{flag} | Mode {mode} | Balanced Acc = {bacc:.4f} | AUC-ROC = {auc:.4f}")
    # 释放显存，避免内存泄漏
    del model, loader
    torch.cuda.empty_cache()
    return bacc, auc  # 返回准确率和AUC，方便后续绘图/记录


if __name__ == "__main__":
    BASE_PATH = r"D:\Pycharm\ISIC_project\final_best_mode_1.pth"
    OURS_PATH = r"D:\Pycharm\ISIC_project\final_best_mode_3.pth"

    # 建议先跑这三个，特别是 pathmnist，因为它跟皮肤病特征完全不同，最能测泛化
    TASKS = ['dermamnist', 'pathmnist', 'bloodmnist']

    # ========== 改动9：接收返回的AUC值，适配双指标输出 ==========
    for flag in TASKS:
        for m, pth in [(1, BASE_PATH), (3, OURS_PATH)]:
            if os.path.exists(pth):
                bacc, auc = run_medmnist_test(flag, pth, m)
            else:
                print(f"权重文件不存在：{pth}")