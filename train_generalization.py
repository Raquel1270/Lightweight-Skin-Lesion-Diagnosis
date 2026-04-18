import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm import tqdm
import os
from collections import OrderedDict

# 确保 generic_dataset.py 和 mobilevit.py 在同级目录
from generic_dataset import SkinDataset, CategoriesSampler
from mobilevit import MobileViT_XXS



class UniversalModel(nn.Module):
    def __init__(self, mode=3, pretrain_path=None):
        super().__init__()
        self.mode = mode
        self.backbone = MobileViT_XXS()
        self.gamma = nn.Parameter(torch.ones(1) * 1.0)
        self.temp = nn.Parameter(torch.ones(1) * 5.0)

        if pretrain_path and os.path.exists(pretrain_path):
            print(f"正在解析: {os.path.basename(pretrain_path)}")
            # 始终先加载到 CPU，防止显存初次分配过大
            ckpt = torch.load(pretrain_path, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt.get('model', ckpt))

            # --- 核心黑科技：不看名字，只看形状对齐 ---
            # 1. 提取权重文件中所有张量的形状和数据
            shape_pool = OrderedDict()
            for k, v in state_dict.items():
                if 'classifier' in k: continue  # 跳过分类头
                s = str(list(v.shape))
                if s not in shape_pool: shape_pool[s] = []
                shape_pool[s].append(v)

            # 2. 遍历当前模型，按顺序从池子里取形状一样的张量
            model_dict = self.backbone.state_dict()
            new_sd = OrderedDict()
            matched = 0
            shape_usage = {s: 0 for s in shape_pool.keys()}

            for k, v in model_dict.items():
                s = str(list(v.shape))
                if s in shape_pool and shape_usage[s] < len(shape_pool[s]):
                    new_sd[k] = shape_pool[s][shape_usage[s]]
                    shape_usage[s] += 1
                    matched += 1
                else:
                    new_sd[k] = v  # 匹配不上则保留原样（随机初始化）

            # 3. 强行注入
            self.backbone.load_state_dict(new_sd, strict=False)
            print(f"成功注入 {matched} / {len(model_dict)} 层参数")

            if matched > 280:
                print("骨干网络已全量加载预训练")
            else:
                print("匹配层数不足")

    def meta_forward(self, x, n_way, k_shot, q_query):
        # 提取特征
        feat = self.backbone(x)  # [N, 320]

         #1. 任务内中心化 (泛化性测试的关键)
        feat = feat - feat.mean(dim=0, keepdim=True)

         #2. 特征高斯化
        if self.mode >= 2:
            feat = torch.sign(feat) * torch.pow(torch.abs(feat) + 1e-12, self.gamma)

        features = F.normalize(feat, dim=-1, p=2)

        #3. 拆分 Support & Query
        z = features.view(n_way, k_shot + q_query, -1)
        support = z[:, :k_shot, :]
        query = z[:, k_shot:, :]

        if self.mode == 3:
            # DPCM 动态原型校准
            proto_init = support.mean(dim=1).unsqueeze(1)
            sim = torch.sum(support * proto_init, dim=2)
            weights = F.softmax(sim * self.temp, dim=1).unsqueeze(2)
            prototypes = torch.sum(weights * support, dim=1)
        else:
            prototypes = support.mean(dim=1)

        prototypes = F.normalize(prototypes, dim=-1, p=2)
        query_flat = query.reshape(-1, query.size(-1))

        # 4. 余弦相似度分类
        logits = torch.mm(query_flat, prototypes.t()) * 5.0
        return logits


def run_experiment(dataset_name, data_root, csv_path, pretrain, mode, k_shot):
    DEVICE = torch.device("cuda:0")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 路径存在性检查
    if not os.path.exists(data_root):
        print(f"错误: 找不到图片路径 {data_root}")
        return

    dataset = SkinDataset(data_root, csv_path, transform=transform)
    Q_QUERY = 15
    # 增加 n_batch 到 300 提高 SCI 数据的可信度
    sampler = CategoriesSampler(dataset.labels, n_batch=500, n_cls=5, n_per=k_shot + Q_QUERY)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)

    model = UniversalModel(mode=mode, pretrain_path=pretrain).to(DEVICE)
    model.eval()

    all_preds, all_targets = [], []
    # 固定的 Query 标签：[0,0...1,1...2,2...] 每类 15 个
    y_query = torch.arange(5).view(5, 1).expand(5, Q_QUERY).reshape(-1).numpy()

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc=f"Testing {dataset_name}"):
            imgs = imgs.to(DEVICE)
            # 使用混合精度推理
            with torch.amp.autocast('cuda'):
                logits = model.meta_forward(imgs, 5, k_shot, Q_QUERY)

            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_query)

    bacc = balanced_accuracy_score(all_targets, all_preds)
    print(f"\n>>> {dataset_name} | {k_shot}-shot | Balanced Acc: {bacc:.4f}")

    # 清理显存防止下一个数据集测试 OOM
    del model, loader
    torch.cuda.empty_cache()
    return bacc


if __name__ == "__main__":

    DATASETS = {
        "SD-198": {
            "root": r"D:\Pycharm\ISIC_project\SD-198\archive\SD-198\roi_square_cropped\images_main",
            "csv": r"D:\Pycharm\ISIC_project\SD-198\archive\SD-198\roi_square_cropped\target_metadata_domain.csv"
        },
        "HAM10000": {
            "root": r"D:\Pycharm\ISIC_project\HAM10000\HAM10000_metadata",
            "csv": r"D:\Pycharm\ISIC_project\HAM10000\HAM10000_metadata.csv"
        }
    }

    # 执行泛化性测试
    # 在 SCI 论文中，建议同时跑 mode=1 和 mode=3 形成对比表格
    for mode_idx in [1,3]:
        # 核心：动态指向你刚才生成的三个文件
        current_weight = f"D:\\Pycharm\\ISIC_project\\final_best_mode_{mode_idx}.pth"

        if not os.path.exists(current_weight):
            print(f"跳过模式 {mode_idx}，因为找不到文件: {current_weight}")
            continue

        print(f"\n" + "=" * 60)
        print(f"{mode_idx} 的权重泛化性测试")
        print(f"权重路径: {current_weight}")
        print("=" * 60)

        for name, paths in DATASETS.items():
            if os.path.exists(paths["csv"]):
                # 这里的 mode 必须跟权重的 mode 对应
                run_experiment(name, paths["root"], paths["csv"], current_weight, mode=mode_idx, k_shot=1)