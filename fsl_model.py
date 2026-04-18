import torch
import torch.nn as nn
import torch.nn.functional as F
from mobilevit import get_mobilevit_xs_for_isic  # 注意这里改为 xs


# ==========================================
# 3.3 动态原型校准模块 (DPCM)
# ==========================================
class DPCM(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super(DPCM, self).__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, support_features):
        scores = self.attn_net(support_features)
        weights = F.softmax(scores, dim=1)
        prototypes = torch.sum(weights * support_features, dim=1)
        return prototypes


# ==========================================
# 3.4 整体网络模型
# ==========================================
class MobileViT_ProtoNet(nn.Module):
    def __init__(self, backbone_path=None, use_dpcm=True, feature_dim=384):  # XS 输出是 384
        super(MobileViT_ProtoNet, self).__init__()

        print(f"初始化 MobileViT-XS Backbone (Dim={feature_dim})...")
        # 调用 xs 的构建函数
        self.encoder = get_mobilevit_xs_for_isic(backbone_path, num_classes=0)
        self.encoder.classifier = nn.Identity()

        self.feature_dim = feature_dim
        self.use_dpcm = use_dpcm

        if self.use_dpcm:
            print("启用 DPCM (Dynamic Prototype Calibration Module)")
            self.dpcm = DPCM(self.feature_dim)
        else:
            print("使用原始 ProtoNet (Mean Prototype)")

    def forward(self, x):
        x = self.encoder.conv_1(x)
        x = self.encoder.layer_1(x)
        x = self.encoder.layer_2(x)
        x = self.encoder.layer_3(x)
        x = self.encoder.layer_4(x)
        x = self.encoder.layer_5(x)
        x = self.encoder.conv_1x1_exp(x)

        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return x

    def meta_forward(self, x, n_way, k_shot, q_query):
        z = self.forward(x)
        z = z.view(n_way, k_shot + q_query, -1)

        support_features = z[:, :k_shot, :]
        query_features = z[:, k_shot:, :]

        if self.use_dpcm:
            prototypes = self.dpcm(support_features)
        else:
            prototypes = support_features.mean(1)

        query_features_flat = query_features.contiguous().view(n_way * q_query, -1)

        dists = self.euclidean_dist(query_features_flat, prototypes)

        log_p_y = F.log_softmax(-dists, dim=1)

        y_query = torch.arange(n_way).view(n_way, 1).expand(n_way, q_query).reshape(-1)
        y_query = y_query.to(x.device)

        return log_p_y, y_query

    def euclidean_dist(self, x, y):
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return torch.pow(x - y, 2).sum(2)