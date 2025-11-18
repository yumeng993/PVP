import torch.nn.functional as F
import torch
import torch.nn as nn
# 定义残差模块
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# ==================== CBAM 注意力模块 ====================
class ChannelAttention(nn.Module):

    def __init__(self, in_channels, reduction=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # 共享的 MLP
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # 相加后经过 sigmoid
        out = self.sigmoid(avg_out + max_out)
        return out

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 使用较大的卷积核捕获空间信息
        padding = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上做平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接后通过卷积
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):


    def __init__(self, in_channels, reduction=4, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先应用通道注意力
        x = x * self.channel_attention(x)
        # 再应用空间注意力
        x = x * self.spatial_attention(x)
        return x


class CBAM_MultiModalFusion(nn.Module):
    def __init__(self, in_channels=8, reduction=4, kernel_size=7):
        super(CBAM_MultiModalFusion, self).__init__()
        self.cbam_t1 = CBAM(in_channels, reduction, kernel_size)
        self.cbam_t2 = CBAM(in_channels, reduction, kernel_size)

        # 可选：对融合后的特征再应用一次 CBAM
        self.cbam_fusion = CBAM(in_channels * 2, reduction, kernel_size)

    def forward(self, F_T1, F_T2):
        """
        F_T1: T1 模态特征 (B, C, D, H, W)
        F_T2: T2FS 模态特征 (B, C, D, H, W)
        """
        # 对每个模态分别应用 CBAM
        F_T1_attended = self.cbam_t1(F_T1)
        F_T2_attended = self.cbam_t2(F_T2)

        # 拼接两个模态
        fused = torch.cat([F_T1_attended, F_T2_attended], dim=1)

        # 对融合后的特征再次应用 CBAM（可选，增强融合效果）
        fused = self.cbam_fusion(fused)

        return fused


# ==================== 带 CBAM 的双流 CNN 模型 ====================

class TwoStreamCNN(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=2,
                 use_cbam=False,
                 weight_t1=0.5,
                 weight_t2=2.0,
                 cbam_reduction=4,
                 cbam_kernel_size=7):
        super(TwoStreamCNN, self).__init__()
        self.use_cbam = use_cbam

        # 第一层卷积+池化（独立处理 T1 和 T2FS）
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # ✅ 根据 use_cbam 决定使用哪种融合方式
        if self.use_cbam:
            # 使用 CBAM 注意力融合
            self.cbam_fusion = CBAM_MultiModalFusion(
                in_channels=8,
                reduction=cbam_reduction,
                kernel_size=cbam_kernel_size
            )
            print(f"使用 CBAM 注意力机制 (reduction={cbam_reduction}, kernel_size={cbam_kernel_size})")
        else:
            # 使用自定义固定权重融合
            self.register_buffer('weight_t1', torch.tensor(weight_t1, dtype=torch.float32))
            self.register_buffer('weight_t2', torch.tensor(weight_t2, dtype=torch.float32))
            print(f"使用固定权重融合 (T1权重={weight_t1:.2f}, T2FS权重={weight_t2:.2f})")

        # 融合后的卷积+池化
        self.conv_after_fusion1 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # 残差块
        self.residual_block = ResidualBlock3D(32, 32)

        # 用于对比学习的额外分支
        self.conv_for_contrastive = nn.Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        # Dropout 和全连接层
        self.dropout = nn.Dropout(p=0.4)
        self.embedding_layer = nn.Linear(196608, embedding_dim)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x1, x2, classify=False, extract_features=False):
        """
        x1: T1 模态输入 (B, 1, D, H, W)
        x2: T2FS 模态输入 (B, 1, D, H, W)
        classify: 是否进行分类
        extract_features: 是否提取用于对比学习的特征
        """
        # T1 和 T2FS 分别经过第一层卷积和池化
        x1 = self.pool1(F.relu(self.conv1(x1)))  # (B, 8, D', H', W')
        x2 = self.pool1(F.relu(self.conv1(x2)))  # (B, 8, D', H', W')

        # ✅ 特征融合：根据 use_cbam 选择融合方式
        if self.use_cbam:
            # 使用 CBAM 进行多模态融合
            x_fusion = self.cbam_fusion(x1, x2)  # (B, 16, D', H', W')
        else:
            # 使用固定权重进行加权拼接
            weighted_x1 = self.weight_t1 * x1
            weighted_x2 = self.weight_t2 * x2
            x_fusion = torch.cat((weighted_x1, weighted_x2), dim=1)

        # 如果需要提取对比学习特征
        if extract_features:
            x_fusion = F.relu(self.conv_for_contrastive(x_fusion))
            x_fusion = self.pool2(x_fusion)
            x_fusion = x_fusion.view(x_fusion.size(0), -1)
            return x_fusion

        # 融合特征后的卷积操作
        x_fusion = F.relu(self.conv_after_fusion1(x_fusion))  # (B, 32, D'', H'', W'')
        x_fusion = self.residual_block(x_fusion)  # (B, 32, D'', H'', W'')
        x_fusion = self.pool2(x_fusion)  # (B, 32, D''', H''', W''')

        # 扁平化特征
        x_fusion = x_fusion.view(x_fusion.size(0), -1)

        # 如果需要分类
        if classify:
            x_fusion = self.dropout(x_fusion)
            embedding = F.relu(self.embedding_layer(x_fusion))
            return self.fc(embedding)

        return x_fusion

    def get_fusion_weights(self):
        """获取当前的融合权重"""
        if self.use_cbam:
            return None  # CBAM 使用动态注意力，没有固定权重
        else:
            return {
                'weight_t1': self.weight_t1.item(),
                'weight_t2': self.weight_t2.item()
            }