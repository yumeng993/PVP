import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from TwoStreamCNN import TwoStreamCNN

class ModelTester:
    def __init__(self, model_path, data_dir, device=None):
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\n{'=' * 70}")
        print(f"PVP疼痛缓解预测 - 模型测试")
        print(f"{'=' * 70}")
        print(f"模型路径: {model_path}")
        print(f"数据目录: {data_dir}")
        print(f"计算设备: {self.device}")
        print(f"{'=' * 70}\n")

    def load_model(self, model_config):
        print("加载模型...")
        # 创建模型实例
        model = TwoStreamCNN(**model_config)
        # 加载权重
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 兼容不同的保存格式
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"模型已加载 (Epoch {checkpoint.get('epoch', 'unknown')})")
                if 'val_accuracy' in checkpoint:
                    print(f"   训练时最佳验证准确率: {checkpoint['val_accuracy']:.2f}%")
            else:
                model.load_state_dict(checkpoint)
                print(f"模型已加载")
        else:
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        model = model.to(self.device)
        model.eval()

        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   模型参数量: {total_params:,}")

        return model

    def load_data(self, data_type='test'):
        print(f"\n加载{data_type}数据...")

        # 数据文件路径
        data1_path = os.path.join(self.data_dir, f'{data_type}_data1.npy')
        data2_path = os.path.join(self.data_dir, f'{data_type}_data2.npy')
        labels_path = os.path.join(self.data_dir, f'{data_type}_labels.npy')

        # 检查文件是否存在
        for path in [data1_path, data2_path, labels_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"数据文件不存在: {path}")

        # 加载数据
        data1 = np.load(data1_path)
        data2 = np.load(data2_path)
        labels = np.load(labels_path)

        # 转换为张量
        tensor_data1 = torch.tensor(data1, dtype=torch.float32)
        tensor_data2 = torch.tensor(data2, dtype=torch.float32)
        tensor_labels = torch.tensor(labels, dtype=torch.long)

        # 添加通道维度（如果需要）
        if tensor_data1.dim() == 3:
            tensor_data1 = tensor_data1.unsqueeze(1)
            tensor_data2 = tensor_data2.unsqueeze(1)
            print(f"   已添加通道维度: {tensor_data1.shape}")

        # 创建数据集和DataLoader
        dataset = TensorDataset(tensor_data1, tensor_data2, tensor_labels)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        return dataloader, labels

    def predict(self, model, dataloader):
        print("\n开始预测...")

        model.eval()
        y_true = []
        y_pred = []
        y_prob = []

        with torch.no_grad():
            for data1, data2, labels in dataloader:
                data1 = data1.to(self.device)
                data2 = data2.to(self.device)

                # 前向传播
                outputs = model(data1, data2, classify=True)

                # 获取概率和预测
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                # 收集结果
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probs[:, 1].cpu().numpy())  # 类别1的概率

        print(f"预测完成，共 {len(y_true)} 个样本")

        return np.array(y_true), np.array(y_pred), np.array(y_prob)

    def calculate_metrics(self, y_true, y_pred, y_prob):
        """
        计算所有评估指标
        """
        print(f"\n{'=' * 70}")
        print("评估指标")
        print(f"{'=' * 70}")

        # 基础指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # 特异度 (Specificity)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # AUC指标
        try:
            auroc = roc_auc_score(y_true, y_prob)
            aupr = average_precision_score(y_true, y_prob)
        except:
            auroc = 0
            aupr = 0
            print(" 无法计算AUC（可能标签只有一个类别）")

        # 打印指标
        print(f"\n基础指标:")
        print(f"  准确率 (Accuracy):     {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  精确率 (Precision):    {precision:.4f}")
        print(f"  召回率 (Recall/敏感度): {recall:.4f}")
        print(f"  特异度 (Specificity):  {specificity:.4f}")
        print(f"  F1分数 (F1-Score):     {f1:.4f}")

        print(f"\nAUC指标:")
        print(f"  AUROC:                 {auroc:.4f}")
        print(f"  AUPR (AP):             {aupr:.4f}")

        print(f"\n混淆矩阵:")
        print(f"  真阴性 (TN): {tn:3d}  |  假阳性 (FP): {fp:3d}")
        print(f"  假阴性 (FN): {fn:3d}  |  真阳性 (TP): {tp:3d}")

        # 详细分类报告
        print(f"\n详细分类报告:")
        print(classification_report(y_true, y_pred,
                                    target_names=['疼痛缓解不佳(0)', '疼痛缓解良好(1)'],
                                    digits=4))

        print(f"{'=' * 70}\n")

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'auroc': auroc,
            'aupr': aupr,

        }

        return metrics

    def plot_roc_curve(self, y_true, y_prob, save_path):
        """绘制ROC曲线"""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2,
                 label=f'ROC曲线 (AUC = {roc_auc:.4f})')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
        plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
        plt.title('ROC曲线 (Receiver Operating Characteristic)', fontsize=14, pad=20)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC曲线已保存: {save_path}")

    def save_predictions(self, y_true, y_pred, y_prob, save_path):
        """保存预测结果到文件"""
        results = np.column_stack([y_true, y_pred, y_prob])
        np.savetxt(save_path, results,
                   header='True_Label,Predicted_Label,Probability_Class1',
                   delimiter=',', fmt='%d,%d,%.6f', comments='')
        print(f"✅ 预测结果已保存: {save_path}")

    def run_full_test(self, model_config, data_type='test', output_dir='./test_results'):
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 1. 加载模型
        model = self.load_model(model_config)

        # 2. 加载数据
        dataloader, labels = self.load_data(data_type)

        # 3. 预测
        y_true, y_pred, y_prob = self.predict(model, dataloader)

        # 4. 计算指标
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)

        self.plot_roc_curve(y_true, y_prob,
                            os.path.join(output_dir, 'roc_curve.png'))

        # 6. 保存预测结果
        self.save_predictions(y_true, y_pred, y_prob,
                              os.path.join(output_dir, 'predictions.csv'))
        # 7. 保存指标到文件
        metrics_path = os.path.join(output_dir, 'metrics.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 70}\n")
            f.write(f"PVP疼痛缓解预测 - 测试结果\n")
            f.write(f"{'=' * 70}\n\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"数据类型: {data_type}\n")
            f.write(f"样本数量: {len(y_true)}\n")
            f.write(f"标签分布: 0={np.sum(y_true == 0)}, 1={np.sum(y_true == 1)}\n\n")

            f.write(f"评估指标:\n")
            f.write(f"  准确率 (Accuracy):     {metrics['accuracy']:.4f}\n")
            f.write(f"  精确率 (Precision):    {metrics['precision']:.4f}\n")
            f.write(f"  召回率 (Recall):       {metrics['recall']:.4f}\n")
            f.write(f"  特异度 (Specificity):  {metrics['specificity']:.4f}\n")
            f.write(f"  F1分数 (F1-Score):     {metrics['f1']:.4f}\n")
            f.write(f"  AUROC:                 {metrics['auroc']:.4f}\n")
            f.write(f"  AUPR:                  {metrics['aupr']:.4f}\n\n")
        print(f"✅ 评估指标已保存: {metrics_path}")

        print(f"\n{'=' * 70}")
        print(f"测试完成！所有结果已保存到: {output_dir}")
        print(f"{'=' * 70}\n")

        return metrics


# ==================== 主程序 ====================

if __name__ == "__main__":
    # ==================== 配置参数 ====================

    # 模型路径
    MODEL_PATH = '/home/data/CNN/model/best_models/best_model_run_0.pth'

    # 数据目录
    DATA_DIR = '/home/data/CNN/model/validation_data/'

    # 输出目录
    OUTPUT_DIR = '/home/data/CNN/model/test_results/'

    MODEL_CONFIG = {
        'embedding_dim': 128,
        'num_classes': 2,
        'use_cbam': False,  # ✅ 改为您使用的配置
        'weight_t1': 0.5,  # ✅ 改为您的最佳权重
        'weight_t2': 2.0,  # ✅ 改为您的最佳权重
        'cbam_reduction': 4,
        'cbam_kernel_size': 7
    }

    # 测试数据类型（'test' 或 'val'）
    DATA_TYPE = 'test'  # 或 'val'

    # ==================== 运行测试 ====================

    tester = ModelTester(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR
    )

    metrics = tester.run_full_test(
        model_config=MODEL_CONFIG,
        data_type=DATA_TYPE,
        output_dir=OUTPUT_DIR
    )