import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from loss import FocalLoss
from loss import contrastive_loss
from utils import stratified_train_val_test_split_random
from utils import create_sample_pairs_by_labels
from TwoStreamCNN import TwoStreamCNN

# 设置保存路径
save_dir = '/home/data/CNN/model/saved_plots/'
parent_dir = '/home/data/CNN/model/My_results'
model_save_dir = '/home/data/CNN/model/best_models/'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(parent_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)


# 设置随机种子
def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


set_seed(42)

# 设置保存路径
validation_save_dir = '/home/data/CNN/model/validation_data/'
os.makedirs(validation_save_dir, exist_ok=True)

# 焦点损失实例化
focal_loss = FocalLoss(alpha=0.4, gamma=1.8, reduction='mean')

# 加载数据
T1 = np.load('/home/data/CNN/127/T1_data.npy')
T2FS = np.load('/home/data/CNN/127/T2FS_data.npy')
labels = np.load('/home/data/CNN/127/labels.npy')
print("加载已经归一化的数据")
print(f"原始数据形状: T1={T1.shape}, T2FS={T2FS.shape}, labels={labels.shape}")

# 数据转换为 PyTorch 张量
tensor_data1 = torch.tensor(T1, dtype=torch.float32)
tensor_data2 = torch.tensor(T2FS, dtype=torch.float32)
labels = torch.tensor(labels.astype(np.int64), dtype=torch.long)

# ============ 关键修复：添加通道维度 ============
# 如果数据是 3D (num_patients, depth, height, width)，需要添加通道维度
# 变成 4D (num_patients, 1, depth, height, width)
if tensor_data1.dim() == 3:
    tensor_data1 = tensor_data1.unsqueeze(1)  # 添加通道维度
    tensor_data2 = tensor_data2.unsqueeze(1)
    print(f"已添加通道维度: T1={tensor_data1.shape}, T2FS={tensor_data2.shape}")
elif tensor_data1.dim() == 4:
    # 如果已经是 4D，检查是否需要调整维度顺序
    # 假设原始格式是 (num_patients, depth, height, width)
    # 需要变成 (num_patients, 1, depth, height, width)
    if tensor_data1.shape[1] != 1:  # 如果第二个维度不是通道
        tensor_data1 = tensor_data1.unsqueeze(1)
        tensor_data2 = tensor_data2.unsqueeze(1)
        print(f"已调整维度: T1={tensor_data1.shape}, T2FS={tensor_data2.shape}")

print(f"最终数据形状: T1={tensor_data1.shape}, T2FS={tensor_data2.shape}")

# 训练和验证过程
num_runs = 2
num_epochs = 50
all_train_losses = []
all_val_accuracies = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

for run in range(num_runs):
    print(f"\n{'=' * 60}")
    print(f"Starting run {run + 1}")
    print(f"{'=' * 60}")

    # 划分数据：训练集、验证集、测试集
    (train_data1, train_data2, train_labels,
     val_data1, val_data2, val_labels,
     test_data1, test_data2, test_labels) = stratified_train_val_test_split_random(
        tensor_data1, tensor_data2, labels,
        val_size0=10, val_size1=10,
        test_size0=5, test_size1=5,
        random_state=42,
        save_dir=validation_save_dir
    )

    # 打印划分后的数据形状
    print(f"训练集形状: data1={train_data1.shape}, data2={train_data2.shape}, labels={train_labels.shape}")
    print(f"验证集形状: data1={val_data1.shape}, data2={val_data2.shape}, labels={val_labels.shape}")
    print(f"测试集形状: data1={test_data1.shape}, data2={test_data2.shape}, labels={test_labels.shape}")

    # 创建数据集
    train_dataset = TensorDataset(train_data1, train_data2, train_labels)
    val_dataset = TensorDataset(val_data1, val_data2, val_labels)
    test_dataset = TensorDataset(test_data1, test_data2, test_labels)

    # 创建 DataLoader
    train_batch_size = 8
    val_batch_size = 20
    test_batch_size = 10
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # 清理 GPU 内存
    torch.cuda.empty_cache()

    # 初始化模型
    model = TwoStreamCNN(use_cbam=False,  # ✅ 参数1: 是否使用CBAM
                 weight_t1=0.5,  # ✅ 参数2: T1权重
                 weight_t2=2.0)
    model = model.to(device)

    # 打印模型期望的输入形状
    print(f"\n模型已加载到 {device}")
    print(f"模型期望输入: [batch_size, 1, depth, height, width]")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.8)
    criterion_classification = nn.CrossEntropyLoss()

    # 损失加权系数
    contrastive_loss_weight = 0.6
    focal_loss_weight = 0.2
    cross_entropy_loss_weight = 0.8

    # 用于跟踪最佳模型
    best_val_accuracy = 0.0
    best_epoch = 0
    best_model_path = os.path.join(model_save_dir, f'best_model_run_{run}.pth')

    train_losses = []
    val_accuracies = []

    # 训练循环
    for epoch in range(num_epochs):
        # ==================== 训练阶段 ====================
        model.train()
        running_loss = 0.0
        for batch_idx, (data1, data2, batch_labels) in enumerate(train_dataloader):
            data1, data2, batch_labels = data1.to(device), data2.to(device), batch_labels.to(device)

            # 调试：打印第一个batch的形状
            if epoch == 0 and batch_idx == 0:
                print(f"\n第一个batch形状: data1={data1.shape}, data2={data2.shape}")

            optimizer.zero_grad()

            # 提取特征
            features = model(data1, data2, extract_features=True)

            # 创建正负样本对
            pos_pairs, neg_pairs = create_sample_pairs_by_labels(batch_labels)

            # 计算对比损失
            contrastive_loss_value = contrastive_loss(features, pos_pairs, neg_pairs, margin=2)

            # 分类损失
            classification_logits = model(data1, data2, classify=True)
            classification_loss = criterion_classification(classification_logits, batch_labels)

            # 计算焦点损失
            focal_loss_value = focal_loss(classification_logits, batch_labels)

            # 总损失
            total_loss = (contrastive_loss_weight * contrastive_loss_value +
                          cross_entropy_loss_weight * classification_loss +
                          focal_loss_weight * focal_loss_value)

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # ==================== 验证阶段 ====================
        model.eval()
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        all_predictions_probabilities = []

        with torch.no_grad():
            for data1, data2, batch_val_labels in val_dataloader:
                data1, data2, batch_val_labels = data1.to(device), data2.to(device), batch_val_labels.to(device)
                outputs = model(data1, data2, classify=True)

                softmax_output = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(softmax_output, 1)

                total += batch_val_labels.size(0)
                correct += (predicted == batch_val_labels).sum().item()

                all_labels.extend(batch_val_labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_predictions_probabilities.extend(softmax_output[:, 1].cpu().numpy())

        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        # 计算评估指标
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        auroc = roc_auc_score(all_labels, all_predictions_probabilities)
        aupr = average_precision_score(all_labels, all_predictions_probabilities)

        # 计算 ROC 曲线
        fpr, tpr, _ = roc_curve(all_labels, all_predictions_probabilities)
        roc_auc = auc(fpr, tpr)

        # 打印当前 epoch 结果
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.2f}%")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"  AUROC: {auroc:.4f} | AUPR: {aupr:.4f}")

        # ==================== 保存最佳模型 ====================
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_metrics': {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auroc': auroc,
                    'aupr': aupr
                }
            }, best_model_path)

            print(f"   新的最佳模型已保存！验证准确率: {val_accuracy:.2f}%")

        # 保存验证准确率 > 70% 的结果
        if val_accuracy > 70:
            epoch_save_dir = os.path.join(parent_dir, f"run_{run}_epoch_{epoch + 1}")
            os.makedirs(epoch_save_dir, exist_ok=True)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUROC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR)')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc='lower right')
            roc_path = os.path.join(epoch_save_dir, "roc_curve.png")
            plt.savefig(roc_path)
            plt.close()

            metrics_path = os.path.join(epoch_save_dir, "metrics.txt")
            with open(metrics_path, "w") as f:
                f.write(f"Validation Accuracy: {val_accuracy:.2f}%\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"AUROC: {auroc:.4f}\n")
                f.write(f"AUPR: {aupr:.4f}\n")

        scheduler.step(val_accuracy)
        torch.cuda.empty_cache()

    # ==================== 测试阶段 ====================
    print(f"\n{'=' * 60}")
    print(f"加载最佳模型进行测试 (Epoch {best_epoch}, Val Acc: {best_val_accuracy:.2f}%)")
    print(f"{'=' * 60}")

    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_correct = 0
    test_total = 0
    test_all_labels = []
    test_all_predictions = []
    test_all_predictions_probabilities = []

    with torch.no_grad():
        for data1, data2, batch_test_labels in test_dataloader:
            data1, data2, batch_test_labels = data1.to(device), data2.to(device), batch_test_labels.to(device)
            outputs = model(data1, data2, classify=True)

            softmax_output = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(softmax_output, 1)

            test_total += batch_test_labels.size(0)
            test_correct += (predicted == batch_test_labels).sum().item()

            test_all_labels.extend(batch_test_labels.cpu().numpy())
            test_all_predictions.extend(predicted.cpu().numpy())
            test_all_predictions_probabilities.extend(softmax_output[:, 1].cpu().numpy())

    test_accuracy = 100 * test_correct / test_total
    test_precision = precision_score(test_all_labels, test_all_predictions, zero_division=0)
    test_recall = recall_score(test_all_labels, test_all_predictions, zero_division=0)
    test_f1 = f1_score(test_all_labels, test_all_predictions, zero_division=0)
    test_auroc = roc_auc_score(test_all_labels, test_all_predictions_probabilities)
    test_aupr = average_precision_score(test_all_labels, test_all_predictions_probabilities)

    test_fpr, test_tpr, _ = roc_curve(test_all_labels, test_all_predictions_probabilities)
    test_roc_auc = auc(test_fpr, test_tpr)

    print(f"\n{'=' * 60}")
    print(f"测试集结果 (Run {run + 1}):")
    print(f"{'=' * 60}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"AUROC: {test_auroc:.4f}")
    print(f"AUPR: {test_aupr:.4f}")
    print(f"{'=' * 60}\n")

    test_results_dir = os.path.join(parent_dir, f"test_results_run_{run}")
    os.makedirs(test_results_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(test_fpr, test_tpr, color='red', lw=2, label=f'Test ROC Curve (AUROC = {test_roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Test Set ROC Curve')
    plt.legend(loc='lower right')
    test_roc_path = os.path.join(test_results_dir, "test_roc_curve.png")
    plt.savefig(test_roc_path)
    plt.close()

    test_metrics_path = os.path.join(test_results_dir, "test_metrics.txt")
    with open(test_metrics_path, "w") as f:
        f.write(f"Best Model from Epoch: {best_epoch}\n")
        f.write(f"Best Validation Accuracy: {best_val_accuracy:.2f}%\n\n")
        f.write(f"Test Set Results:\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        f.write(f"Precision: {test_precision:.4f}\n")
        f.write(f"Recall: {test_recall:.4f}\n")
        f.write(f"F1 Score: {test_f1:.4f}\n")
        f.write(f"AUROC: {test_auroc:.4f}\n")
        f.write(f"AUPR: {test_aupr:.4f}\n")

    print(f"测试结果已保存到 {test_results_dir}")

    all_train_losses.append(train_losses)
    all_val_accuracies.append(val_accuracies)

# 创建训练损失图
plt.figure(figsize=(8, 6))
for i in range(num_runs):
    plt.plot(all_train_losses[i], label=f"Run {i + 1}")
plt.title("Variability of Training Loss Across Runs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(save_dir, "training_loss_plot.png"))
plt.close()

# 创建验证准确率图
plt.figure(figsize=(8, 6))
for i in range(num_runs):
    plt.plot(all_val_accuracies[i], label=f"Run {i + 1}")
plt.title("Accuracy Across Multiple Runs with Different Random Seeds")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig(os.path.join(save_dir, "validation_accuracy_plot.png"))
plt.close()

print("\n 所有训练和测试已完成！")
print(f"最佳模型保存在: {model_save_dir}")
print(f"测试结果保存在: {parent_dir}/test_results_run_*")