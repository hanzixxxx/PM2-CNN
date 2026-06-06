import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from sklearn.preprocessing import StandardScaler
import random
import os


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设备选择
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 数据文件路径
data_files = [
    {'train': 'data/Contact_feature/ant/ant-1.5.csv', 'test': 'data/Contact_feature/ant/ant-1.6.csv'},
    {'train': 'data/Contact_feature/ant/ant-1.6.csv', 'test': 'data/Contact_feature/ant/ant-1.7.csv'},
    {'train': 'data/Contact_feature/camel/camel-1.2.csv', 'test': 'data/Contact_feature/camel/camel-1.4.csv'},
    {'train': 'data/Contact_feature/camel/camel-1.4.csv', 'test': 'data/Contact_feature/camel/camel-1.6.csv'},
    {'train': 'data/Contact_feature/jedit/jedit-4.1.csv', 'test': 'data/Contact_feature/jedit/jedit-4.2.csv'},
    {'train': 'data/Contact_feature/jedit/jedit-4.2.csv', 'test': 'data/Contact_feature/jedit/jedit-4.3.csv'},
    {'train': 'data/Contact_feature/lucene/lucene-2.0.csv', 'test': 'data/Contact_feature/lucene/lucene-2.2.csv'},
    {'train': 'data/Contact_feature/lucene/lucene-2.2.csv', 'test': 'data/Contact_feature/lucene/lucene-2.4.csv'},
    {'train': 'data/Contact_feature/poi/poi-1.5.csv', 'test': 'data/Contact_feature/poi/poi-2.5.csv'},
    {'train': 'data/Contact_feature/poi/poi-2.5.csv', 'test': 'data/Contact_feature/poi/poi-3.0.csv'},
    {'train': 'data/Contact_feature/velocity/velocity-1.4.csv', 'test': 'data/Contact_feature/velocity/velocity-1.5.csv'},
    {'train': 'data/Contact_feature/velocity/velocity-1.5.csv', 'test': 'data/Contact_feature/velocity/velocity-1.6.csv'},
    {'train': 'data/Contact_feature/xalan/xalan-2.5.csv','test': 'data/Contact_feature/xalan/xalan-2.6.csv'},
    {'train': 'data/Contact_feature/xalan/xalan-2.6.csv','test': 'data/Contact_feature/xalan/xalan-2.7.csv'},

    # 可以在这里添加更多的文件路径
]
# 替换 GHLSTMModel 为多通道 CNN 模型
class MultiChannelCNNModel(nn.Module):
    def __init__(self, input_dim, num_filters=128, kernel_sizes=[3, 4], output_dim=1):
        super(MultiChannelCNNModel, self).__init__()

        # 语义特征 CNN 通道
        self.semantic_convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])

        # 自然语言特征 CNN 通道
        self.nl_convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])

        # 全局最大池化层
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # 全连接层（拼接两个 CNN 通道的输出）
        self.fc = nn.Linear(len(kernel_sizes) * num_filters * 2, output_dim)
        self.sigmoid = nn.Sigmoid()  # 用于二分类

    def forward(self, semantic_features, nl_features):
        """
        输入：
        - semantic_features: [batch_size, input_dim]
        - nl_features: [batch_size, input_dim]
        """
        # 需要调整形状，以适应 1D CNN 输入格式 [batch_size, channels, seq_len]
        semantic_features = semantic_features.unsqueeze(1)  # 变成 [batch_size, 1, input_dim]
        nl_features = nl_features.unsqueeze(1)  # 变成 [batch_size, 1, input_dim]

        # 语义特征 CNN 提取
        semantic_outputs = [self.global_max_pool(torch.relu(conv(semantic_features))).squeeze(2) for conv in
                            self.semantic_convs]
        semantic_outputs = torch.cat(semantic_outputs, dim=1)  # 拼接所有卷积核的输出

        # 自然语言特征 CNN 提取
        nl_outputs = [self.global_max_pool(torch.relu(conv(nl_features))).squeeze(2) for conv in self.nl_convs]
        nl_outputs = torch.cat(nl_outputs, dim=1)  # 拼接所有卷积核的输出

        # 特征拼接
        combined_features = torch.cat((semantic_outputs, nl_outputs), dim=1)

        # 分类层
        output = self.sigmoid(self.fc(combined_features))
        return output

# 损失函数
criterion = nn.BCELoss()
results = []
# 遍历文件
for files in data_files:
    data = pd.read_csv(files['train'])
    data_test = pd.read_csv(files['test'])

    X_features = data[[f'Feature_{i}' for i in range(1,769)]]
    additional_features = data[[f'text_{i}' for i in range(1,769)]]

    y = data['bug'].values

    X_test_features = data_test[[f'Feature_{i}' for i in range(1,769)]]
    X_test_additional = data_test[[f'text_{i}' for i in range(1,769)]]
    y_test = data_test['bug'].values

    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    X_test_features = scaler.transform(X_test_features)
    X_features = abs(X_features)
    X_test_features = abs(X_test_features)
    trad_features_scaler = StandardScaler()
    additional_features = trad_features_scaler.fit_transform(additional_features)
    X_test_additional = trad_features_scaler.transform(X_test_additional)

    X_train_tensor = torch.tensor(X_features, dtype=torch.float32).to(device)
    additional_features_tensor = torch.tensor(additional_features, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test_features, dtype=torch.float32).to(device)
    additional_features_test_tensor = torch.tensor(X_test_additional, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device).unsqueeze(1)
    # 模型参数
    inter_matrix_dim = X_features.shape[1]
    trad_feature_dim = 20
    hidden_dim = 128
    output_dim = 2
    learning_rate = 0.002
    num_epochs = 80
    num_runs = 5
    best_scores = []

    for run in range(1, num_runs + 1):
        print(f'\n===== 开始大轮次 {run} =====')

        set_seed(42 + run)

        model = MultiChannelCNNModel(inter_matrix_dim, trad_feature_dim).to(device)

        best_f1 = 0.0
        best_metrics = {}

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            outputs = model(X_train_tensor, additional_features_tensor)
            loss = criterion(outputs, y_train_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

            model.eval()
            with torch.no_grad():
                print(f"Max index in inter_matrix: {torch.max(X_test_tensor).item()}")
                outputs = model(X_test_tensor, additional_features_test_tensor)
                predicted = (outputs >= 0.5).float().cpu().numpy()

                y_test_np = y_test_tensor.cpu().numpy()
                predicted_np = predicted

                accuracy = accuracy_score(y_test_np, predicted_np)
                f1 = f1_score(y_test_np, predicted_np, average='weighted')
                recall = recall_score(y_test_np, predicted_np, average='weighted')
                auc = roc_auc_score(y_test_np, outputs.cpu().numpy())

                if f1 > best_f1:
                    best_f1 = f1
                    best_metrics = {
                        'F1': f1,
                        'AUC': auc,
                        'Accuracy': accuracy,
                    }

                print(f'--- 大轮次 {run}, Epoch [{epoch + 1}/{num_epochs}] 评估结果 ---')
                print(f'F1 Score: {f1:.4f}')
                print(f'AUC: {auc:.4f}')
                print(f'Accuracy: {accuracy:.4f}')

        best_scores.append(best_metrics)
        print(f'大轮次 {run} 的最佳 F1 分数及指标: {best_metrics}')

    average_metrics = {
        'F1': np.mean([score['F1'] for score in best_scores]),
        'AUC': np.mean([score['AUC'] for score in best_scores]),
        'Accuracy': np.mean([score['Accuracy'] for score in best_scores]),
    }

    results.append(
        {'F1': average_metrics["F1"], 'AUC': average_metrics["AUC"], 'Accuracy': average_metrics["Accuracy"]})

results_df = pd.DataFrame(results)
results_df.to_excel('model_results80_version.xlsx', index=False)
print("结果已保存到 model_results1.xlsx")
