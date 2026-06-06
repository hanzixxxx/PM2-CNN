import os
import torch
import re
import csv
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # 导入tqdm库

# 处理文件名
def cut(file_path):
    delimiter = "org"
    parts = re.split(f"({delimiter})", file_path, maxsplit=1)
    if len(parts) > 1:
        parts[1] = parts[1] + parts[2]
        parts.pop(2)
    filenames = parts[1].split(".")
    filenames = filenames[0]
    filenames = ''.join(filenames)
    filenames = filenames.replace("\\", ".")
    return filenames

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载 UniXcoder 模型和 tokenizer
model_name = "microsoft/unixcoder-base"  # 请根据实际模型名称调整
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = model.to(device)

# 切换模型到评估模式（禁用dropout等操作）
model.eval()

# 设置最大长度（根据模型的实际支持长度调整）
max_length = 1024  # 根据模型文档设置适当的长度

# 设置批量大小（batch size）
batch_size = 1
# 指定文件夹的根目录
root_directory = 'D:\\work\\Data\\dataset\\sourcetext\\xalan\\xalan-2.7'

# 创建一个字典来保存文件路径与其特征
file_features = {}

# 遍历文件夹
for foldername, subfolders, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.endswith('.txt'):  # 处理 .txt 文件
            filepath = os.path.join(foldername, filename)

            print("*********开始执行" + foldername + "\\" + filename + "*********")

            # 读取Java文件中的代码
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()

            # 将代码片段封装为列表
            code_data = [code]

            # 处理和表示提取
            representations = []

            # 使用tqdm来显示进度条
            with tqdm(total=(len(code_data) // batch_size + (len(code_data) % batch_size != 0)), desc=f"Processing {filename}", unit="batch") as pbar:
                for i in range(0, len(code_data), batch_size):
                    batch_text = code_data[i:i + batch_size]

                    # 处理批量代码片段
                    encoded_batch = tokenizer(batch_text,
                                              return_tensors="pt",
                                              padding=True,
                                              truncation=True,
                                              max_length=max_length)

                    # 将输入数据移至GPU
                    encoded_batch = {key: value.to(device) for key, value in encoded_batch.items()}

                    # 提取表示
                    with torch.no_grad():
                        output = model(**encoded_batch)

                    # 获取节点特征（假设 GraphBERT 的最后一层的CLS向量作为特征）
                    batch_cls_vectors = output.last_hidden_state[:, 0, :]  # 获取批量CLS向量
                    representations.extend(batch_cls_vectors.cpu().numpy())  # 转换为NumPy数组并移至CPU

                    pbar.update(1)  # 更新进度条

            # 将提取的特征向量存储在字典中
            filepath_cut = cut(filepath)
            file_features[filepath_cut] = representations

            # 显示完成进度
            print(f"Processing {filename} completed!")

# 输出结果
csv_file = "../data/Original_extraction/xalan/xalan-2.7.csv"

# 写入CSV文件
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # 写入表头
    if representations:
        header = ["name"] + [f"text_{i+1}" for i in range(len(representations[0]))]
        writer.writerow(header)

        # 写入数据
        for filepath, features in file_features.items():
            for feature_vector in features:
                row = [filepath] + feature_vector.tolist()  # 将 NumPy 数组转换为列表
                writer.writerow(row)

print(f"Feature data has been written to {csv_file}")
