import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
data = pd.read_csv("logs/results.csv")

# 创建子图
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# 绘制Accuracy图
axs[0].plot(data['epoch'], data['train acc'], label='Train Accuracy', color='blue')
axs[0].plot(data['epoch'], data['test acc'], label='Test Accuracy', color='orange')
axs[0].set_title('Model Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy (%)')
axs[0].legend()

# 绘制Loss图
axs[1].plot(data['epoch'], data['train loss'], label='Train Loss', color='blue')
axs[1].plot(data['epoch'], data['test loss'], label='Test Loss', color='orange')
axs[1].set_title('Model Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend()

# 显示图表
plt.tight_layout()
plt.show()
