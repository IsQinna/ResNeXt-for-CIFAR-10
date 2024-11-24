import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.nn.init as init
from resnext import CifarResNeXt
import os
from tqdm import tqdm
import csv

# 超参数设置
batch_size = 16
epochs = 200
learning_rate = 0.1
cardinality = 32  # 设置为32来选择ResNeXt的宽度
depth = 50  # 选择ResNeXt的深度
base_width = 4
nlabels = 10  # 如果是CIFAR-10，标签数为10，CIFAR-100则为100

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 创建模型
model = CifarResNeXt(cardinality=cardinality, depth=depth, nlabels=nlabels, base_width=base_width)

# 使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", ncols=100)):
        inputs, targets = inputs.to(device), targets.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)
        loss.backward()

        # 更新权重
        optimizer.step()

        # 计算统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Train Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")  # 打印训练损失和准确率
    return accuracy, avg_loss

# 测试函数
def test(epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # 计算损失
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / len(test_loader)  # 平均损失
    accuracy = 100 * correct / total
    print(f"Test Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")  # 打印测试损失和准确率
    return accuracy, avg_loss  # 返回准确率和损失

# 学习率调度
def adjust_learning_rate(optimizer, epoch, lr_schedule=[60, 120, 160]):
    if epoch in lr_schedule:
        new_lr = optimizer.param_groups[0]['lr'] * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"Learning rate adjusted to {new_lr}")

# 保存结果为CSV格式
def save_results(epoch, train_acc, test_acc, train_loss, test_loss):
    file_exists = os.path.exists("results.csv")
    with open("results.csv", mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # 写入表头
            writer.writerow(["epoch", "train acc", "test acc", "train loss", "test loss"])
        # 写入数据
        writer.writerow([epoch, train_acc, test_acc, train_loss, test_loss])

# 主训练循环
# 主训练循环
if __name__ == '__main__':
    model_save_path = './pth'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epoch)  # 调整学习率
        train_acc, train_loss = train(epoch)
        test_acc, test_loss = test(epoch)  # 获取测试准确率和损失

        # 保存结果到CSV
        save_results(epoch, train_acc, test_acc, train_loss, test_loss)

        # 每隔一定轮次保存模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(model_save_path, f'model_epoch_{epoch}.pth'))

