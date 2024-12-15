import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from resnext import CifarResNeXt  # 从resnext.py加载模型

# 设置环境变量，避免OpenMP问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载CIFAR-10测试数据集
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 加载模型和权重
model = CifarResNeXt(cardinality=32, depth=50, nlabels=10, base_width=4, widen_factor=4)
latest_weights = './pth/model_epoch_190.pth'
model.load_state_dict(torch.load(latest_weights, map_location='cpu',weights_only=True))
model.eval()

# 定位最后一个卷积层
target_layers = [model.stage_3[-1].conv_reduce]

# 初始化Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)

# 生成一个批次的数据
data_iter = iter(test_loader)
images, labels = next(data_iter)

# 逆归一化函数
def reverse_normalize(tensor_image):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    tensor_image = tensor_image * std[:, None, None] + mean[:, None, None]
    return tensor_image

# CIFAR-10类别
class_names = test_dataset.classes

# 可视化批次中的每个图像
batch_size = images.size(0)
rows = batch_size // 4 + (1 if batch_size % 4 != 0 else 0)
fig, axes = plt.subplots(rows, 8, figsize=(15, 4 * rows))

for i in range(batch_size):
    input_image = images[i].unsqueeze(0).requires_grad_()
    output = model(input_image)
    pred_class = output.argmax(dim=1).item()

    targets = [ClassifierOutputTarget(pred_class)]
    grayscale_cam = cam(input_tensor=input_image, targets=targets)[0]

    rgb_img = reverse_normalize(images[i]).numpy().transpose((1, 2, 0))
    rgb_img = np.clip(rgb_img, 0, 1)
    cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    row, col = divmod(i, 4)
    col *= 2

    # 显示原图
    ax = axes[row, col]
    ax.imshow(rgb_img)
    ax.axis('off')
    ax.set_title(f'Label: {class_names[labels[i]]}')

    # 显示热力图
    ax = axes[row, col + 1]
    ax.imshow(cam_img)
    ax.axis('off')
    ax.set_title(f'Grad-CAM: {pred_class}')

plt.tight_layout()
plt.show()
