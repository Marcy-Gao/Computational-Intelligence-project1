# ==================== 1. 导入必要的库 ====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# ==================== 2. 基础配置 ====================

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'当前使用的设备: {device}')


data_dir = './organized_data'        
model_save_path = './models/best_model.pth'  

# ==================== 3. 数据预处理与增强 ====================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),           
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomRotation(degrees=20),    
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),                    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== 4. 加载数据集 ====================
full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

# 打印类别信息
print(f'类别名称: {full_dataset.classes}')        
print(f'类别索引: {full_dataset.class_to_idx}')   
print(f'总图片数: {len(full_dataset)}')

train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform

# ==================== 5. 创建 DataLoader ====================
batch_size = 32  

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f'训练集图片数: {len(train_dataset)}')
print(f'验证集图片数: {len(val_dataset)}')
print(f'每个 batch 的图片数: {batch_size}')

# ==================== 6. 构建模型 ====================
def build_model(num_classes=2):
    """
    加载预训练的 ResNet50，并修改最后的全连接层
    """
    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),  
        nn.ReLU(),
        nn.Dropout(0.5),               
        nn.Linear(512, num_classes)   
    )
    
    return model

model = build_model(num_classes=2)
model = model.to(device)

print(model)


# ==================== 7. 定义损失函数和优化器 ====================
criterion = nn.CrossEntropyLoss()   

optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)



# ==================== 8. 训练函数 ====================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
    
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



def validate(model, loader, criterion, device):
    model.eval()    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc




# ==================== 9. 训练循环 ====================
num_epochs = 30             
best_val_acc = 0.0       
train_losses, val_losses = [], []
train_accs, val_accs = [], []

print('开始训练...')
start_time = time.time()

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    scheduler.step()
    
    # 保存历史记录
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    
    # 保存最好的模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f'>>> 保存最佳模型，验证准确率: {val_acc:.4f}')
    
    # 每轮打印信息
    print(f'Epoch [{epoch+1:2d}/{num_epochs}]  '
          f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
          f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

total_time = time.time() - start_time
print(f'训练完成！总用时: {total_time:.2f} 秒')
print(f'最佳验证准确率: {best_val_acc:.4f}')


# ==================== 10. 绘制训练曲线 ====================
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accs, label='Train Acc')
plt.plot(range(1, num_epochs+1), val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.tight_layout()
plt.savefig('./outputs/training_curves.png', dpi=300)
plt.show()