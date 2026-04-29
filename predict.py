import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd

# ==================== 配置 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

test_folder = './test_data'
model_path = './models/best_model.pth'
output_csv = './outputs/test_result.csv'

os.makedirs('./outputs', exist_ok=True)

# ==================== 加载模型 ====================
def load_model(model_path, num_classes=2):
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

print('加载模型...')
model = load_model(model_path)
print('模型加载完成！')

# ==================== 图像预处理 ====================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== 预测 ====================
results = []
count = 0

print('开始预测...')
for root, dirs, files in os.walk(test_folder):
    if '__MACOSX' in root:
        continue
    for filename in files:
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(root, filename)
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                label = 'normal' if predicted.item() == 0 else 'potholes'
            results.append({'filename': filename, 'label': label})
            count += 1
            if count % 100 == 0:
                print(f'已处理 {count} 张图片...')
        except Exception as e:
            print(f'处理 {filename} 时出错: {e}')

print(f'预测完成，共处理 {count} 张图片。')

# ==================== 保存结果 ====================
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f'结果已保存至: {output_csv}')
print('前5行预览:')
print(df.head())