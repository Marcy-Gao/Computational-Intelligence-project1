import os
import shutil

source_folder = './data'          
target_folder = './organized_data'

normal_dir = os.path.join(target_folder, 'normal')
potholes_dir = os.path.join(target_folder, 'potholes')
os.makedirs(normal_dir, exist_ok=True)
os.makedirs(potholes_dir, exist_ok=True)

print('开始整理图片...')

for filename in os.listdir(source_folder):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue 
    
    src_path = os.path.join(source_folder, filename)
    
    if 'normal' in filename.lower():
        dst_path = os.path.join(normal_dir, filename)
        shutil.copy2(src_path, dst_path) 
        print(f'复制 {filename} -> normal')
    else:
        dst_path = os.path.join(potholes_dir, filename)
        shutil.copy2(src_path, dst_path)   
        print(f'复制 {filename} -> potholes')

print('整理完成！')
print(f'正常道路图片数量: {len(os.listdir(normal_dir))}')
print(f'坑洼道路图片数量: {len(os.listdir(potholes_dir))}')