import os

# 指定包含图片的文件夹路径
folder_path = r'D:\\桌面\\机器学习\\CNN图片分类实验\\CNN图片分类实验\\data\\cifar10augmented_cifar-10\\train\\truck'

# 获取文件夹中所有的 png 文件
png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

# 遍历文件列表，重命名每一个文件
for filename in png_files:
    # 解析旧文件名中的编号
    old_num = int(filename.split('.')[0])  # 假设文件名格式为 "1001.png"
    # 计算新的编号
    new_num = old_num - 500
    # 构建新文件名
    new_filename = f"{new_num:04d}.png"  # 生成格式如 "0501.png"
    # 构建完整的旧文件路径和新文件路径
    old_file_path = os.path.join(folder_path, filename)
    new_file_path = os.path.join(folder_path, new_filename)
    # 重命名文件
    os.rename(old_file_path, new_file_path)

print("图片重命名完成!")
