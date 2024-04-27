import os
import numpy as np
from PIL import Image
import albumentations as A

# 定义单独的数据增强操作
transformations = {
    "horizontal_flip": A.HorizontalFlip(p=1.0),
    "vertical_flip": A.VerticalFlip(p=1.0),
    "sharpen": A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
    "blur": A.Blur(blur_limit=7, p=1.0),
    "random_brightness_contrast": A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
    "color_jitter": A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1.0),
    "gaussian_blur": A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    "clahe": A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    "gauss_noise": A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    "rotate": A.Rotate(limit=45, p=1.0),
    "image_compression": A.ImageCompression(quality_lower=60, quality_upper=100, p=1.0),
    "random_resized_crop": A.RandomResizedCrop(height=32, width=32, scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
    "random_rain": A.RandomRain(p=1.0),
    "random_snow": A.RandomSnow(p=1.0),
    "random_fog": A.RandomFog(p=1.0)
}

# 指定输入输出文件夹
base_input_folder = 'D:\\桌面\\机器学习\\CNN图片分类实验\\CNN图片分类实验\\data\\cifar10\\train'  # 替换为你的输入文件夹路径
base_output_folder = 'D:\\桌面\\机器学习\\CNN图片分类实验\\CNN图片分类实验\\data\\cifar10augmented_cifar-10\\train'  # 替换为你希望保存增强图片的文件夹路径

# 创建类别文件夹
categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for category in categories:
    category_folder = os.path.join(base_output_folder, category)
    if not os.path.exists(category_folder):
        os.makedirs(category_folder)

# 应用变换并保存图片
for category in categories:
    category_input_folder = os.path.join(base_input_folder, category)
    category_output_folder = os.path.join(base_output_folder, category)
    img_list = os.listdir(category_input_folder)
    
    for index, (transform_name, transform) in enumerate(transformations.items(), start=1):
        offset = 500 * index
        for i, img_filename in enumerate(img_list):
            if img_filename.endswith('.png'):
                img_path = os.path.join(category_input_folder, img_filename)
                img = Image.open(img_path)
                img_np = np.array(img)
                
                transformed = transform(image=img_np)
                img_transformed = Image.fromarray(transformed['image'])
                
                # 构建新的文件名
                new_filename = f"{i+1+offset:04d}.png"
                output_path = os.path.join(category_output_folder, new_filename)
                img_transformed.save(output_path)

print("数据增强完成!")


