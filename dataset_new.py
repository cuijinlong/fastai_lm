import os
import shutil
import random
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
from PIL import Image, ImageDraw, ImageFont
import numpy as np


class MedicalDatasetSplitter:
    """
    医疗影像数据集分割工具类
    支持生成包含分类标签和生化指标的多模态数据集
    """

    def __init__(self, random_seed=42):
        """
        初始化

        Args:
            random_seed: 随机种子，确保结果可重现
        """
        self.random_seed = random_seed
        random.seed(random_seed)

    def split_dataset(self, source_dir, output_dir,
                      train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                      extensions=None, copy_files=True,
                      excel_path=None, image_col='image_name', label_col=None):
        """
        分割数据集并生成CSV元数据文件

        Args:
            source_dir: 原始数据目录，包含按类别分组的子文件夹
            output_dir: 输出目录，将创建train/val/test子目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            extensions: 支持的图片扩展名，默认为常见图片格式
            copy_files: 是否复制文件（True）还是移动文件（False）
            excel_path: Excel文件路径，包含生化指标数据（可选）
            image_col: Excel中图片文件名的列名
            label_col: Excel中标签列的列名（如果与文件夹分类不一致时使用）

        Returns:
            dict: 包含分割统计信息的字典
        """

        # 检查比例总和是否为1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-5:
            raise ValueError(f"比例总和应为1.0，当前为{total_ratio}")

        # 设置默认图片扩展名
        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.dcm'}

        # 创建输出目录结构
        output_path = Path(output_dir)
        train_dir = output_path / 'train'
        val_dir = output_path / 'val'
        test_dir = output_path / 'test'

        for dir_path in [train_dir, val_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 获取所有类别
        source_path = Path(source_dir)
        classes = [d.name for d in source_path.iterdir()
                   if d.is_dir() and not d.name.startswith('.')]

        if not classes:
            raise ValueError(f"在 {source_dir} 中未找到任何类别文件夹")

        print(f"找到 {len(classes)} 个类别: {classes}")

        # 读取Excel数据（如果提供）
        excel_data = None
        if excel_path:
            if not Path(excel_path).exists():
                warnings.warn(f"Excel文件 {excel_path} 不存在，将仅使用文件夹分类信息")
            else:
                try:
                    excel_data = pd.read_excel(excel_path)
                    print(f"成功读取Excel文件，包含 {len(excel_data)} 行数据")
                    print(f"Excel列名: {list(excel_data.columns)}")
                except Exception as e:
                    warnings.warn(f"读取Excel文件失败: {e}，将仅使用文件夹分类信息")
                    excel_data = None

        # 统计数据
        stats = {
            'total_images': 0,
            'class_distribution': {},
            'split_distribution': {'train': 0, 'val': 0, 'test': 0},
            'excel_metadata_used': excel_data is not None
        }

        # 存储所有文件的分割信息用于生成CSV
        all_files_info = {
            'train': [],
            'val': [],
            'test': []
        }

        # 为每个类别创建输出目录并分割数据
        for class_name in classes:
            class_path = source_path / class_name

            # 在输出目录中创建类别子文件夹
            for dir_path in [train_dir, val_dir, test_dir]:
                (dir_path / class_name).mkdir(parents=True, exist_ok=True)

            # 获取该类别的所有图片文件
            image_files = []
            for ext in extensions:
                image_files.extend(list(class_path.glob(f'*{ext}')))
                image_files.extend(list(class_path.glob(f'*{ext.upper()}')))

            # 过滤出文件（排除目录）
            image_files = [f for f in image_files if f.is_file()]

            if not image_files:
                print(f"警告: 在 {class_name} 中未找到图片文件")
                continue

            stats['class_distribution'][class_name] = len(image_files)
            stats['total_images'] += len(image_files)

            # 随机打乱文件列表
            random.shuffle(image_files)

            # 计算分割点
            n_total = len(image_files)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            # 分割数据集
            train_files = image_files[:n_train]
            val_files = image_files[n_train:n_train + n_val]
            test_files = image_files[n_train + n_val:]

            # 处理训练集文件
            train_info = self._process_files(train_files, train_dir / class_name,
                                             class_name, copy_files, excel_data, image_col, label_col)
            all_files_info['train'].extend(train_info)

            # 处理验证集文件
            val_info = self._process_files(val_files, val_dir / class_name,
                                           class_name, copy_files, excel_data, image_col, label_col)
            all_files_info['val'].extend(val_info)

            # 处理测试集文件
            test_info = self._process_files(test_files, test_dir / class_name,
                                            class_name, copy_files, excel_data, image_col, label_col)
            all_files_info['test'].extend(test_info)

            # 更新统计
            stats['split_distribution']['train'] += len(train_files)
            stats['split_distribution']['val'] += len(val_files)
            stats['split_distribution']['test'] += len(test_files)

            print(f"类别 {class_name}: {len(image_files)} 张图片 -> "
                  f"训练: {len(train_files)}, 验证: {len(val_files)}, 测试: {len(test_files)}")

        # 生成CSV文件
        self._generate_csv_files(all_files_info, output_path, excel_data is not None)

        return stats

    def _process_files(self, file_list, target_dir, class_name, copy_files, excel_data, image_col, label_col):
        """
        处理文件并返回文件信息

        Args:
            file_list: 文件列表
            target_dir: 目标目录
            class_name: 类别名称
            copy_files: 是否复制文件
            excel_data: Excel数据
            image_col: 图片列名
            label_col: 标签列名

        Returns:
            list: 文件信息列表
        """
        files_info = []

        for file_path in file_list:
            target_path = target_dir / file_path.name

            # 处理文件名冲突
            counter = 1
            while target_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                target_path = target_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            if copy_files:
                shutil.copy2(file_path, target_path)
            else:
                shutil.move(str(file_path), str(target_path))

            # 构建文件信息
            file_info = {
                'image_path': str(target_path.relative_to(target_dir.parent.parent)),  # 相对路径
                'filename': target_path.name,
                'label': class_name
            }

            # 如果提供了Excel数据，添加生化指标
            if excel_data is not None:
                # 在Excel中查找匹配的行
                matched_rows = excel_data[excel_data[image_col] == file_path.name]

                if not matched_rows.empty:
                    # 取第一行匹配的数据
                    row = matched_rows.iloc[0]

                    # 如果指定了标签列且与文件夹分类不同，使用Excel中的标签
                    if label_col and label_col in excel_data.columns:
                        file_info['label'] = row[label_col]

                    # 添加所有其他列作为生化指标
                    for col in excel_data.columns:
                        if col not in [image_col, label_col]:
                            file_info[col] = row[col]
                else:
                    # 如果没有匹配，标记缺失数据
                    file_info['excel_data_missing'] = True
                    for col in excel_data.columns:
                        if col not in [image_col, label_col]:
                            file_info[col] = None

            files_info.append(file_info)

        return files_info

    def _generate_csv_files(self, all_files_info, output_path, has_excel_data):
        """
        为每个分割生成CSV文件

        Args:
            all_files_info: 所有文件信息
            output_path: 输出路径
            has_excel_data: 是否使用了Excel数据
        """
        for split_name, files_info in all_files_info.items():
            if files_info:  # 确保有数据
                df = pd.DataFrame(files_info)

                # 重新排列列，将重要列放在前面
                preferred_order = ['image_path', 'filename', 'label']
                other_cols = [col for col in df.columns if col not in preferred_order]
                df = df[preferred_order + other_cols]

                csv_path = output_path / f'{split_name}_metadata.csv'
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"生成 {split_name} CSV文件: {csv_path}")

    def create_dataset_info(self, output_dir, stats):
        """
        创建数据集信息文件

        Args:
            output_dir: 输出目录
            stats: 统计信息字典
        """
        info_file = Path(output_dir) / 'dataset_info.txt'

        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("医疗影像数据集信息\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"总图片数量: {stats['total_images']}\n")
            f.write(f"使用Excel元数据: {'是' if stats['excel_metadata_used'] else '否'}\n\n")

            f.write("类别分布:\n")
            for class_name, count in stats['class_distribution'].items():
                f.write(f"  {class_name}: {count} 张图片\n")

            f.write(f"\n数据集分割:\n")
            f.write(f"  训练集: {stats['split_distribution']['train']} 张图片\n")
            f.write(f"  验证集: {stats['split_distribution']['val']} 张图片\n")
            f.write(f"  测试集: {stats['split_distribution']['test']} 张图片\n")

            f.write(f"\n生成的文件:\n")
            f.write(f"  train_metadata.csv - 训练集元数据\n")
            f.write(f"  val_metadata.csv - 验证集元数据\n")
            f.write(f"  test_metadata.csv - 测试集元数据\n")

            f.write(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


def create_sample_images():
    """
    创建模拟的医疗图像数据
    生成三个文件夹(1,2,3)，每个文件夹包含一些模拟的医疗图像
    混合使用JPG和PNG格式
    """
    base_dir = Path("sample_medical_images")

    # 创建三个类别文件夹
    categories = ['1', '2', '3']
    images_per_category = 30  # 每个类别生成30张图片

    # 为每个类别生成图片
    for category in categories:
        category_dir = base_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        print(f"为类别 {category} 生成 {images_per_category} 张图片...")

        for i in range(images_per_category):
            # 创建一张模拟的医疗图像
            img = Image.new('RGB', (256, 256), color=(random.randint(200, 255),
                                                      random.randint(200, 255),
                                                      random.randint(200, 255)))
            draw = ImageDraw.Draw(img)

            # 添加一些随机形状模拟医疗图像特征
            for _ in range(random.randint(3, 8)):
                x1, y1 = random.randint(0, 200), random.randint(0, 200)
                x2, y2 = x1 + random.randint(20, 50), y1 + random.randint(20, 50)
                color = (random.randint(0, 150), random.randint(0, 150), random.randint(0, 150))

                if random.choice([True, False]):
                    draw.ellipse([x1, y1, x2, y2], fill=color)
                else:
                    draw.rectangle([x1, y1, x2, y2], fill=color)

            # 添加类别标签文本
            draw.text((10, 10), f"Category {category}", fill=(0, 0, 0))

            # 随机选择保存为JPG或PNG格式
            filename = f"{category}_{i + 1:03d}"
            if i % 3 == 0:  # 约1/3的图片保存为PNG
                filename += ".png"
                img.save(category_dir / filename)
            else:  # 其他保存为JPG
                filename += ".jpg"
                # 对于JPG，使用高质量保存
                img.save(category_dir / filename, "JPEG", quality=95)

    print(f"模拟图像生成完成，保存在 {base_dir} 目录")

    # 显示每个类别的文件类型统计
    for category in categories:
        category_dir = base_dir / category
        jpg_count = len(list(category_dir.glob("*.jpg")))
        png_count = len(list(category_dir.glob("*.png")))
        print(f"类别 {category}: {jpg_count} 张JPG, {png_count} 张PNG")

    return base_dir


def create_sample_excel(image_dir):
    """
    创建模拟的Excel数据，包含生化指标
    处理JPG和PNG格式的图片
    """
    # 收集所有生成的图片文件名（包括JPG和PNG）
    all_images = []
    for category in ['1', '2', '3']:
        category_dir = Path(image_dir) / category

        # 使用glob同时查找JPG和PNG文件
        for img_file in category_dir.glob("*.*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                all_images.append(img_file.name)

    # 为每张图片生成模拟的生化指标数据
    data = []
    for img_name in all_images:
        # 从文件名提取类别
        category = img_name.split('_')[0]

        # 生成模拟的生化指标
        row = {
            'image_name': img_name,
            'diagnosis': f"诊断{category}",  # 与文件夹分类可能不同的标签
            'patient_age': random.randint(20, 80),
            'patient_gender': random.choice(['M', 'F']),
            'wbc_count': round(random.uniform(4.0, 15.0), 1),  # 白细胞计数
            'crp_level': round(random.uniform(0.5, 50.0), 1),  # C反应蛋白
            'temperature': round(random.uniform(36.5, 39.5), 1),  # 体温
            'blood_pressure_systolic': random.randint(100, 160),  # 收缩压
            'blood_pressure_diastolic': random.randint(60, 100),  # 舒张压
        }
        data.append(row)

    # 创建DataFrame并保存为Excel
    df = pd.DataFrame(data)
    excel_path = Path(image_dir) / "medical_data_sample.xlsx"
    df.to_excel(excel_path, index=False)

    print(f"模拟Excel数据生成完成，包含 {len(df)} 行数据")
    print(f"Excel文件保存为: {excel_path}")

    # 显示前几行数据
    print("\nExcel数据样例:")
    print(df.head())

    # 显示文件格式分布
    jpg_count = len([name for name in all_images if name.lower().endswith(('.jpg', '.jpeg'))])
    png_count = len([name for name in all_images if name.lower().endswith('.png')])
    print(f"\n图片格式分布: {jpg_count} 张JPG, {png_count} 张PNG")

    return excel_path


def demo_without_excel():
    """
    演示无Excel数据的情况
    """
    print("\n" + "=" * 50)
    print("演示: 无Excel数据的情况")
    print("=" * 50)

    # 创建分割器实例
    splitter = MedicalDatasetSplitter(random_seed=42)

    # 分割数据集（不使用Excel）
    stats = splitter.split_dataset(
        source_dir="sample_medical_images",
        output_dir="output_dataset_basic",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        copy_files=True
    )

    # 创建数据集信息文件
    splitter.create_dataset_info("output_dataset_basic", stats)

    print(f"\n无Excel数据情况处理完成!")
    print(f"输出目录: output_dataset_basic")

    # 显示生成的CSV文件内容
    train_csv = pd.read_csv("output_dataset_basic/train_metadata.csv")
    print(f"\n训练集CSV前5行:")
    print(train_csv.head())

    return stats


def demo_with_excel():
    """
    演示有Excel数据的情况
    """
    print("\n" + "=" * 50)
    print("演示: 有Excel数据的情况")
    print("=" * 50)

    # 创建分割器实例
    splitter = MedicalDatasetSplitter(random_seed=42)

    # 分割数据集（使用Excel）
    stats = splitter.split_dataset(
        source_dir="sample_medical_images",
        output_dir="output_dataset_multimodal",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        copy_files=True,
        excel_path="medical_data_sample.xlsx",
        image_col="image_name",
        label_col="diagnosis"  # 使用Excel中的诊断标签，而不是文件夹名称
    )

    # 创建数据集信息文件
    splitter.create_dataset_info("output_dataset_multimodal", stats)

    print(f"\n有Excel数据情况处理完成!")
    print(f"输出目录: output_dataset_multimodal")

    # 显示生成的CSV文件内容
    train_csv = pd.read_csv("output_dataset_multimodal/train_metadata.csv")
    print(f"\n训练集CSV前5行:")
    print(train_csv.head())

    return stats


def main():
    """
    主函数：完整演示流程
    """
    print("开始创建模拟医疗数据集演示...")

    # 步骤1: 创建模拟图像数据（混合JPG和PNG）
    # image_dir = create_sample_images()
    image_dir = Path('/Users/cuijinlong/Documents/datasets/pifubing/optional_image')
    # 步骤2: 创建模拟Excel数据
    excel_path = create_sample_excel(image_dir)

    # # 步骤3: 演示无Excel数据的情况
    # stats_basic = demo_without_excel()
    #
    # # 步骤4: 演示有Excel数据的情况
    # stats_multimodal = demo_with_excel()
    #
    # # 步骤5: 比较两种情况的输出
    # print("\n" + "=" * 50)
    # print("结果比较")
    # print("=" * 50)
    #
    # print(f"无Excel数据情况:")
    # print(f"  - 总图片数: {stats_basic['total_images']}")
    # basic_csv_cols = len(pd.read_csv('output_dataset_basic/train_metadata.csv').columns)
    # print(f"  - CSV列数: {basic_csv_cols}")
    #
    # print(f"\n有Excel数据情况:")
    # print(f"  - 总图片数: {stats_multimodal['total_images']}")
    # multimodal_csv_cols = len(pd.read_csv('output_dataset_multimodal/train_metadata.csv').columns)
    # print(f"  - CSV列数: {multimodal_csv_cols}")
    # print(f"  - 包含生化指标: {stats_multimodal['excel_metadata_used']}")
    #
    # print(f"\n演示完成!")
    # print(f"生成的文件:")
    # print(f"  - 原始图像: sample_medical_images/")
    # print(f"  - Excel数据: medical_data_sample.xlsx")
    # print(f"  - 无Excel输出: output_dataset_basic/")
    # print(f"  - 有Excel输出: output_dataset_multimodal/")


if __name__ == "__main__":
    main()