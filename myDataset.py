import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from PIL import Image
from torchvision import transforms

import time


# 多物镜数据集
class MOAFDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_type: str,
        transform = None,
        glob_pattern: str = "*.csv",
        label_choice: str = "dof_score"
    ):
        super().__init__()
        
        # 参数验证
        if dataset_type not in {'train', 'val', 'test'}:
            raise ValueError("dataset_type 必须是 'train', 'val' 或 'test'")
        if not isinstance(dataset_dir, Path):
            raise TypeError("dataset_dir 必须是 pathlib.Path 对象")
        if '.csv' not in glob_pattern:
            raise ValueError("glob_pattern 必须包含 '.csv' 扩展名")

        # 构建数据集路径
        self.dataset_dir = dataset_dir
        self.dataset_dir = dataset_dir / f".{dataset_type}"
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录 {self.dataset_dir} 不存在")

        # 并行加载所有CSV文件
        csv_files = list(self.dataset_dir.glob(glob_pattern))
        if not csv_files:
            raise FileNotFoundError(f"未找到匹配 {glob_pattern} 的CSV文件")

        # 使用pd.concat一次性合并所有DataFrame
        dfs = [pd.read_csv(f) for f in csv_files]
        combined_df = pd.concat(dfs, ignore_index=True)

        # 列名验证（合并后只需验证一次）
        required_cols = {
            'image_path', 'magnification', 'NA',
            'defocus_dof_label', 'defocus_label'  # 两个标签列都需存在
        }
        missing_cols = required_cols - set(combined_df.columns)
        if missing_cols:
            raise ValueError(f"CSV文件缺少必要列: {missing_cols}")

        # 向量化处理路径并将反斜杠全部替换为正斜杠
        combined_df['abs_image_path'] = dataset_dir / combined_df['image_path'].str.replace('\\', '/', regex=False)
        
        # 转换为绝对路径列表（替代字典存储）
        self.image_paths = combined_df['abs_image_path'].tolist()
        
        # 预处理数值特征（提前转换为张量）
        self.features = torch.tensor(
            combined_df[['magnification', 'NA']].values.astype(float),
            dtype=torch.float32
        )
        
        # 预处理标签并根据 label_dof_score 选择使用离焦距离还是离焦景深分数
        label_col = 'defocus_dof_label' if label_choice == "dof_score" else 'defocus_label'
        self.labels = torch.tensor(
            combined_df[label_col].values.astype(float),
            dtype=torch.float32
        )

        # 图像预处理管道
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # 直接返回预处理的张量
        return img, self.features[idx], self.labels[idx]


# 单物镜数据集
class SOAFDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_type: str,
        transform = None,
        glob_pattern: str = "10x_0.3*.csv"
    ):
        super().__init__()
        
        # 参数验证
        if dataset_type not in {'train', 'val', 'test'}:
            raise ValueError("dataset_type 必须是 'train', 'val' 或 'test'")
        if not isinstance(dataset_dir, Path):
            raise TypeError("dataset_dir 必须是 pathlib.Path 对象")
        if '.csv' not in glob_pattern:
            raise ValueError("glob_pattern 必须包含 '.csv' 扩展名")

        # 构建数据集路径
        self.dataset_dir = dataset_dir
        self.dataset_dir = dataset_dir / f".{dataset_type}"
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"数据集目录 {self.dataset_dir} 不存在")

        # 加载并合并CSV文件
        csv_files = list(self.dataset_dir.glob(glob_pattern))
        if not csv_files:
            raise FileNotFoundError(f"未找到匹配 {glob_pattern} 的CSV文件")

        # 批量合并DataFrame
        dfs = [pd.read_csv(f) for f in csv_files]
        combined_df = pd.concat(dfs, ignore_index=True)

        # 列名验证
        required_cols = {'image_path', 'defocus_label'}
        missing_cols = required_cols - set(combined_df.columns)
        if missing_cols:
            raise ValueError(f"CSV文件缺少必要列: {missing_cols}")

        # 向量化处理路径并将反斜杠全部替换为正斜杠
        combined_df['abs_image_path'] = dataset_dir / combined_df['image_path'].str.replace('\\', '/', regex=False)
        
        # 转换为高效数据结构
        self.image_paths = combined_df['abs_image_path'].tolist()
        self.labels = torch.tensor(
            combined_df['defocus_label'].values.astype(float),
            dtype=torch.float32
        )

        # 图像预处理
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, self.labels[idx]


# 使用示例
if __name__ == "__main__":
    # 初始化配置
    root_path = Path("E:/Datasets/MultiObjectives")
    custom_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    # 创建数据集实例
    start = time.time()
    train_set = SOAFDataset(
        dataset_dir=root_path,
        dataset_type="train",
        transform=custom_transform,
        glob_pattern="10x_0.3*.csv"
    )
    print(f"Create dataset time cost: {time.time() - start:.3f}s")
    
    # 获取样本
    sample_img, sample_label = train_set[0]
    print(f"图像尺寸: {sample_img.shape}")
    print(f"目标标签: {sample_label}")
    print(f"数据集大小：{len(train_set)}")