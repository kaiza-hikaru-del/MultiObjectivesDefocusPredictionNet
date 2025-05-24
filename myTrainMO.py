import os
import argparse
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from myDataset import MOAFDataset
from myModel import MOAFModel
from torchvision import transforms
from tqdm.rich import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler


# 训练参数配置
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--dataset_choice', type=str, default="all", help='数据集筛选条件，可选: all, <magnification>_<NA>')
    parser.add_argument('--label_choice', type=str, default="dof_score", help='数据集标签形式，可选：dof_score, defocus_distance')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--fe_str', type=str, default='mobilenetv4_conv_small')
    parser.add_argument('--fusion_mode', type=str, default='film', help='特征融合模式，可选: no_fusion, add, concat, film, gating')
    parser.add_argument('--device_ids', type=str, default='0', help='使用的GPU编号，例如 0,1,2 或留空使用CPU')
    parser.add_argument('--resume', action='store_true', help='从最佳检查点恢复训练')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='预热阶段的 epoch 数')
    parser.add_argument('--eta_min', type=float, default=1e-6, help='余弦退火的最小学习率')
    return parser.parse_args()


# 训练流程封装
# 修改函数定义，添加 scheduler 参数
def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for images, aux_inputs, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        aux_inputs = aux_inputs.to(device)
        labels = labels.unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images, aux_inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # 每个 batch 更新学习率

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


# 验证流程封装
def validation_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, aux_inputs, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            aux_inputs = aux_inputs.to(device)
            labels = labels.unsqueeze(1).to(device)

            outputs = model(images, aux_inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


# 主训练函数
def main():
    args = parse_args()

    # === 多 GPU 配置 ===
    # 解析 device_ids 参数
    device_ids = [int(i) for i in args.device_ids.split(',')] if ',' in args.device_ids else [int(args.device_ids)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}, GPU IDs: {device_ids}")

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    val_transform = transforms.ToTensor()

    # 数据集加载
    glob_pattern = "*.csv" if args.dataset_choice == "all" else f"{args.dataset_choice}*.csv"
    print(f"glob_pattern: {glob_pattern}")
    train_set = MOAFDataset(
        dataset_dir=Path(args.dataset_dir),
        dataset_type='train',
        transform=train_transform,
        glob_pattern=glob_pattern,
        label_choice=args.label_choice
    )
    val_set = MOAFDataset(
        dataset_dir=Path(args.dataset_dir),
        dataset_type='val',
        transform=val_transform,
        glob_pattern=glob_pattern,
        label_choice=args.label_choice
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    print("Data loaded")

    # === 模型初始化 ===
    model = MOAFModel(model_name=args.fe_str, fusion_mode=args.fusion_mode).to(device)

    # === 多 GPU 并行训练 ===
    if len(device_ids) > 1 and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {len(device_ids)} GPUs")
        model = nn.DataParallel(model, device_ids=device_ids)

    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 计算预热和余弦退火的步数
    warmup_steps = args.warmup_epochs * len(train_loader)
    total_steps = args.epochs * len(train_loader)
    cosine_steps = total_steps - warmup_steps

    # 预热调度器（线性增加学习率）
    warmup_scheduler = LinearLR(optimizer, 
                                start_factor=0.1, 
                                end_factor=1.0, 
                                total_iters=warmup_steps)

    # 余弦退火调度器
    cosine_scheduler = CosineAnnealingLR(optimizer, 
                                        T_max=cosine_steps, 
                                        eta_min=args.eta_min)

    # 组合调度器
    scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

    # === 检查点设置 ===
    ckpt_dir = Path(f'.ckpts/MO-{args.fe_str}-{args.fusion_mode}-{args.dataset_choice}-{args.label_choice}')
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    best_ckpt = ckpt_dir / 'ckpt_best.pt'

    start_epoch = 0
    best_loss = float('inf')
    print("Model created")

    # === 恢复训练 ===
    # 恢复训练时加载 scheduler 状态
    if args.resume and best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])  # 新增
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

    # === TensorBoard 配置 ===
    with SummaryWriter(f'.runs/MO-{args.fe_str}-{args.fusion_mode}-{args.dataset_choice}-{args.label_choice}') as writer:
        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()

            # 训练与验证
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss = validation_epoch(model, val_loader, criterion, device)

            # 学习率记录
            current_lr = optimizer.param_groups[0]['lr']

            # 生成检查点文件名
            ckpt_path = ckpt_dir / f'ckpt_{epoch:04d}.pt'

            # 按条件保存常规检查点
            save_regular = False
            if epoch < 200 and epoch % 10 == 0:
                save_regular = True
            elif 200 <= epoch < 400 and epoch % 5 == 0:
                save_regular = True
            elif 400 <= epoch < 600 and epoch % 3 == 0:
                save_regular = True
            elif epoch >= 600:
                save_regular = True

            # 保存常规检查点（独立判断）
            if save_regular:
                # 保存时不带 DataParallel 的 module 前缀
                save_model = model.module if isinstance(model, nn.DataParallel) else model
                # 保存检查点时添加 scheduler 状态
                torch.save({
                    'epoch': epoch,
                    'model': save_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),  # 新增
                    'best_loss': best_loss,
                    'args': vars(args)
                }, ckpt_path)
                print(f"保存常规检查点: {ckpt_path.name}")

            # 保存最佳检查点（独立判断）
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                save_model = model.module if isinstance(model, nn.DataParallel) else model
                # 保存检查点时添加 scheduler 状态
                torch.save({
                    'epoch': epoch,
                    'model': save_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),  # 新增
                    'best_loss': best_loss,
                    'args': vars(args)
                }, best_ckpt)
                print(f"保存最佳检查点: {best_ckpt.name} (loss={best_loss:.4f})")

            # 训练信息输出
            epoch_time = time.time() - start_time
            formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f'Epoch {epoch+1}/{args.epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Time: {epoch_time:.2f}s | '
                  f'Timestamp: {formatted_time}')
            
            # TensorBoard记录
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Other/Learning Rate', current_lr, epoch)
            writer.add_scalar('Other/Epoch Time Cost', epoch_time, epoch)


if __name__ == '__main__':
    main()