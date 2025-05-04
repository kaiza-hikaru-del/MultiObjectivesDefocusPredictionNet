import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from myDataset import SOAFDataset
from myModel import SOAFModel
from torchvision import transforms

from tqdm.rich import tqdm


# 训练参数配置
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--fe_str', type=str, default='mobilenetv4_conv_small')
    parser.add_argument('--resume', action='store_true', help='从最佳检查点恢复训练')
    return parser.parse_args()


# 训练流程封装
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.unsqueeze(1).to(device)  # 保持维度一致
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


# 验证流程封装
def validation_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            labels = labels.unsqueeze(1).to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


# 主训练函数
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])
    val_transform = transforms.ToTensor()
    
    # 数据集加载
    train_set = SOAFDataset(
        root_dir=Path(args.root_dir),
        dataset_type='train',
        transform=train_transform,
        glob_pattern="20x_0.7*.csv"
    )
    val_set = SOAFDataset(
        root_dir=Path(args.root_dir),
        dataset_type='val',
        transform=val_transform,
        glob_pattern="20x_0.7*.csv"
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
    
    # 模型初始化
    model = SOAFModel(model_name=args.fe_str).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 检查点设置
    ckpt_dir = Path(f'.ckpts/{args.fe_str}')
    ckpt_dir.mkdir(exist_ok=True)
    best_ckpt = ckpt_dir / 'ckpt_best.pt'
    
    start_epoch = 0
    best_loss = float('inf')
    print("Model created")
    
    # 恢复训练
    if args.resume and best_ckpt.exists():
        checkpoint = torch.load(best_ckpt)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        print(f'从检查点恢复训练，当前轮数：{start_epoch}，最佳损失：{best_loss:.4f}')
    
    # TensorBoard配置
    with SummaryWriter(f'.runs/{args.fe_str}') as writer:
        for epoch in range(start_epoch, args.epochs):
            start_time = time.time()
            
            # 训练与验证
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = validation_epoch(model, val_loader, criterion, device)
            
            # 学习率记录
            current_lr = optimizer.param_groups[0]['lr']
            
            # TensorBoard记录
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning Rate', current_lr, epoch)
            
            # 检查点保存逻辑
            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)
            
            # 生成检查点文件名
            ckpt_path = ckpt_dir / f'ckpt_{epoch:04d}.pt'
            
            # 按条件保存常规检查点
            save_regular = False
            if epoch < 200 and epoch % 10 == 0:
                save_regular = True
            elif 200 <= epoch and epoch < 400 and epoch % 5 == 0:
                save_regular = True
            elif 400 <= epoch and epoch < 600 and epoch % 3 == 0:
                save_regular = True
            elif epoch >= 600:
                save_regular = True
                
            # 保存常规检查点（独立判断）
            if save_regular:
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'args': vars(args)
                }, ckpt_path)
                print(f"保存常规检查点: {ckpt_path.name}")

            # 保存最佳检查点（独立判断）
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'args': vars(args)
                }, best_ckpt)
                print(f"保存最佳检查点: {best_ckpt.name} (loss={best_loss:.4f})")
            
            # 训练信息输出
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{args.epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Time: {epoch_time:.2f}s')


if __name__ == '__main__':
    main()