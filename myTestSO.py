import argparse
import time
import pandas as pd
from pathlib import Path
import numpy as np
import torch
import onnxruntime
from torch.utils.data import DataLoader
from myDataset import SOAFDataset
from myModel import SOAFModel
from torchvision import transforms

from tqdm.rich import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--fe_str', type=str, default='mobilenetv4_conv_small')
    return parser.parse_args()


def convert_to_onnx(pt_model, device, save_model):
    save_dir = Path(".onnx")
    save_dir.mkdir(exist_ok=True)
    onnx_path = save_dir / f"{save_model}_best.onnx"
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    torch.onnx.export(
        pt_model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=13
    )
    return onnx_path


def test_pytorch_model(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            outputs = model(images).cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_preds.extend(outputs.squeeze())
    
    return all_labels, all_preds


def test_onnx_model(onnx_path, dataset):
    session = onnxruntime.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    print(f"ONNX session providers: {session.get_providers()}")
    print(f"ONNX session provider options: {session.get_provider_options()}")
    all_preds = []
    time_costs = []
    
    for img, label in tqdm(dataset):
        img_tensor = img.unsqueeze(0).numpy()  # Add batch dimension
        
        start_time = time.perf_counter()
        pred = session.run(
            ["output"],
            {"input": img_tensor}
        )[0][0][0]
        time_cost = time.perf_counter() - start_time
        
        all_preds.append(pred)
        time_costs.append(time_cost)
    
    return all_preds, time_costs


def save_results(results, save_result):
    save_dir = Path(".results")
    save_dir.mkdir(exist_ok=True)
    
    # 创建DataFrame并指定列名
    df = pd.DataFrame(results, columns=[
        'image_path', 
        'label', 
        'pt_pred', 
        'onnx_pred', 
        'onnx_time_cost'
    ])
    
    # 添加路径处理（可选）
    df['image_path'] = df['image_path'].astype(str)
    
    # 优化数据类型
    df = df.astype({
        'label': 'float32',
        'pt_pred': 'float32',
        'onnx_pred': 'float32',
        'onnx_time_cost': 'float64'
    })
    
    csv_path = save_dir / f"{save_result}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"测试结果已保存至: {csv_path}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载测试集
    test_set = SOAFDataset(
        root_dir=Path(args.root_dir),
        dataset_type='test',
        transform=transforms.ToTensor(),
        glob_pattern="20x_0.7*.csv"
    )
    
    # 创建PyTorch数据加载器
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 初始化模型并加载最佳检查点
    model = SOAFModel(model_name=args.fe_str).to(device)
    ckpt = torch.load(Path('.ckpts') / args.fe_str / 'ckpt_best.pt')
    model.load_state_dict(ckpt['model'])
    
    # 转换ONNX模型
    onnx_path = convert_to_onnx(model, device)
    print(f"ONNX模型已保存至: {onnx_path}")
    
    # 测试PyTorch模型
    labels, pt_preds = test_pytorch_model(model, test_loader, device)
    print("PyTorch模型测试结束")
    
    # 测试ONNX模型
    onnx_preds, onnx_times = test_onnx_model(onnx_path, test_set)
    print("ONNX模型测试结束")
    
    # 收集结果
    results = []
    for i in range(len(test_set)):
        results.append((
            str(test_set.image_paths[i]),  # 原始路径已转换为绝对路径
            labels[i],
            pt_preds[i],
            onnx_preds[i],
            onnx_times[i]
        ))
    
    # 保存结果
    save_results(results)

if __name__ == '__main__':
    main()