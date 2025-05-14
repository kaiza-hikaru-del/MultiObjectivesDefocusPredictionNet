import argparse
import time
import pandas as pd
from pathlib import Path
import numpy as np
import torch
import onnxruntime
from torch.utils.data import DataLoader
from myDataset import MOAFDataset
from myModel import MOAFModel
from torchvision import transforms
from tqdm.rich import tqdm


# 参数解析
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='数据集根目录')
    parser.add_argument('--dataset_choice', type=str, default="all", help='数据集筛选条件，可选: all, <magnification>_<NA>')
    parser.add_argument('--label_choice', type=str, default="dof_score", help='数据集标签形式，可选: dof_score, defocus_distance')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--fe_str', type=str, default='mobilenetv4_conv_small')
    parser.add_argument('--fusion_mode', type=str, default='film', help='特征融合模式，可选: no_fusion, add, concat, film, gating')
    return parser.parse_args()


def convert_to_onnx(pt_model, device, fe_str, fusion_mode, dataset_choice, label_choice):
    save_dir = Path(".onnx")
    save_dir.mkdir(exist_ok=True, parents=True)
    onnx_path = save_dir / f"MO_{fe_str}_{fusion_mode}_{dataset_choice}_{label_choice}_best.onnx"
    
    # 创建 dummy 输入（图像 + 辅助输入）
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    dummy_aux = torch.randn(1, 2).to(device)  # 2D 物镜参数 (放大倍率, 数值孔径)
    
    torch.onnx.export(
        pt_model,
        (dummy_image, dummy_aux),
        onnx_path,
        input_names=["image", "aux_input"],
        output_names=["output"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "aux_input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
    )
    return onnx_path


def test_pytorch_model(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, aux_inputs, labels in tqdm(loader):
            images = images.to(device)
            aux_inputs = aux_inputs.to(device)
            
            outputs = model(images, aux_inputs).cpu().numpy()
            
            all_labels.extend(labels.numpy())
            all_preds.extend(outputs.squeeze())
    
    return all_labels, all_preds


def test_onnx_model(onnx_path, dataset, fusion_mode):
    session = onnxruntime.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    print(f"ONNX session providers: {session.get_providers()}")
    print(f"ONNX session provider options: {session.get_provider_options()}")
    all_preds = []
    time_costs = []
    
    for img, aux, label in tqdm(dataset):
        # 添加 batch 维度
        img_tensor = img.unsqueeze(0).numpy()
        aux_tensor = aux.unsqueeze(0).numpy()
        
        start_time = time.perf_counter()
        
        # 处理无融合情况下没有 aux_input
        if fusion_mode == "no_fusion":
            pred = session.run(
            ["output"],
            {"image": img_tensor}
        )[0][0][0]
        else:
            pred = session.run(
                ["output"],
                {"image": img_tensor, "aux_input": aux_tensor}
            )[0][0][0]

        time_cost = time.perf_counter() - start_time
        
        all_preds.append(pred)
        time_costs.append(time_cost)
    
    return all_preds, time_costs


def save_results(results, fe_str, fusion_mode, dataset_choice, label_choice):
    save_dir = Path(".results")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # === 未分组结果 ===
    # 创建DataFrame并指定列名
    df = pd.DataFrame(results, columns=[
        'image_path', 
        'magnification',
        'NA',
        'label', 
        'pt_pred', 
        'onnx_pred', 
        'onnx_time_cost'
    ])
    
    # 添加路径处理
    df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)
    
    # 优化数据类型
    df = df.astype({
        'magnification': 'float32',
        'NA': 'float32',
        'label': 'float32',
        'pt_pred': 'float32',
        'onnx_pred': 'float32',
        'onnx_time_cost': 'float64'
    })

    # 添加基于已有列计算的新列
    df['pt_error'] = (df['label'] - df['pt_pred']).abs()
    df['pt_signed_error'] = df['label'] - df['pt_pred']
    df['pt_direction'] = np.sign(df['label'] * df['pt_pred'])

    df['onnx_error'] = (df['label'] - df['onnx_pred']).abs()
    df['onnx_signed_error'] = df['label'] - df['onnx_pred']
    df['onnx_direction'] = np.sign(df['label'] * df['onnx_pred'])

    # 添加统计信息作为新列
    df['pt_error_mean'] = df['pt_error'].mean()
    df['pt_error_std'] = df['pt_error'].std()

    df['onnx_error_mean'] = df['onnx_error'].mean()
    df['onnx_error_std'] = df['onnx_error'].std()

    df['onnx_time_cost_mean'] = df['onnx_time_cost'].mean()
    df['onnx_time_cost_std'] = df['onnx_time_cost'].std()
    
    csv_path = save_dir / f"MO_{fe_str}_{fusion_mode}_{dataset_choice}_{label_choice}_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"原始测试结果已保存至: {csv_path}")

    # === 分组后结果 ===
    # 使用正则表达式提取关键字段
    pattern = r"^.*?/(\d+)x_(\d+\.\d+)_\w+/sample(\d+)/field(\d+)/roi\d+/(\d+)\.jpg$"
    extracted = df['image_path'].str.extract(pattern, expand=True)

    # 检查提取结果（可选调试）
    if extracted.isnull().any().any():
        print("！警告：部分路径匹配失败，以下为失败示例：")
        failed_paths = df[extracted.isnull().any(axis=1)]['image_path'].head(5)
        for p in failed_paths:
            print(f"  - {p}")

    # 将提取结果添加为新列
    df['magnification'] = pd.to_numeric(extracted[0], errors='coerce')
    df['NA'] = pd.to_numeric(extracted[1], errors='coerce')
    df['sample_no'] = extracted[2].astype(int)
    df['field_no'] = extracted[3].astype(int)
    df['order'] = extracted[4].astype(int)

    # 过滤掉提取失败的行
    df_filtered = df.dropna(subset=['magnification', 'NA', 'sample_no', 'field_no', 'order'])

    # 按提取后的字段进行分组，计算中位数和行数
    grouped_df = df_filtered.groupby(
        ['magnification', 'NA', 'sample_no', 'field_no', 'order'],
        as_index=False
    ).agg(
        label=('label', 'median'),
        pt_pred=('pt_pred', 'median'),
        onnx_pred=('onnx_pred', 'median'),
        onnx_time_cost=('onnx_time_cost', 'median'),
        count=('label', 'size')  # 新增：每组行数统计
    )

    # 调整列顺序，将 count 插入到 NA 后面
    columns_order = [
        'magnification', 'NA', 'count', 'sample_no', 'field_no', 'order',
        'label', 'pt_pred', 'onnx_pred', 'onnx_time_cost'
    ]
    grouped_df = grouped_df[columns_order]

    # 添加基于已有列计算的新列
    grouped_df['pt_error'] = (grouped_df['label'] - grouped_df['pt_pred']).abs()
    grouped_df['pt_signed_error'] = grouped_df['label'] - grouped_df['pt_pred']
    grouped_df['pt_direction'] = np.sign(grouped_df['label'] * grouped_df['pt_pred'])

    grouped_df['onnx_error'] = (grouped_df['label'] - grouped_df['onnx_pred']).abs()
    grouped_df['onnx_signed_error'] = grouped_df['label'] - grouped_df['onnx_pred']
    grouped_df['onnx_direction'] = np.sign(grouped_df['label'] * grouped_df['onnx_pred'])

    # 添加统计信息作为新列
    grouped_df['pt_error_mean'] = grouped_df['pt_error'].mean()
    grouped_df['pt_error_std'] = grouped_df['pt_error'].std()

    grouped_df['onnx_error_mean'] = grouped_df['onnx_error'].mean()
    grouped_df['onnx_error_std'] = grouped_df['onnx_error'].std()

    grouped_df['onnx_time_cost_mean'] = grouped_df['onnx_time_cost'].mean()
    grouped_df['onnx_time_cost_std'] = grouped_df['onnx_time_cost'].std()

    # 保存处理结果
    csv_grouped_path = save_dir / f"MO_grouped_{fe_str}_{fusion_mode}_{dataset_choice}_{label_choice}_results.csv"
    grouped_df.to_csv(csv_grouped_path, index=False)
    print(f"分组测试结果已保存至: {csv_grouped_path}")


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    glob_pattern = "*.csv" if args.dataset_choice == "all" else f"{args.dataset_choice}*.csv"
    
    # 加载测试集（包含辅助输入）
    test_set = MOAFDataset(
        dataset_dir=Path(args.dataset_dir),
        dataset_type='test',
        transform=transforms.ToTensor(),
        glob_pattern=glob_pattern,
        label_choice=args.label_choice
    )
    
    # 创建PyTorch数据加载器
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 初始化模型并加载最佳检查点
    model = MOAFModel(model_name=args.fe_str, fusion_mode=args.fusion_mode).to(device)
    ckpt = torch.load(
        Path('.ckpts') / f"MO_{args.fe_str}_{args.fusion_mode}_{args.dataset_choice}_{args.label_choice}" / 'ckpt_best.pt',
        map_location=device
    )
    model.load_state_dict(ckpt['model'])
    
    # 转换ONNX模型
    onnx_path = convert_to_onnx(model, device, args.fe_str, args.fusion_mode, args.dataset_choice, args.label_choice)
    print(f"ONNX模型已保存至: {onnx_path}")
    
    # 测试PyTorch模型
    labels, pt_preds = test_pytorch_model(model, test_loader, device)
    print("PyTorch模型测试结束")
    
    # 测试ONNX模型
    onnx_preds, onnx_times = test_onnx_model(onnx_path, test_set, args.fusion_mode)
    print("ONNX模型测试结束")
    
    # 收集结果
    results = []
    for i in range(len(test_set)):
        results.append((
            str(test_set.image_paths[i]),   # 原始路径已转换为绝对路径
            test_set.features[i][0],        # 放大倍率
            test_set.features[i][1],        # 数值孔径
            labels[i],
            pt_preds[i],
            onnx_preds[i],
            onnx_times[i]
        ))
    
    # 保存结果

    save_results(results, args.fe_str, args.fusion_mode, args.dataset_choice, args.label_choice)

if __name__ == '__main__':
    main()