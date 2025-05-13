import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Process and aggregate CSV data.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to input CSV file.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to output CSV file.")
    args = parser.parse_args()

    # 参数校验
    if not args.input_path.endswith('.csv'):
        raise ValueError("Input path must end with '.csv'")
    if not args.output_path.endswith('.csv'):
        raise ValueError("Output path must end with '.csv'")

    # 创建输出目录
    output_dir = Path(args.output_path).parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # 读取原始 CSV 数据
    df = pd.read_csv(args.input_path)

    # 标准化路径：将反斜杠替换为正斜杠
    df['image_path'] = df['image_path'].str.replace('\\', '/', regex=False)

    # 使用正则表达式提取关键字段
    pattern = r"^.*?/(\d+)x_(\d+\.\d+)_\w+/sample(\d+)/field(\d+)/roi\d+/(\d+)\.jpg$"
    extracted = df['image_path'].str.extract(pattern, expand=True)

    # 检查提取结果（可选调试）
    if extracted.isnull().any().any():
        print("⚠️ 警告：部分路径匹配失败，以下为失败示例：")
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
    grouped_df.to_csv(args.output_path, index=False)
    print(f"✅ Results saved to: {args.output_path}")

if __name__ == "__main__":
    main()