import torch
import timm
from torch import nn
from torchinfo import summary

class SOAFModel(nn.Module):
    def __init__(self, 
                 model_name: str = "mobilenetv4_conv_small", 
                 pretrained: bool = True):
        super().__init__()
        
        # 创建特征提取器（使用-1获取最后一个特征层）
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),  # 修改点1：使用-1获取最后一个特征层
        )

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 获取特征维度（先池化再展平）
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            # 修改点2：先池化再展平计算特征维度
            features = self.feature_extractor(dummy_input)[0]
            pooled_features = self.global_pool(features)
        feature_dim = pooled_features.view(pooled_features.size(0), -1).shape[-1]

        # 构建全连接网络
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)[0]
        
        # 池化 + 展平
        pooled = self.global_pool(features)
        flattened = torch.flatten(pooled, 1)
        
        return self.fc(flattened)


def export_onnx(model, device, save_path="soaf_model.onnx"):
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=13  # 添加明确的操作集版本
    )
    print(f"ONNX模型已导出至: {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SOAFModel().to(device)

    # 验证修改后的特征维度计算
    test_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        features = model.feature_extractor(test_input)[0]
        pooled = model.global_pool(features)
        print(f"池化后特征形状: {pooled.shape}")  # 应显示如 torch.Size([2, 512, 1, 1])
        print(f"全连接输入维度: {model.fc[0].in_features}")  # 显示实际特征维度

    # 测试推理
    output = model(test_input)
    print(f"\n批量测试输出形状: {output.shape}")  # 应显示 torch.Size([2, 1])

    # 打印模型结构
    summary(model, input_size=(1, 3, 224, 224))

    # 导出ONNX
    model.eval()
    export_onnx(model, device)

