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


class FiLMLayer(nn.Module):
    def forward(self, x, gamma, beta):
        return gamma * x + beta  # 直接使用外部参数


class MOAFModel(nn.Module):
    def __init__(self, 
                 model_name: str = "mobilenetv4_conv_small", 
                 pretrained: bool = True):
        super().__init__()
        
        # 图像特征提取器
        self.feature_extractor = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 动态获取特征维度
        dummy_img = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            features = self.feature_extractor(dummy_img)[0]
            pooled = self.global_pool(features)
        img_feat_dim = pooled.view(pooled.size(0), -1).shape[-1]
        
        # 图像特征降维网络
        self.img_fc = nn.Sequential(
            nn.Linear(img_feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )
        
        # 辅助参数处理网络
        self.aux_fc = nn.Sequential(
            nn.Linear(2, 64),  # 输入为(magnification, NA)
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True)
        )
        
        # FiLM调制层
        self.film = FiLMLayer()
        
        # 最终回归网络
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, img, aux_input):
        # 图像特征提取
        img_feat = self.feature_extractor(img)[0]
        img_feat = self.global_pool(img_feat)
        img_feat = torch.flatten(img_feat, 1)
        img_feat = self.img_fc(img_feat)  # [B, 64]
        
        # 辅助参数处理
        aux_feat = self.aux_fc(aux_input)  # [B, 128]
        gamma_feat = aux_feat[:, :64]      # 分割前半部分作为gamma参数
        beta_feat = aux_feat[:, 64:]      # 分割后半部分作为beta参数
        
        # FiLM特征融合
        fused_feat = self.film(img_feat, gamma_feat, beta_feat)
        
        # 最终回归
        return self.regressor(fused_feat)


def MOAF_info_with_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MOAFModel().to(device)
    
    # 验证输入输出维度
    test_img = torch.randn(2, 3, 224, 224).to(device)
    test_aux = torch.randn(2, 2).to(device)  # 模拟(magnification, NA)
    
    with torch.no_grad():
        output = model(test_img, test_aux)
        print(f"输出形状: {output.shape}")  # 预期输出: torch.Size([2, 1])
    
    # 打印模型结构
    summary(model, input_size=[(1, 3, 224, 224), (1, 2)])
    
    # 导出ONNX模型
    model.eval()
    dummy_img = torch.randn(1, 3, 224, 224).to(device)
    dummy_aux = torch.randn(1, 2).to(device)
    
    torch.onnx.export(
        model,
        (dummy_img, dummy_aux),
        "moaf_model.onnx",
        input_names=["image_input", "aux_input"],
        output_names=["output"],
        dynamic_axes={
            "image_input": {0: "batch_size"},
            "aux_input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=13
    )
    print("ONNX模型导出完成")


def SOAF_info_with_onnx():
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
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        "test_soaf_model.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=13  # 添加明确的操作集版本
    )
    print(f"ONNX模型导出完成")


if __name__ == "__main__":
    MOAF_info_with_onnx()

