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
    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim

    def forward(self, img_feat, aux_feat):
        gamma = aux_feat[:, :self.in_dim]
        beta = aux_feat[:, self.in_dim:]
        return gamma * img_feat + beta


class AdditiveFusion(nn.Module):
    def forward(self, img_feat, aux_feat):
        return img_feat + aux_feat


class ConcatFusion(nn.Module):
    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.downsample = nn.Linear(in_dim * 2, in_dim)

    def forward(self, img_feat, aux_feat):
        combined = torch.cat([img_feat, aux_feat], dim=1)
        return self.downsample(combined)


class GatingFusion(nn.Module):
    def __init__(self, in_dim: int = 64):
        super().__init__()
        self.gate = nn.Linear(in_dim * 2, in_dim)

    def forward(self, img_feat, aux_feat):
        combined = torch.cat([img_feat, aux_feat], dim=1)
        gate_weights = torch.sigmoid(self.gate(combined))
        return gate_weights * img_feat + (1 - gate_weights) * aux_feat


class NoFusion(nn.Module):
    def forward(self, img_feat, aux_feat):
        return img_feat  # 直接返回图像特征，忽略辅助输入


class MOAFModel(nn.Module):
    def __init__(self,
                 model_name: str = "mobilenetv4_conv_small",
                 fusion_mode: str = "film",
                 pretrained: bool = True):
        super().__init__()
        self.fusion_mode = fusion_mode

        # 图像特征提取
        self.feature_extractor = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            features_only=True, 
            out_indices=(-1,)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 动态获取图像特征维度
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, 224, 224)
            features = self.feature_extractor(dummy_img)[0]
            pooled = self.global_pool(features)
            img_feat_dim = pooled.view(1, -1).shape[-1]

        # 图像降维网络（统一结构）
        self.img_fc = nn.Sequential(
            nn.Linear(img_feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)
        )

        # 辅助输入处理网络（根据融合模式调整）
        if fusion_mode in ["add", "concat", "gating"]:
            self.aux_fc = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 64),
                nn.ReLU(inplace=True)
            )
        elif fusion_mode == "film":
            self.aux_fc = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 128),
                nn.ReLU(inplace=True)
            )
        elif fusion_mode == "no_fusion":
            # 无融合模式下，辅助输入无需处理（可选：保留空结构或移除）
            self.aux_fc = nn.Identity()  # 保留空结构，避免 forward 中报错
        else:
            raise ValueError(f"不支持的融合模式: {fusion_mode}")

        # 特征融合模块（显式传入 in_dim）
        if fusion_mode == "film":
            self.fusion_layer = FiLMLayer(in_dim=64)
        elif fusion_mode == "add":
            self.fusion_layer = AdditiveFusion()
        elif fusion_mode == "concat":
            self.fusion_layer = ConcatFusion(in_dim=64)
        elif fusion_mode == "gating":
            self.fusion_layer = GatingFusion(in_dim=64)
        elif fusion_mode == "no_fusion":
            self.fusion_layer = NoFusion()  # 添加 no_fusion 融合层
        else:
            raise ValueError(f"不支持的融合模式: {fusion_mode}")

        # 最终回归网络
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, img, aux_input):
        img_feat = self.feature_extractor(img)[0]
        img_feat = self.global_pool(img_feat).flatten(1)
        img_feat = self.img_fc(img_feat)  # [B, 64]

        aux_feat = self.aux_fc(aux_input)  # [B, 64] 或 [B, 128]

        fused_feat = self.fusion_layer(img_feat, aux_feat)
        return self.regressor(fused_feat)


def MOAF_info_with_onnx():
    fusion_mode = "no_fusion"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MOAFModel(fusion_mode=fusion_mode).to(device)
    
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
        f"moaf_model_{fusion_mode}.onnx",
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

