import torch
import torch.nn as nn
import torchvision.models as models
from networks.ViT import ViT
from networks.KAN import KAN
from networks.layers import CrossAttention2d, LocalAttention, AdP, Decoder


class ACIR(nn.Module):
    """

    This model uses CNN backbone and Pixel Transformer to extract global and local features, which are then fused
    for classification/hashing. Optionally, a reconstruction decoder can be adopted for OOD detection.

    Parameters:
        input_size (int): Input image size (assumes square input, e.g., 224).
        bit (int): Output dimension for the final classification or other task.
        use_recon_decoder (bool): If True, the reconstruction decoder branch is added. Defaults to False.
            For OOD detection, the network should be trained in two stages:
                1. Train the model without the reconstruction decoder.
                2. Enable the reconstruction decoder and freeze the weights of the remaining parts of ACIR.
    """

    def __init__(self, input_size, bit, num_cls, use_recon_decoder=False):
        super(ACIR, self).__init__()
        self.input_size = input_size
        self.use_recon_decoder = use_recon_decoder

        # CNN stem using ResNet50 (with fully connected and pooling removed)
        self.cnn_stem = models.resnet50(pretrained=True)
        self.cnn_stem.fc = nn.Identity()
        self.cnn_stem.avgpool = nn.Identity()

        # CNN-based Local attention modules for shallow (layer2) and deep (layer4) features
        self.attn_bottel1 = LocalAttention(512, dim_out=None, window_size=7, k=1, heads=8, dim_head=64)
        self.attn_bottel2 = LocalAttention(2048, dim_out=None, window_size=7, k=1, heads=8, dim_head=64)

        # Mapping layers to adjust the feature maps to a fixed spatial size (input_size // 32)
        self.mapping_layers = nn.ModuleList([
            AdP(in_channel=256 * 2, target_size=input_size // 32),  # For layer2 (512 channels)
            AdP(in_channel=256 * 8, target_size=input_size // 32)  # For layer4 (2048 channels)
        ])

        # CNN-based Cross-attention fusion of CNN features
        self.cnn_fusion = CrossAttention2d(in_dim1=512, in_dim2=2048, k_dim=128, v_dim=128, num_heads=12)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        # Pixel-wise Transformer to refine the features from CNN stem
        self.pit = ViT(
            image_size=input_size // 32,
            patch_size=1,
            channels=2048,
            num_classes=bit,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
            emb_dropout=0.1
        )

        # Fully connected layers for global and local branches
        self.fc_global = nn.Linear(768 * 6, 1536)
        self.conv_local = nn.Conv2d(768 * 6, 1536, kernel_size=1)

        # Final classification layer using KAN with fused features
        self.KANlayers = KAN([3072 + 512, 512, bit])
        self.cls_head = nn.Linear(512, num_cls)

        # Reconstruction decoder is defined only if used
        if self.use_recon_decoder:
            self.recon_decoder = nn.Sequential(
                Decoder(2048, 512, 512),
                Decoder(512, 256, 256, scale=4),
                Decoder(256, 128, 64, scale=4),
                nn.Conv2d(64, 3, kernel_size=3, padding='same')
            )

    def forward(self, x, return_featureEmd=False, ood_test=False):
        # ---------------------- CNN Stem & CNN-based attention ----------------------
        # --------- Add with torch.no_grad(): when training the OOD decoder ----------
        x = self.cnn_stem.conv1(x)
        x = self.cnn_stem.bn1(x)
        x = self.cnn_stem.relu(x)
        x = self.cnn_stem.maxpool(x)
        x = self.cnn_stem.layer1(x)
        x = self.cnn_stem.layer2(x)  # Output:  /8, 512 channels
        shallow, _ = self.attn_bottel1(x)
        shallow = x + shallow
        x = self.cnn_stem.layer3(x)  # Output: /16, 1024 channels
        x = self.cnn_stem.layer4(x)  # Output: /32, 2048 channels
        deep, _ = self.attn_bottel2(x)
        deep = x + deep

        # ---------------------- Reconstruction Decoder ----------------------
        if self.use_recon_decoder:
            # Clone the feature map for reconstruction
            recon = x.clone()
            for dec in self.recon_decoder:
                recon = dec(recon)
        else:
            recon = None

        # ---------------------- Pixel Transformer and Feature Fusion ----------------------
        # ------------- Add with torch.no_grad(): when training the OOD decoder ------------
        shallow = self.mapping_layers[0](shallow)
        deep = self.mapping_layers[1](deep)
        fused_cnn = self.avg(self.cnn_fusion(shallow, deep)).view(x.shape[0], -1)

        pit_logits, pit_features = self.pit(x)
        global_features = []
        local_features = []
        # Extract features from the last 6 layers of the PiT
        for feature in pit_features[-6:]:
            global_features.append(feature[:, 0, :])
            local_token = feature[:, 1:, :].permute(0, 2, 1).view(
                -1, 768, self.input_size // 32, self.input_size // 32)
            local_features.append(local_token)

        global_input = torch.cat(global_features, dim=1)
        local_input = torch.cat(local_features, dim=1)
        global_out = self.fc_global(global_input)
        local_out = self.avg(self.conv_local(local_input)).view(x.shape[0], -1)
        fused_features = torch.cat([global_out, local_out, fused_cnn], dim=1)
        output, KANmedium = self.KANlayers(fused_features, return_mid=True)
        pd_cls_logits = self.cls_head(KANmedium)

        if return_featureEmd:
            return output, pd_cls_logits, fused_features
        elif ood_test:
            return output, pd_cls_logits, recon, fused_features
        else:
            return output, pd_cls_logits