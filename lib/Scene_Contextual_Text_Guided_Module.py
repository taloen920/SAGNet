import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierProcessor(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv_before = nn.Conv2d(dim, dim, kernel_size=1)
        self.freq_sim = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.freq_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_orig = x
        x = self.conv_before(x)
        freq_feats = self.freq_sim(x)
        attention = self.freq_attention(freq_feats)
        freq_feats = freq_feats * attention
        enhanced = self.fusion(torch.cat([x_orig, freq_feats], dim=1))

        return enhanced


class SceneClassifier(nn.Module):

    def __init__(self, dim, num_contexts=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_contexts, kernel_size=1)
        )
    def forward(self, x):
        context_logits = self.classifier(x)
        context_map = F.softmax(context_logits, dim=1)
        return context_map


class TextGuidedEnhancer(nn.Module):

    def __init__(self, visual_dim, text_dim, num_contexts):
        super().__init__()
        self.num_contexts = num_contexts
        self.text_projections = nn.ModuleList([
            nn.Linear(text_dim, visual_dim)
            for _ in range(num_contexts)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, kernel_size=1),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, visual_feat, context_map, text_feat, text_mask):
        batch, c, h, w = visual_feat.shape
        if text_feat.shape[1] == 768 and text_feat.shape[2] != 768:
            text_feat = text_feat.transpose(1, 2)   #[B, D, L]-->[B, L, D]
        if text_mask.shape[2] != 1 and text_mask.shape[1] == 1:
            text_mask = text_mask.transpose(1, 2)  #[B, 1, L]-->[B, L, 1]
        text_attention_mask = text_mask.squeeze(-1).float()  # [B, L]
        mask_sum = text_attention_mask.sum(dim=1, keepdim=True)
        text_attention_mask = text_attention_mask / (mask_sum + 1e-6)
        global_text = torch.bmm(text_attention_mask.unsqueeze(1), text_feat).squeeze(1)  # [B, D]
        context_guided_features = []
        for i in range(self.num_contexts):
            text_proj = self.text_projections[i](global_text)  # [B, C]
            text_spatial = text_proj.view(batch, c, 1, 1).expand_as(visual_feat)
            context_weight = context_map[:, i:i + 1]  # [B, 1, H, W]
            context_guided_features.append(text_spatial * context_weight)
        combined_guidance = sum(context_guided_features)
        enhanced = self.fusion(torch.cat([visual_feat, combined_guidance], dim=1))

        return enhanced


class FourierContextEnhancerModule(nn.Module):

    def __init__(self, visual_dim, text_dim=768, num_contexts=3):
        super().__init__()
        self.num_contexts = num_contexts
        self.scene_classifier = SceneClassifier(visual_dim, num_contexts)
        self.text_enhancer = TextGuidedEnhancer(visual_dim, text_dim, num_contexts)
        self.fourier_processor = FourierProcessor(visual_dim)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, kernel_size=1),
            nn.BatchNorm2d(visual_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, visual_feat, text_feat, text_mask):
        context_map = self.scene_classifier(visual_feat)
        text_enhanced = self.text_enhancer(visual_feat, context_map, text_feat, text_mask)
        fourier_enhanced = self.fourier_processor(visual_feat)
        final_output = self.final_fusion(torch.cat([text_enhanced, fourier_enhanced], dim=1))

        return final_output


class CrossScaleAdapter(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size=None):
        x = self.adapt(x)
        if target_size is not None and x.size()[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class SCTG(nn.Module):

    def __init__(self, visual_dims, text_dim=768, num_contexts=3):
        super().__init__()
        self.visual_dims = visual_dims
        self.context_enhancers = nn.ModuleList([
            FourierContextEnhancerModule(dim, text_dim, num_contexts)
            for dim in visual_dims
        ])
        self.down_adapters = nn.ModuleList([
            CrossScaleAdapter(visual_dims[i + 1], visual_dims[i])
            for i in range(len(visual_dims) - 1)
        ])
        self.up_adapters = nn.ModuleList([
            CrossScaleAdapter(visual_dims[i], visual_dims[i + 1])
            for i in range(len(visual_dims) - 1)
        ])
        self.fusion_gates = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                nn.Sigmoid()
            ) for dim in visual_dims
        ])
        self.final_fusions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            ) for dim in visual_dims
        ])

    def forward(self, features, text_feat, text_mask):
        original_features = [feat.clone() for feat in features]
        context_enhanced = [
            self.context_enhancers[i](feat, text_feat, text_mask)
            for i, feat in enumerate(features)
        ]
        feature_sizes = [feat.size()[2:] for feat in context_enhanced]
        top_down_features = context_enhanced.copy()
        for i in range(len(self.visual_dims) - 2, -1, -1):
            higher_feat = self.down_adapters[i](top_down_features[i + 1], feature_sizes[i])
            gate = self.fusion_gates[i](torch.cat([top_down_features[i], higher_feat], dim=1))
            top_down_features[i] = top_down_features[i] + higher_feat * gate
        bottom_up_features = top_down_features.copy()
        for i in range(len(self.visual_dims) - 1):
            lower_feat = self.up_adapters[i](bottom_up_features[i], feature_sizes[i + 1])
            gate = self.fusion_gates[i + 1](torch.cat([bottom_up_features[i + 1], lower_feat], dim=1))
            bottom_up_features[i + 1] = bottom_up_features[i + 1] + lower_feat * gate
        final_features = [
            self.final_fusions[i](torch.cat([original, enhanced], dim=1))
            for i, (original, enhanced) in enumerate(zip(original_features, bottom_up_features))
        ]

        return final_features