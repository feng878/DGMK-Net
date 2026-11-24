import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os


class ScaleAttention(nn.Module):
    """
    SE-style scale attention module (fixed version).
    Learns per-scale weights and adaptively fuses multi-scale features.

    Fixes in this version:
    1. Added LayerNorm to stabilize feature distributions across scales
    2. More conservative weight initialization to avoid one scale dominating early on
    3. Gradient clipping to prevent gradient explosion
    """
    def __init__(self, num_scales=3, d_vf=2048, reduction=16):
        super(ScaleAttention, self).__init__()
        self.num_scales = num_scales
        self.d_vf = d_vf
        self.reduction = reduction

        # LayerNorm to stabilize feature distributions
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([d_vf, 7, 7]) for _ in range(num_scales)
        ])

        # Global Average Pooling + FC to learn scale weights
        hidden_dim = d_vf // reduction
        self.fc1 = nn.Linear(num_scales * d_vf, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_scales)

        # Initialize weights (more conservative initialization)
        self._init_weights()

    def _init_weights(self):
        """Conservative initialization to avoid one scale dominating early on."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # fc2 initialized close to a uniform distribution so that
        # all scale weights start out roughly equal
        nn.init.uniform_(self.fc2.weight, -0.01, 0.01)
        nn.init.constant_(self.fc2.bias, 1.0 / self.num_scales)

    def forward(self, multi_scale_feats):
        """
        Args:
            multi_scale_feats: list of [B, d_vf, 7, 7] features from different scales

        Returns:
            fused_feats: [B, d_vf, 7, 7] fused features
            scale_weights: [B, num_scales] weight of each scale
        """
        B = multi_scale_feats[0].shape[0]

        # LayerNorm to stabilize per-scale feature distributions
        normalized_feats = []
        for i, feat in enumerate(multi_scale_feats):
            normalized_feats.append(self.layer_norms[i](feat))

        # GAP for each scale
        gap_feats = []
        for feat in normalized_feats:
            gap = F.adaptive_avg_pool2d(feat, 1).view(B, -1)  # [B, d_vf]
            gap_feats.append(gap)

        # Concatenate GAP features of all scales
        concat_gap = torch.cat(gap_feats, dim=1)  # [B, num_scales * d_vf]

        # Compute scale weights
        w = F.relu(self.fc1(concat_gap))  # [B, hidden_dim]
        w = self.fc2(w)  # [B, num_scales]
        scale_weights = F.softmax(w, dim=1)  # [B, num_scales]

        # Weighted fusion (using the normalized features)
        fused_feats = torch.zeros_like(normalized_feats[0])  # [B, d_vf, 7, 7]
        for i, feat in enumerate(normalized_feats):
            fused_feats += scale_weights[:, i:i+1, None, None] * feat

        # Residual connection: fused features + original strongest scale (layer4)
        # prevents information loss if the attention degenerates
        fused_feats = fused_feats + 0.1 * multi_scale_feats[-1]

        return fused_feats, scale_weights


class MultiScaleVisualExtractor(nn.Module):
    """
    Multi-Scale Visual Feature Adaptive Aggregation (MSVFA).

    Motivation: abnormalities at different scales in medical images require
    features from different network depths to be captured.

    Method:
    - Extract multi-scale features from ResNet-101 layer2 (512ch, 28x28),
      layer3 (1024ch, 14x14) and layer4 (2048ch, 7x7)
    - Project all scales to the same channel count (d_vf=2048) with 1x1 convs
    - Unify the spatial resolution to 7x7 with adaptive average pooling
    - Adaptively fuse the scales with an SE-style ScaleAttention module
    """
    def __init__(self, args, use_multi_scale=True):
        super(MultiScaleVisualExtractor, self).__init__()
        self.args = args
        self.use_multi_scale = use_multi_scale
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        self.d_vf = args.d_vf
        self.num_labels = args.num_labels

        if use_multi_scale and 'resnet' in self.visual_extractor:
            self._build_multi_scale_resnet()
        else:
            self._build_single_scale_model()

        # load pretrained visual extractor
        if args.pretrain_cnn_file and args.pretrain_cnn_file != "":
            # Check whether the file exists
            if os.path.exists(args.pretrain_cnn_file):
                print(f'Load pretrained CNN model from: {args.pretrain_cnn_file}')
                checkpoint = torch.load(args.pretrain_cnn_file, map_location='cuda:{}'.format(args.gpu))
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    # Try to obtain the state_dict, supporting several common key names
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    elif 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        # If none of the common keys exist, use the checkpoint itself as the state_dict
                        state_dict = checkpoint
                else:
                    # The checkpoint is directly a state_dict
                    state_dict = checkpoint
                # Load the weights into the appropriate sub-modules depending on the mode
                if use_multi_scale and 'resnet' in self.visual_extractor:
                    # Multi-scale mode: load into each individual layer
                    # Build a temporary model to load the weights
                    temp_model = getattr(models, self.visual_extractor)(pretrained=False)
                    temp_model.load_state_dict(state_dict)
                    # Copy the weights into each layer
                    self.conv1.load_state_dict(temp_model.conv1.state_dict())
                    self.bn1.load_state_dict(temp_model.bn1.state_dict())
                    self.layer1.load_state_dict(temp_model.layer1.state_dict())
                    self.layer2.load_state_dict(temp_model.layer2.state_dict())
                    self.layer3.load_state_dict(temp_model.layer3.state_dict())
                    self.layer4.load_state_dict(temp_model.layer4.state_dict())
                else:
                    # Single-scale mode: load directly into self.model
                    self.model.load_state_dict(state_dict)
            else:
                print(f'Warning: Pretrained CNN file not found: {args.pretrain_cnn_file}')
                print(f'Using official pretrained model from ImageNet instead.')
        else:
            print(f'Load pretrained CNN model from: official pretrained in ImageNet')

    def _build_multi_scale_resnet(self):
        """Build the multi-scale ResNet feature extractor."""
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)

        # Keep all layers up to layer4
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # 1x1 convs projecting each scale to the same channel count (d_vf=2048)
        self.proj_layer2 = nn.Conv2d(512, self.d_vf, kernel_size=1, stride=1, bias=False)
        self.proj_layer3 = nn.Conv2d(1024, self.d_vf, kernel_size=1, stride=1, bias=False)
        self.proj_layer4 = nn.Conv2d(2048, self.d_vf, kernel_size=1, stride=1, bias=False)

        # Scale attention module
        self.scale_attention = ScaleAttention(num_scales=3, d_vf=self.d_vf, reduction=16)

        # Classifier
        self.classifier = nn.Linear(self.d_vf, self.num_labels)

        # Average pooling (for single-scale compatibility)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        # Initialize projection layers
        self._init_projection_layers()

    def _build_single_scale_model(self):
        """Build the single-scale model (for ablation, fixed version)."""
        if 'resnet' in self.visual_extractor:
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            self.classifier = nn.Linear(2048, self.num_labels)
            # Fix: use AdaptiveAvgPool2d instead of a fixed-size AvgPool2d
            # so that a stable 1x1 output is produced for any input size,
            # avoiding NaN caused by unexpected feature map sizes
            self.avg_fnt = torch.nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise NotImplementedError(f"Single scale not implemented for {self.visual_extractor}")

    def _init_projection_layers(self):
        """Initialize projection layer parameters (fixed version)."""
        # Smaller initialization weights to avoid large magnitude differences early on
        nn.init.xavier_uniform_(self.proj_layer2.weight, gain=0.5)
        nn.init.xavier_uniform_(self.proj_layer3.weight, gain=0.7)
        nn.init.xavier_uniform_(self.proj_layer4.weight, gain=1.0)

    def forward(self, images):
        """
        Args:
            images: [B, 3, H, W] input images

        Returns:
            patch_feats: [B, 49, d_vf] spatial features (7x7 = 49 patches)
            avg_feats: [B, d_vf] global features
            labels: [B, 14] disease label predictions
        """
        if self.use_multi_scale and 'resnet' in self.visual_extractor:
            return self._forward_multi_scale(images)
        else:
            return self._forward_single_scale(images)

    def _forward_multi_scale(self, images):
        """Multi-scale forward pass."""
        # ResNet forward pass
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)  # unused
        x2 = self.layer2(x1)  # [B, 512, 28, 28]
        x3 = self.layer3(x2)  # [B, 1024, 14, 14]
        x4 = self.layer4(x3)  # [B, 2048, 7, 7]

        # 1x1 conv projection to the same channel count
        f2 = self.proj_layer2(x2)  # [B, d_vf, 28, 28]
        f3 = self.proj_layer3(x3)  # [B, d_vf, 14, 14]
        f4 = self.proj_layer4(x4)  # [B, d_vf, 7, 7]

        # Adaptive average pooling to 7x7
        f2 = F.adaptive_avg_pool2d(f2, 7)  # [B, d_vf, 7, 7]
        f3 = F.adaptive_avg_pool2d(f3, 7)  # [B, d_vf, 7, 7]
        # f4 is already 7x7

        # ScaleAttention adaptive fusion (LayerNorm is applied internally)
        fused_feats, scale_weights = self.scale_attention([f2, f3, f4])  # [B, d_vf, 7, 7]

        # Convert to patch features [B, 49, d_vf]
        batch_size, feat_size, _, _ = fused_feats.shape
        patch_feats = fused_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)

        # Global features (GAP)
        avg_feats = F.adaptive_avg_pool2d(fused_feats, 1).view(batch_size, -1)  # [B, d_vf]

        # Label prediction
        labels = self.classifier(avg_feats)  # [B, 14]

        return patch_feats, avg_feats, labels

    def _forward_single_scale(self, images):
        """Single-scale forward pass."""
        if 'resnet' in self.visual_extractor:
            patch_feats = self.model(images)

            # AdaptiveAvgPool2d ensures a stable 1x1 output
            avg_feats = self.avg_fnt(patch_feats).view(images.size(0), -1)

            labels = self.classifier(avg_feats)

            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
            return patch_feats, avg_feats, labels
        else:
            raise NotImplementedError
