import torch
import torch.nn as nn
import numpy as np

from modules.multi_scale_extractor import MultiScaleVisualExtractor
from modules.Transformer import TransformerModel
from modules.text_encoder import TextEncoder
from modules.graph_attention import DiseaseDependencyGAT


class DGMKNet(nn.Module):
    """
    DGMK-Net: Disease-Graph and Multi-Granularity Knowledge Enhanced Radiology Report Generation

    Three main contributions:
    1. MSVFA: Multi-Scale Visual Feature Adaptive Aggregation
    2. DD-GAT: Disease Dependency Graph Attention Network
    3. SACL: Symptom-Aware Contrastive Learning

    Architecture:
    Image → ResNet(layer2+3+4) → MSVFA → patch_feats[B,49,d_vf] + avg_feats[B,d_vf]
                                              ↓
                                   Linear → label_logits
                                              ↓
                                   DD-GAT (graph attention enhancement)
                                              ↓
                                   graph_query[B,14,d_model]
                                              ↓
                                   MHA queries the Knowledge Base
                                              ↓
              [patch_feats ⊕ prior] → Transformer → Report

    Loss = L_lm + α_label·L_bce + α_contrastive·L_SACL
    """
    def __init__(self, args, tokenizer):
        super(DGMKNet, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        # Model hyperparameters
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_layers = args.num_layers
        self.num_labels = args.num_labels
        self.tgt_vocab = len(tokenizer.idx2token) + 1
        self.h = args.num_heads
        self.num_slots = args.num_slots
        self.d_vf = args.d_vf
        self.dropout = args.dropout

        # 1. Multi-scale visual feature extractor (MSVFA)
        self.visual_extractor = MultiScaleVisualExtractor(
            args,
            use_multi_scale=not getattr(args, 'no_multi_scale', False)
        )

        # 2. Text encoder (used to update the knowledge base during training)
        self.txt_encoder = TextEncoder(
            self.d_model, self.d_ff, self.num_layers,
            self.tgt_vocab, self.num_labels,
            self.h, self.dropout
        )

        # 3. Knowledge base modules
        self.update_memory = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=args.num_memory_heads,
            dropout=self.dropout
        )
        self.select_prior = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=args.num_memory_heads,
            dropout=self.dropout
        )

        # 4. Disease dependency graph attention network (DD-GAT)
        self.dd_gat = DiseaseDependencyGAT(
            num_labels=self.num_labels,
            d_model=self.d_model,
            num_gat_layers=getattr(args, 'num_gat_layers', 2),
            num_gat_heads=getattr(args, 'num_gat_heads', 4),
            dropout=self.dropout
        )

        # 5. Projection layers
        self.get_mem = nn.Linear(self.d_model, self.d_model)
        self.linear_z = nn.Linear(self.d_vf, self.d_model)
        self.linear_feat = nn.Linear(self.d_model, self.d_vf)

        # 6. Transformer decoder
        self.encoder_decoder = TransformerModel(args, tokenizer)

        # 7. Knowledge base (registered as buffers to exclude from optimizer updates)
        self._init_memory_buffers()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights."""
        nn.init.kaiming_normal_(self.linear_z.weight)
        self.linear_z.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.get_mem.weight)
        self.get_mem.bias.data.fill_(0)
        nn.init.kaiming_normal_(self.linear_feat.weight)
        self.linear_feat.bias.data.fill_(0)

    def _init_memory_buffers(self):
        """Initialize the knowledge base matrix as buffers (excluded from gradient updates)."""
        memory = torch.eye(self.num_slots).unsqueeze(0)
        if self.d_model > self.num_slots:
            diff = self.d_model - self.num_slots
            pad = torch.zeros((1, self.num_slots, diff))
            memory = torch.cat([memory, pad], -1)
        elif self.d_model < self.num_slots:
            memory = memory[:, :, :self.d_model]

        mask = torch.ones((self.num_slots, self.d_model))
        mask[:, self.num_slots:] = 0

        self.register_buffer('memory', memory)
        self.register_buffer('mask', mask)

    def forward_iu_xray(self, images, targets=None, labels=None, mode='train'):
        """Forward pass for the IU-Xray dataset (two-image input)."""
        # Extract multi-scale features (two images)
        att_feats_0, fc_feats_0, labels_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1, labels_1 = self.visual_extractor(images[:, 1])

        # Fuse features of the two images - average instead of concatenation
        fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1]), dim=0)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        # Use averaged labels as the visual labels
        vis_labels = (labels_0 + labels_1) / 2.0

        return att_feats, fc_feats, vis_labels

    def forward_mimic_cxr(self, images, targets=None, labels=None, mode='train'):
        """Forward pass for the MIMIC-CXR dataset (single-image input)."""
        # Extract multi-scale features
        att_feats, fc_feats, vis_labels = self.visual_extractor(images)

        return att_feats, fc_feats, vis_labels

    def forward(self, images, targets=None, labels=None, mode='train'):
        """
        Full forward pass.

        Args:
            images: input images [B, 3, H, W] or [B, 2, 3, H, W] (IU-Xray)
            targets: target report token IDs (used in training)
            labels: ground-truth labels [B, 14]
            mode: 'train' or 'sample'

        Returns:
            train: (output, vis_labels, txt_labels, z_img, z_txt)
            sample: (output, vis_labels)
        """
        bsz = images.shape[0]

        # Select the dataset-specific forward function
        ve = self.forward_iu_xray if self.args.dataset_name == 'iu_xray' else self.forward_mimic_cxr
        att_feats, avg_feats, vis_labels = ve(images, targets, labels, mode)

        # Compute the image embedding z_img
        z_img = self.linear_z(avg_feats)  # [B, d_model]

        # Initialize the knowledge base (buffers follow the device automatically)
        memory = self.get_mem(self.memory).expand(bsz, -1, -1)  # [B, num_slots, d_model]
        mask = self.mask.expand(bsz, -1, -1)

        # During training: update the knowledge base with the text encoder
        txt_labels, z_txt = None, None
        if mode == 'train':
            txt_feats, z_txt, txt_labels = self.txt_encoder(targets)
            memory = self.update_memory(
                memory.transpose(0, 1),  # [num_slots, B, d_model]
                txt_feats.transpose(0, 1),  # [seq_len, B, d_model]
                txt_feats.transpose(0, 1)
            )[0].transpose(0, 1)  # [B, num_slots, d_model]

        # DD-GAT: graph attention enhanced label representation
        graph_query = self.dd_gat(vis_labels)  # [B, 14, d_model]

        # Retrieve prior knowledge from the knowledge base
        prior = self.select_prior(
            graph_query.transpose(0, 1),  # [14, B, d_model]
            memory.transpose(0, 1),  # [num_slots, B, d_model]
            memory.transpose(0, 1)
        )[0].transpose(0, 1)  # [B, 14, d_model]

        # Project prior knowledge into the visual feature space and append to att_feats
        att_feats = torch.cat((att_feats, self.linear_feat(prior)), dim=1)

        # Transformer decoder generates the report
        if mode == 'train':
            output = self.encoder_decoder(avg_feats, att_feats, targets, mode='forward')
            return (output, vis_labels, txt_labels, z_img, z_txt)
        elif mode == 'sample':
            output, _ = self.encoder_decoder(avg_feats, att_feats, opt=self.args, mode='sample')
            return output, vis_labels
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __str__(self):
        """Print the number of model parameters."""
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
