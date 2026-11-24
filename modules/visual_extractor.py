import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    """Single-scale visual extractor (ResNet-101) for the single-scale ablation.

    The full DGMK-Net model uses the multi-scale extractor
    (modules/multi_scale_extractor.py); this class is only used when
    multi-scale feature aggregation is disabled.
    """

    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.args = args
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        print(f"=> creating model '{args.visual_extractor}'")

        # Use the official torchvision pretrained model
        model = getattr(models, self.visual_extractor)(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        self.classifier = nn.Linear(2048, args.num_labels)

        # load pretrained visual extractor
        if args.pretrain_cnn_file and args.pretrain_cnn_file != "":
            print(f'Load pretrained CNN model from: {args.pretrain_cnn_file}')
            checkpoint = torch.load(args.pretrain_cnn_file, map_location='cuda:{}'.format(args.gpu))
            # Compatible with different checkpoint formats: 'state_dict', 'model', or a raw state_dict
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model' in checkpoint:
                    state_dict = checkpoint['model']
                else:
                    # Assume the checkpoint itself is the state_dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict)
        else:
            print(f'Load pretrained CNN model from: official pretrained in ImageNet')

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        labels = self.classifier(avg_feats)

        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats, labels
