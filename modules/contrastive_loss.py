import torch
import torch.nn as nn
import torch.nn.functional as F

# Module-level singletons: avoid re-creating criteria on every compute_loss_dgmk call
_criterion_lm = None
_criterion_bce = None
_sacl_loss = None


class SymptomAwareContrastiveLoss(nn.Module):
    """
    Symptom-Aware Contrastive Learning (SACL).

    Motivation: the ranking loss of M2KT only compares adjacent samples and does
    not fully exploit all in-batch negatives. Samples with larger label
    differences receive a larger contrastive margin.

    Method:
    - Replaces the simple ranking loss with an InfoNCE-style contrastive loss
    - Computes a cosine similarity matrix over all image-text pairs in the batch
    - Adds an adaptive margin for hard negatives based on the Jaccard
      dissimilarity of the labels
    - Negative pairs with larger label differences get larger margins, forcing
      the model to learn more discriminative representations
    """
    def __init__(self, temperature=0.07, margin_weight=0.5):
        super(SymptomAwareContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin_weight = margin_weight

    def compute_jaccard_distance(self, labels1, labels2):
        """
        Compute the Jaccard distance (dissimilarity) between two sets of label vectors.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|
        Jaccard_Distance = 1 - Jaccard(A, B)

        Args:
            labels1: [B, 14] binary labels (may be logits; thresholded first)
            labels2: [B, 14]

        Returns:
            distance: [B, B] Jaccard distance matrix
        """
        # Convert logits to binary labels (threshold=0)
        binary1 = (labels1 > 0).float()  # [B, 14]
        binary2 = (labels2 > 0).float()  # [B, 14]

        # Compute intersection and union
        intersection = torch.matmul(binary1, binary2.t())  # [B, B]
        sum1 = binary1.sum(dim=1, keepdim=True)  # [B, 1]
        sum2 = binary2.sum(dim=1, keepdim=True)  # [B, 1]
        union = sum1 + sum2.t() - intersection  # [B, B]

        # Avoid division by zero
        union = torch.clamp(union, min=1e-8)

        # Jaccard similarity
        jaccard_sim = intersection / union  # [B, B]

        # Jaccard distance (dissimilarity)
        jaccard_distance = 1.0 - jaccard_sim

        return jaccard_distance

    def forward(self, z_img, z_txt, labels):
        """
        Args:
            z_img: [B, d_model] image embeddings
            z_txt: [B, d_model] text embeddings
            labels: [B, 14] disease labels

        Returns:
            loss: SACL loss value
        """
        batch_size = z_img.shape[0]

        # Numerical stability check: make sure the inputs contain no NaN/Inf
        if torch.isnan(z_img).any() or torch.isinf(z_img).any():
            return torch.tensor(0.0, device=z_img.device, requires_grad=True)
        if torch.isnan(z_txt).any() or torch.isinf(z_txt).any():
            return torch.tensor(0.0, device=z_img.device, requires_grad=True)

        # Normalize the embeddings (for cosine similarity)
        z_img_norm = F.normalize(z_img, p=2, dim=1)  # [B, d_model]
        z_txt_norm = F.normalize(z_txt, p=2, dim=1)  # [B, d_model]

        # Compute the cosine similarity matrix
        sim = torch.matmul(z_img_norm, z_txt_norm.t()) / self.temperature  # [B, B]

        # Numerical stability: clamp the similarities to prevent exponent overflow
        sim = torch.clamp(sim, min=-50, max=50)

        # Compute the Jaccard dissimilarity as the margin
        jaccard_dist = self.compute_jaccard_distance(labels, labels)  # [B, B]
        margin = self.margin_weight * jaccard_dist / self.temperature  # [B, B]

        # Numerical stability: clamp the margin
        margin = torch.clamp(margin, min=-50, max=50)

        # Set the margin of the diagonal (positive pairs) to 0
        mask = torch.eye(batch_size, device=labels.device).bool()
        margin = margin * (~mask).float()

        # Add the margin to the similarities
        sim_with_margin = sim + margin  # [B, B]

        # Numerical stability: clamp the margin-augmented similarities
        sim_with_margin = torch.clamp(sim_with_margin, min=-50, max=50)

        # Labels: the diagonal holds the positive pairs (index = 0, 1, 2, ..., B-1)
        labels_i2t = torch.arange(batch_size, device=labels.device)  # [B]
        labels_t2i = torch.arange(batch_size, device=labels.device)  # [B]

        # Bidirectional contrastive loss (i2t + t2i)
        # Image-to-Text: each row is one image's similarity to all texts
        loss_i2t = F.cross_entropy(sim_with_margin, labels_i2t)

        # Text-to-Image: each column is one text's similarity to all images
        loss_t2i = F.cross_entropy(sim_with_margin.t(), labels_t2i)

        loss = 0.5 * (loss_i2t + loss_t2i)

        # Final numerical stability check
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=z_img.device, requires_grad=True)

        return loss


def compute_loss_dgmk(output, reports_ids, reports_masks, labels=None, vis_label=None, txt_label=None,
                      z_img=None, z_txt=None, args={}, similarity_function='cosine'):
    """
    Loss function of DGMK-Net.

    Loss = L_lm + α_label * L_bce + α_contrastive * L_SACL

    Args:
        output: Transformer output logits
        reports_ids: target report token IDs
        reports_masks: target report masks
        labels: ground-truth labels [B, 14]
        vis_label: visual predicted labels [B, 14]
        txt_label: textual predicted labels [B, 14]
        z_img: image embeddings [B, d_model]
        z_txt: text embeddings [B, d_model]
        args: configuration arguments
        similarity_function: similarity function to use

    Returns:
        loss: total loss
    """
    global _criterion_lm, _criterion_bce, _sacl_loss

    # Language modeling loss - chosen based on the label_smoothing setting
    from modules.loss import LanguageModelCriterion, LabelSmoothing
    label_smoothing = getattr(args, 'label_smoothing', 0.0)

    if label_smoothing > 0:
        # Use LabelSmoothing (KL divergence)
        if _criterion_lm is None or not isinstance(_criterion_lm, LabelSmoothing):
            _criterion_lm = LabelSmoothing(smoothing=label_smoothing)
        loss_lm = _criterion_lm(output[:, :-1], reports_ids[:, 1:], reports_masks[:, 1:])
    else:
        # Use standard cross entropy
        if _criterion_lm is None or not isinstance(_criterion_lm, LanguageModelCriterion):
            _criterion_lm = LanguageModelCriterion()
        loss_lm = _criterion_lm(output[:, :-1], reports_ids[:, 1:], reports_masks[:, 1:]).mean()

    # Label loss (BCE) - reuse the singleton
    loss_bce = 0
    if args.label_loss:
        if _criterion_bce is None:
            _criterion_bce = torch.nn.BCEWithLogitsLoss()
        loss_bce = _criterion_bce(vis_label, labels)

    # Contrastive loss (SACL) - reuse the singleton
    loss_contrastive = 0
    if args.contrastive_loss and z_img is not None and z_txt is not None:
        if _sacl_loss is None or _sacl_loss.temperature != args.contrastive_temp:
            _sacl_loss = SymptomAwareContrastiveLoss(
                temperature=args.contrastive_temp,
                margin_weight=args.margin_weight
            )
        loss_contrastive = _sacl_loss(z_img, z_txt, labels)

    # Total loss
    alpha_label = getattr(args, 'alpha_label', 0.1)
    alpha_contrastive = getattr(args, 'alpha_contrastive', 0.1)

    total_loss = loss_lm + alpha_label * loss_bce + alpha_contrastive * loss_contrastive

    # Final safety valve: keep the loss non-negative and in a sane range (typically 0.5~15)
    total_loss = total_loss.clamp(min=0, max=100)

    return total_loss
