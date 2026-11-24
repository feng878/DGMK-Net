import torch
import torch.nn as nn


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


class LabelSmoothing(nn.Module):
    """
    Label smoothing loss (numerically stable version).

    Computes the cross entropy directly with log_softmax + smoothed targets
    instead of KLDivLoss, avoiding cloned large tensors and extreme values.

    Mathematically equivalent to: KL(smoothed_target || softmax(logits))
    but computed in a more stable way: -smoothed_target @ log_softmax(logits)
    """

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing   # weight of the target token
        self.smoothing = smoothing           # spread uniformly over the other tokens
        self.padding_idx = padding_idx

    def forward(self, input, target, mask):
        """
        Args:
            input: Transformer output log-probs [B, seq_len, vocab_size]
            target: target token IDs [B, seq_len]
            mask: validity mask [B, seq_len]
        """
        # Align dimensions
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        # Flatten: [N, vocab_size], [N], [N]
        log_probs = input.reshape(-1, input.size(-1))   # log_softmax output
        target = target.reshape(-1)
        mask = mask.reshape(-1).to(log_probs)
        vocab_size = log_probs.size(1)

        # Make sure log_probs are valid (defensive: in case log_softmax was
        # forgotten upstream). log_softmax outputs are in (-inf, 0]; clamp
        # prevents extreme values
        log_probs = torch.clamp(log_probs, min=-50, max=0)

        # Method: compute the smoothed cross entropy directly
        # loss_i = -(confidence * log_probs[i, target_i] + smoothing * Σ log_probs[i, j])
        # where Σ sums over the vocabulary dimension

        # Contribution of the target token (non-negative, since log_probs <= 0 and confidence > 0)
        target_log_prob = log_probs.gather(1, target.unsqueeze(1).clamp(min=0)).squeeze(1)

        # Contribution of the uniform distribution (non-negative, since log_probs <= 0 and smoothing > 0)
        uniform_log_prob = log_probs.sum(dim=1)

        # Combined loss (always >= 0)
        loss_per_element = -(self.confidence * target_log_prob +
                            (self.smoothing / (vocab_size - 1)) * uniform_log_prob)

        # Apply the mask and average
        result = (loss_per_element * mask).sum() / mask.sum().clamp(min=1)

        # Final safety valve: keep the loss non-negative and not too large
        return result.clamp(min=0, max=100)
