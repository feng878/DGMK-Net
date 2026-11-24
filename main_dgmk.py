import warnings
warnings.simplefilter("ignore", UserWarning)

import torch
import argparse
import numpy as np
import logging
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.tokenizers import Tokenizer
from modules.dataloaders import LADataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import BaseTrainer
from modules.contrastive_loss import compute_loss_dgmk

# Import the DGMK-Net models
from models.dgmk_model import DGMKNet
from models.dgmk_model_single_scale import DGMKNetSingleScale

from config import opts
from tqdm import tqdm


class DGMKTrainer(BaseTrainer):
    """Trainer dedicated to DGMK-Net."""
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler,
                 train_dataloader, val_dataloader, test_dataloader):
        super(DGMKTrainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # AMP mixed precision training
        self.use_amp = getattr(args, 'use_amp', False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        # Gradient accumulation
        self.grad_accum_steps = max(1, getattr(args, 'grad_accum_steps', 1))

    def _train_epoch(self, epoch):
        """Train for one epoch."""
        train_loss = 0
        nan_count = 0       # number of NaN batches
        total_batches = 0   # total number of batches
        self.model.train()
        t = tqdm(self.train_dataloader, ncols=80)

        for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(t):
            total_batches += 1
            images, reports_ids, reports_masks, labels = images.to(self.device), reports_ids.to(self.device), \
                                                         reports_masks.to(self.device), labels.to(self.device)

            # Zero the gradients at the beginning of each batch so that NaN
            # gradient leftovers cannot pollute subsequent batches
            self.optimizer.zero_grad()

            # AMP forward pass
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(images, reports_ids, labels, mode='train')

                # DGMK-Net specific loss function
                loss = self.criterion(
                    *((outputs[0],) + (reports_ids, reports_masks, labels) + outputs[1:] + (self.args,))
                )
                # Gradient accumulation: scale the loss
                loss = loss / self.grad_accum_steps

            # NaN / extreme value detection: catch NaN/Inf as well as finite
            # but abnormal losses (e.g. -5.61e+26)
            if not torch.isfinite(loss) or abs(loss.item()) > 1e6:
                nan_count += 1
                t.set_description(f'loss:BAD [{nan_count}/{total_batches}]')
                continue

            train_loss += loss.item() * self.grad_accum_steps  # restore the true loss

            # AMP backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation: update the parameters only every N steps
            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                # AMP unscale + gradient clipping + step
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            t.set_description(f'loss:{loss.item() * self.grad_accum_steps:.3} [NaN:{nan_count}/{total_batches}]')

            # Periodic testing (optional)
            if self.args.test_steps > 0 and epoch > 1 and (batch_idx + 1) % self.args.test_steps == 0:
                self.model.eval()

        # End of the epoch: print the NaN statistics summary
        nan_rate = nan_count / total_batches * 100 if total_batches > 0 else 0
        status = "OK" if nan_rate < 1 else ("WARN" if nan_rate < 5 else "DANGER")
        print(f"\n[Epoch {epoch}] NaN summary: {nan_count}/{total_batches} batches ({nan_rate:.1f}%) - {status}")

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        # Validation set evaluation
        ilog = self._test_step(epoch, 0, 'val')
        log.update(**ilog)

        # Test set evaluation
        ilog = self._test_step(epoch, 0, 'test')
        log.update(**ilog)

        self.lr_scheduler.step()

        return log

    def _test_step(self, epoch, iters=0, mode='test'):
        """Evaluate one epoch on the validation/test split."""
        ilog = {}
        self.model.eval()
        data_loader = self.val_dataloader if mode == 'val' else self.test_dataloader

        with torch.no_grad():
            val_gts, val_res, val_idxs = [], [], []
            t = tqdm(data_loader, ncols=80)

            for batch_idx, (images_id, images, reports_ids, reports_masks, labels) in enumerate(t):
                images, reports_ids, reports_masks, labels = images.to(self.device), reports_ids.to(self.device), \
                                                             reports_masks.to(self.device), labels.to(self.device)

                outputs = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(outputs[0].cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                val_res.extend(reports)
                val_gts.extend(ground_truths)
                val_idxs.extend(images_id)

            # Compute the NLG metrics only (BLEU, METEOR, ROUGE-L, CIDEr)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})

            ilog.update(**{f'{mode}_' + k: v for k, v in val_met.items()})

            # Dump the generated reports
            self._output_generation(val_res, val_gts, val_idxs, epoch, iters, mode)

        return ilog


def main():
    # Parse arguments
    args = opts.parse_opt()
    logging.info(str(args))

    # Fix the random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Create the tokenizer
    tokenizer = Tokenizer(args)

    # Create the data loaders
    train_dataloader = LADataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = LADataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = LADataLoader(args, tokenizer, split='test', shuffle=False)

    # Build the model architecture
    # Select the model based on the no_multi_scale flag
    if getattr(args, 'no_multi_scale', False):
        logging.info("Using DGMKNet Single Scale (Ablation)")
        model = DGMKNetSingleScale(args, tokenizer)
    else:
        logging.info("Using DGMKNet Full Model")
        model = DGMKNet(args, tokenizer)
    
    logging.info(str(model))
    
    # Get the loss and metric function handles
    criterion = compute_loss_dgmk
    metrics = compute_scores
    
    # Build the optimizer and learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    
    # Build the trainer and start training
    trainer = DGMKTrainer(model, criterion, metrics, optimizer, args, lr_scheduler,
                          train_dataloader, val_dataloader, test_dataloader)
    trainer.train()
    
    logging.info(str(args))


if __name__ == '__main__':
    main()
