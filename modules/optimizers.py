import torch


def build_optimizer(args, model):
    ve_params = list(map(id, model.visual_extractor.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
         {'params': ed_params, 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer


def build_lr_scheduler(args, optimizer):
    if args.lr_scheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)
    elif args.lr_scheduler == "WarmupStepLR":
        # Warmup + StepLR: grows linearly for warmup_epochs, then StepLR decay
        lr_scheduler = WarmupStepLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            step_size=args.step_size,
            gamma=args.gamma
        )
    elif args.lr_scheduler == "cosine":
        print(f"Using CosineAnnealingWarmRestarts lr_scheduler, T_0={args.step_size}, T_mult={args.gamma}")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=args.step_size,
                                                                            T_mult=int(args.gamma)
                                                                            )
    else:
        raise NotImplementedError

    return lr_scheduler


class WarmupStepLR:
    """Warmup + StepLR learning rate scheduler.

    LR schedule:
    - Warmup phase: cosine interpolation growing smoothly from 10% to 100%
    - StepLR phase: decayed by gamma every step_size epochs

    Cosine warmup keeps the relative LR increase per epoch uniform and avoids
    the large early jumps of a linear warmup.
    """
    def __init__(self, optimizer, warmup_epochs=5, step_size=20, gamma=0.5, last_epoch=-1):
        import math
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.math = math

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        for i, param_group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if epoch < self.warmup_epochs:
                # Cosine warmup: grow smoothly from 10% to 100%.
                # Compared to linear warmup the early increase is gentler
                progress = (epoch + 1) / self.warmup_epochs  # 0->1
                warmup_ratio = 0.1 + 0.9 * (1 - self.math.cos(
                    self.math.pi * progress)) / 2
                param_group['lr'] = base_lr * warmup_ratio
            else:
                # StepLR phase: decay every step_size epochs
                step_epoch = epoch - self.warmup_epochs
                decay = self.gamma ** (step_epoch // self.step_size)
                param_group['lr'] = base_lr * decay
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
