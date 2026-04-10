import math
import torch


def set_task_specific_params(model, decay=1e-4):
    # 1. Collect parameters for each component
    backbone_params = []
    neck_head_params = []
    decoder_params = []

    # Map the modules to their desired LR
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'backbone' in name:
            backbone_params.append(param)
        elif 'neck' in name or 'head' in name:
            neck_head_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        # Note: If there are shared parameters, you must choose one group for them.

    # 2. Define the parameter groups with specific LRs
    param_groups = [
        # Group 1: Backbone (Very low LR for fine-tuning/domain adaptation)
        {'params': backbone_params, 'lr': 1e-4, 'weight_decay': decay},

        # Group 2: Neck & Head (Moderate LR for task adaptation)
        {'params': neck_head_params, 'lr': 1e-4, 'weight_decay': decay},

        # Group 3: LLIE Decoder (High LR for new task training)
        {'params': decoder_params, 'lr': 1e-3, 'weight_decay': decay}
    ]

    return param_groups


class CosineAnnealingWarmupLR:
    """Cosine Annealing with Linear Warmup Scheduler."""
    def __init__(self, optimizer, total_steps, warmup_steps):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step_count = 0  # internal counter

    def step(self):
        """Increment internal step and apply new learning rates."""
        self.step_count += 1
        step = self.step_count

        if step < self.warmup_steps:
            # Linear warmup: 0 → 1
            multiplier = step / self.warmup_steps
        else:
            # Cosine annealing decay: 1 → 0
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Apply multiplier to all param groups
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.initial_lrs[i] * multiplier



class CompositeScoreCallback:
    """Monitors a composite score based on weighted metrics and saves the model."""
    def __init__(self, model_name: str, save_path_template: str = 'best_{}_composite.pt'):
        self.save_path = save_path_template.format(model_name)
        self.best_score = -float('inf')

        # Define weights and PSNR normalization factor
        self.weights = {
            'psnr': 0.05 / 40.0,  # Normalized PSNR
            'ssim': 0.30,
            'precision': 0.05,
            'recall': 0.05,
            'mAP50': 0.55
        }
        self.best_epoch = -1

    def calculate_score(self, metrics: dict) -> float:
        """Calculates the weighted composite score for the current epoch."""
        score = (
            metrics['val_psnr'] * self.weights['psnr'] +
            metrics['val_ssim'] * self.weights['ssim'] +
            metrics['val_precision'] * self.weights['precision'] +
            metrics['val_recall'] * self.weights['recall'] +
            metrics['val_mAP50'] * self.weights['mAP50']
        )
        return score

    def check_and_save(self, model, epoch_history: dict, epoch: int):
        current_score = self.calculate_score(epoch_history)

        if current_score > self.best_score:
            print(f"\n🏆 **Composite Score Improved!** ({current_score:.4f} > {self.best_score:.4f}). Saving model weights at Epoch {epoch}.")

            self.best_score = current_score
            self.best_epoch = epoch

            torch.save(model.state_dict(), self.save_path)
            print(f"Model saved to {self.save_path}")
            return True
        else:
            print(f"\nSkipping save at Epoch {epoch}. Composite Score ({current_score:.4f}) did not improve over best ({self.best_score:.4f}).")
            return False


class BestMAPCallback:
    def __init__(self, model_name: str, save_path_template='best_{}_mAP.pt'):
        self.save_path = save_path_template.format(model_name)
        self.best_map = -float('inf')
        self.best_epoch = -1

    def check_and_save(self, model, epoch_history: dict, epoch: int):
        current_map = epoch_history.get('val_mAP50', None)

        if current_map is None:
            print("⚠️ val_mAP50 not available, skipping checkpoint.")
            return False

        if current_map > self.best_map:
            print(
                f"\n🏆 **mAP Improved!** "
                f"({current_map:.4f} > {self.best_map:.4f}) at Epoch {epoch}. Saving model."
            )

            self.best_map = current_map
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.save_path)
            return True

        print(
            f"\nNo improvement in mAP "
            f"({current_map:.4f} ≤ {self.best_map:.4f}). Skipping save."
        )
        return False
