import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import kornia as K
import kornia.augmentation as KA
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchattacks
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
import random
from pathlib import Path
import pickle


class HybridAugmentationPipeline:
    """
    Multi-Stage Hybrid Augmentation Pipeline for robust model training.
    
    Stages:
    1. Hybrid Pixel-Level Augmentation (Kornia + Albumentations)
    2. Batch-Level Mixing (MixUp & CutMix) 
    3. Adversarial Hardening (FGSM)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        p_mix: float = 0.5,
        p_adv: float = 0.25,
        fgsm_epsilon: float = 8/255,
        mixup_alpha: float = 1.0,
        cutmix_alpha: float = 1.0,
        kornia_strength: float = 0.5,
        albumentations_cache_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.image_size = image_size
        self.p_mix = p_mix
        self.p_adv = p_adv
        self.fgsm_epsilon = fgsm_epsilon
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.kornia_strength = kornia_strength
        self.device = device
        
        # Initialize augmentation components
        self._setup_kornia_augmentations()
        self._setup_albumentations()
        self.albumentations_cache = self._load_albumentations_cache(albumentations_cache_path)
        
    def _setup_kornia_augmentations(self):
        """Setup Kornia augmentations for fast GPU-based transforms."""
        self.kornia_augs = nn.Sequential(
            # Random Resized Crop
            KA.RandomResizedCrop(
                size=self.image_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33),
                p=0.8
            ),
            
            # Horizontal Flip
            KA.RandomHorizontalFlip(p=0.5),
            
            # Color Jitter
            KA.ColorJitter(
                brightness=0.2 * self.kornia_strength,
                contrast=0.2 * self.kornia_strength,
                saturation=0.2 * self.kornia_strength,
                hue=0.1 * self.kornia_strength,
                p=0.7
            ),
            
            # Gaussian Blur
            KA.RandomGaussianBlur(
                kernel_size=(3, 3),
                sigma=(0.1, 2.0),
                p=0.3
            ),
            
            # Motion Blur
            KA.RandomMotionBlur(
                kernel_size=3,
                angle=(-45, 45),
                direction=(0.0, 1.0),
                p=0.3
            ),
            
            # Random Rotation
            KA.RandomRotation(
                degrees=15 * self.kornia_strength,
                p=0.4
            ),
            
            # Random Perspective
            KA.RandomPerspective(
                distortion_scale=0.2 * self.kornia_strength,
                p=0.3
            ),
            
            # Random Erasing
            KA.RandomErasing(
                scale=(0.02, 0.1),
                ratio=(0.3, 3.3),
                p=0.3
            )
        ).to(self.device)
    
    def _setup_albumentations(self):
        """Setup Albumentations for complex CPU-based transforms."""
        self.albumentations_transform = A.Compose([
            # Advanced noise models
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], per_channel=True, p=1.0)
            ], p=0.6),
            
            # Grid and elastic distortions
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=1.0)
            ], p=0.4),
            
            # Weather and sensor effects
            A.OneOf([
                A.RandomRain(slant_range=(-10, 10), drop_length=20, drop_width=1, 
                           drop_color=(200, 200, 200), blur_value=7, p=1.0),
                A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=1.0),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_range=(0, 1), p=1.0)
            ], p=0.3),
            
            # Compression artifacts
            A.OneOf([
                A.JpegCompression(quality_range=(70, 95), p=1.0),
                A.Downscale(scale_range=(0.7, 0.9), p=1.0),
                A.ImageCompression(quality_range=(70, 95), p=1.0)
            ], p=0.4),
            
            # Additional challenging transforms
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.ChannelShuffle(p=0.2),
        ])
    
    def _load_albumentations_cache(self, cache_path: Optional[str]) -> Optional[Dict]:
        """Load pre-computed Albumentations cache if available."""
        if cache_path and Path(cache_path).exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    def create_albumentations_cache(self, dataset, cache_path: str, cache_size: int = 10000):
        """Create a cache of heavily augmented samples using Albumentations."""
        print(f"Creating Albumentations cache with {cache_size} samples...")
        cache = {'images': [], 'labels': []}
        
        for i in range(min(cache_size, len(dataset))):
            if i % 1000 == 0:
                print(f"Processed {i}/{cache_size} samples")
            
            image, label = dataset[i]
            
            # Convert tensor to numpy if needed
            if isinstance(image, torch.Tensor):
                image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            else:
                image_np = np.array(image)
            
            # Apply Albumentations
            augmented = self.albumentations_transform(image=image_np)
            aug_image = augmented['image']
            
            # Convert back to tensor
            aug_tensor = torch.from_numpy(aug_image).permute(2, 0, 1).float() / 255.0
            
            cache['images'].append(aug_tensor)
            cache['labels'].append(label)
        
        # Save cache
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        
        self.albumentations_cache = cache
        print(f"Cache saved to {cache_path}")
    
    def stage1_pixel_augmentation(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Stage 1: Hybrid Pixel-Level Augmentation
        
        Args:
            batch: Input batch of images [B, C, H, W]
        
        Returns:
            Augmented batch
        """
        batch = batch.to(self.device)
        
        # Apply Kornia augmentations (fast, on-the-fly)
        with torch.no_grad():
            kornia_augmented = self.kornia_augs(batch)
        
        # Optionally mix in samples from Albumentations cache
        if self.albumentations_cache and random.random() < 0.3:  # 30% chance to use cache
            batch_size = batch.shape[0]
            n_replace = max(1, batch_size // 4)  # Replace 25% of batch
            
            # Randomly select samples from cache
            cache_indices = random.sample(range(len(self.albumentations_cache['images'])), n_replace)
            replace_indices = random.sample(range(batch_size), n_replace)
            
            for i, cache_idx in zip(replace_indices, cache_indices):
                cache_sample = self.albumentations_cache['images'][cache_idx].to(self.device)
                # Resize if needed
                if cache_sample.shape != kornia_augmented[i].shape:
                    cache_sample = F.interpolate(
                        cache_sample.unsqueeze(0), 
                        size=self.image_size, 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                kornia_augmented[i] = cache_sample
        
        return kornia_augmented
    
    def stage2_batch_mixing(
        self, 
        batch: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2: Batch-Level Mixing (MixUp & CutMix)
        
        Args:
            batch: Input batch [B, C, H, W]
            labels: Corresponding labels [B]
        
        Returns:
            Mixed batch and labels
        """
        if random.random() > self.p_mix:
            return batch, labels
        
        batch_size = batch.shape[0]
        
        # Choose between MixUp and CutMix
        if random.random() < 0.5:
            # MixUp
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(batch_size).to(self.device)
            
            mixed_batch = lam * batch + (1 - lam) * batch[index]
            
            # For classification, create soft labels
            if labels.dim() == 1:  # Hard labels
                mixed_labels = torch.zeros(batch_size, labels.max().item() + 1).to(self.device)
                mixed_labels.scatter_(1, labels.unsqueeze(1), lam)
                mixed_labels.scatter_add_(1, labels[index].unsqueeze(1), 1 - lam)
            else:  # Already soft labels
                mixed_labels = lam * labels + (1 - lam) * labels[index]
                
        else:
            # CutMix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            index = torch.randperm(batch_size).to(self.device)
            
            # Generate random bounding box
            W, H = batch.shape[3], batch.shape[2]
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)
            
            cx = np.random.randint(W)
            cy = np.random.randint(H)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
            
            mixed_batch = batch.clone()
            mixed_batch[:, :, bby1:bby2, bbx1:bbx2] = batch[index, :, bby1:bby2, bbx1:bbx2]
            
            # Adjust lambda based on actual cut area
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            
            # Create mixed labels
            if labels.dim() == 1:  # Hard labels
                mixed_labels = torch.zeros(batch_size, labels.max().item() + 1).to(self.device)
                mixed_labels.scatter_(1, labels.unsqueeze(1), lam)
                mixed_labels.scatter_add_(1, labels[index].unsqueeze(1), 1 - lam)
            else:  # Already soft labels
                mixed_labels = lam * labels + (1 - lam) * labels[index]
        
        return mixed_batch, mixed_labels
    
    def stage3_adversarial_hardening(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module
    ) -> torch.Tensor:
        """
        Stage 3: Adversarial Hardening using FGSM
        
        Args:
            batch: Input batch [B, C, H, W]
            labels: Corresponding labels
            model: The model being trained
            
        Returns:
            Adversarially perturbed batch
        """
        if random.random() > self.p_adv:
            return batch
        
        # Create FGSM attack
        attack = torchattacks.FGSM(model, eps=self.fgsm_epsilon)
        
        # Generate adversarial examples
        model.eval()
        with torch.enable_grad():
            # Handle soft labels for FGSM
            if labels.dim() > 1:  # Soft labels from mixing
                # Use hard labels (argmax) for adversarial attack
                hard_labels = labels.argmax(dim=1)
            else:
                hard_labels = labels
            
            adv_batch = attack(batch, hard_labels)
        
        model.train()
        return adv_batch
    
    def __call__(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor,
        model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply the complete multi-stage augmentation pipeline.
        
        Args:
            batch: Input batch [B, C, H, W]
            labels: Input labels [B] or [B, num_classes]
            model: The model being trained
            
        Returns:
            Final augmented batch and mixed labels
        """
        # Stage 1: Hybrid Pixel-Level Augmentation
        x_aug = self.stage1_pixel_augmentation(batch)
        
        # Stage 2: Batch-Level Mixing
        x_mix, y_mix = self.stage2_batch_mixing(x_aug, labels)
        
        # Stage 3: Adversarial Hardening
        x_final = self.stage3_adversarial_hardening(x_mix, y_mix, model)
        
        return x_final, y_mix
    
    def update_curriculum(self, epoch: int, total_epochs: int):
        """
        Update augmentation strength based on training curriculum.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of training epochs
        """
        progress = epoch / total_epochs
        
        # Gradually increase augmentation strength
        self.kornia_strength = 0.3 + 0.4 * progress  # 0.3 -> 0.7
        self.p_mix = 0.3 + 0.3 * progress  # 0.3 -> 0.6
        self.p_adv = 0.1 + 0.2 * progress  # 0.1 -> 0.3
        
        # Recreate Kornia augmentations with new strength
        self._setup_kornia_augmentations()
    
    def mixed_loss(self, pred: torch.Tensor, y_mixed: torch.Tensor, criterion) -> torch.Tensor:
        """
        Compute loss for mixed labels (from MixUp/CutMix).
        
        Args:
            pred: Model predictions [B, num_classes]
            y_mixed: Mixed labels [B, num_classes] or [B]
            criterion: Loss function
            
        Returns:
            Computed loss
        """
        if y_mixed.dim() == 1:  # Hard labels
            return criterion(pred, y_mixed)
        else:  # Soft labels from mixing
            # Compute soft cross-entropy
            log_probs = F.log_softmax(pred, dim=1)
            loss = -(y_mixed * log_probs).sum(dim=1).mean()
            return loss


class RobustTrainingLoop:
    """
    Example training loop using the hybrid augmentation pipeline.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        augmentation_pipeline: HybridAugmentationPipeline,
        device: str = 'cuda'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.aug_pipeline = augmentation_pipeline
        self.device = device
        
        self.model.to(device)
    
    def train_epoch(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Train for one epoch with augmentation pipeline."""
        self.model.train()
        self.aug_pipeline.update_curriculum(epoch, total_epochs)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Apply augmentation pipeline
            aug_data, mixed_target = self.aug_pipeline(data, target, self.model)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(aug_data)
            
            # Compute loss (handles mixed labels)
            loss = self.aug_pipeline.mixed_loss(output, mixed_target, self.criterion)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return {'train_loss': total_loss / num_batches}
    
    def validate(self) -> Dict[str, float]:
        """Validate model on clean validation set."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total
        avg_loss = val_loss / len(self.val_loader)
        
        return {'val_loss': avg_loss, 'val_accuracy': accuracy}
