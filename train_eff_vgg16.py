# -*- coding: utf-8 -*-
"""
Medical Image Classification Framework with Ensemble Learning
Refactored for better modularity, maintainability and extensibility.
"""

import os
import gc
import copy
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import models
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    data_dir: str
    classes: List[str]
    image_size: int = 224
    dropout: float = 0.5
    epochs: int = 25
    learning_rate: float = 3e-4
    batch_size: int = 16
    num_workers: int = 4
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    weight_decay: float = 1e-4
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        if len(self.classes) < 2:
            raise ValueError("At least 2 classes are required")


class Logger:
    """Centralized logging management."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        
        # Clear existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def get_logger(self) -> logging.Logger:
        return self.logger


class RandomSeedManager:
    """Manages random seed setting for reproducibility."""
    
    @staticmethod
    def set_seeds(seed: int = 42):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class MedicalImageTransforms:
    """Handles image transformations for medical images."""
    
    def __init__(self, image_size: int = 224):
        self.image_size = image_size
        self.imagenet_mean = [0.485, 0.456, 0.406]
        self.imagenet_std = [0.229, 0.224, 0.225]
    
    def get_training_transforms(self) -> A.Compose:
        """Get training transforms with data augmentation."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.7),
            A.RandomBrightnessContrast(p=0.7),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
            ToTensorV2(),
        ])
    
    def get_validation_transforms(self) -> A.Compose:
        """Get validation transforms without augmentation."""
        return A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean=self.imagenet_mean, std=self.imagenet_std),
            ToTensorV2()
        ])


class MedicalImageDataset(Dataset):
    """Dataset class for medical images."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Corrupted or missing image: {image_path}")
        
        # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)["image"]
        
        return image, label, str(image_path)


class DataManager:
    """Manages data loading and splitting."""
    
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def prepare_data_splits(self) -> Dict[str, Tuple[List[str], List[int]]]:
        """Prepare train/validation/test splits."""
        all_paths, all_labels = [], []
        
        # Collect all image paths and labels
        for class_idx, class_name in enumerate(self.config.classes):
            class_path = Path(self.config.data_dir) / class_name
            if not class_path.is_dir():
                self.logger.error(f"Class directory not found: {class_path}")
                continue
            
            # Find all image files
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
            paths = []
            for ext in extensions:
                paths.extend(class_path.glob(ext))
            
            all_paths.extend(str(p) for p in paths)
            all_labels.extend([class_idx] * len(paths))
        
        if not all_paths:
            raise FileNotFoundError(f"No images found in directory: {self.config.data_dir}")
        
        # Split data
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_paths, all_labels, 
            test_size=self.config.test_size, 
            random_state=self.config.seed, 
            stratify=all_labels
        )
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=self.config.val_size / (1 - self.config.test_size),
            random_state=self.config.seed,
            stratify=train_val_labels
        )
        
        self.logger.info(f"Dataset split - Train: {len(train_paths)}, "
                        f"Validation: {len(val_paths)}, Test: {len(test_paths)}")
        
        return {
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels),
            'test': (test_paths, test_labels)
        }
    
    def create_data_loaders(self, data_splits: Dict) -> Dict[str, DataLoader]:
        """Create data loaders for training, validation, and testing."""
        transforms = MedicalImageTransforms(self.config.image_size)
        
        # Create datasets
        datasets = {
            'train': MedicalImageDataset(
                data_splits['train'][0], 
                data_splits['train'][1], 
                transforms.get_training_transforms()
            ),
            'val': MedicalImageDataset(
                data_splits['val'][0], 
                data_splits['val'][1], 
                transforms.get_validation_transforms()
            ),
            'test': MedicalImageDataset(
                data_splits['test'][0], 
                data_splits['test'][1], 
                transforms.get_validation_transforms()
            )
        }
        
        # Calculate class weights for balanced sampling
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(data_splits['train'][1]), 
            y=data_splits['train'][1]
        )
        
        # Create weighted sampler for training
        sample_weights = [class_weights[label] for label in data_splits['train'][1]]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders
        dataloaders = {
            'train': DataLoader(
                datasets['train'],
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=self.config.num_workers,
                pin_memory=True
            ),
            'val': DataLoader(
                datasets['val'],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            ),
            'test': DataLoader(
                datasets['test'],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        }
        
        return dataloaders, class_weights


class EfficientNetVGGEnsemble(nn.Module):
    """Ensemble model combining EfficientNet-B4 and VGG16."""
    
    def __init__(self, num_classes: int, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        
        # EfficientNet-B4 backbone
        self.efficientnet = timm.create_model(
            'efficientnet_b4', 
            pretrained=pretrained, 
            num_classes=0
        )
        
        # VGG16 backbone
        self.vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.vgg.classifier = nn.Identity()
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(1792 + 25088, 1024),  # EfficientNet-B4: 1792, VGG16: 25088
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract features from both backbones
        eff_features = self.efficientnet(x)  # Shape: (batch, 1792)
        
        vgg_features = self.vgg.features(x)  # Shape: (batch, 512, 7, 7)
        vgg_features = torch.flatten(vgg_features, 1)  # Shape: (batch, 25088)
        
        # Combine features
        combined_features = torch.cat([eff_features, vgg_features], dim=1)
        
        # Final classification
        output = self.classifier(combined_features)
        return output


class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability."""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        self.hook_handles.append(
            self.target_layer.register_forward_hook(self._forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(self._backward_hook)
        )
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
    
    def generate_heatmap(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """Generate GradCAM heatmap."""
        self.model.eval()
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate heatmap
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
        
        return heatmap.cpu().numpy()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []


class Visualizer:
    """Handles visualization and saving of results."""
    
    def __init__(self, output_dir: str, classes: List[str]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.classes = classes
    
    def save_confusion_matrix(self, y_true: List[int], y_pred: List[int], title: str = "Confusion Matrix"):
        """Save confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes
        )
        plt.title(title)
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        
        cm_path = self.output_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(cm_path)
    
    @staticmethod
    def unnormalize_image(tensor: torch.Tensor) -> np.ndarray:
        """Unnormalize image tensor for visualization."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        tensor = tensor.clone().permute(1, 2, 0).cpu().numpy()
        unnorm = (tensor * std + mean) * 255
        return np.clip(unnorm, 0, 255).astype(np.uint8)
    
    def generate_gradcam_visualizations(
        self, 
        model: nn.Module, 
        target_layer: nn.Module,
        dataloader: DataLoader, 
        device: torch.device,
        logger: logging.Logger,
        image_size: int = 224,
        num_images: int = 1
    ):
        """Generate and save GradCAM visualizations."""
        gradcam_dir = self.output_dir / "gradcam"
        gradcam_dir.mkdir(exist_ok=True)
        
        grad_cam = GradCAM(model, target_layer)
        class_counts = defaultdict(int)
        
        logger.info(f"Generating GradCAM for {num_images} example(s) per class...")
        
        for inputs, targets, paths in dataloader:
            if all(count >= num_images for count in class_counts.values()) and \
               len(class_counts) == len(self.classes):
                break
            
            for i in range(inputs.size(0)):
                true_label_idx = targets[i].item()
                true_label_name = self.classes[true_label_idx]
                
                if class_counts[true_label_name] < num_images:
                    img_tensor = inputs[i].unsqueeze(0).to(device)
                    heatmap = grad_cam.generate_heatmap(img_tensor, true_label_idx)
                    
                    # Prepare original image
                    original_img = self.unnormalize_image(inputs[i])
                    
                    # Resize heatmap and apply colormap
                    heatmap_resized = cv2.resize(heatmap, (image_size, image_size))
                    heatmap_color = cv2.applyColorMap(
                        np.uint8(255 * heatmap_resized), 
                        cv2.COLORMAP_JET
                    )
                    
                    # Superimpose heatmap on original image
                    superimposed = cv2.addWeighted(heatmap_color, 0.5, original_img, 0.5, 0)
                    
                    # Save image
                    filename = gradcam_dir / f"{true_label_name}_{Path(paths[i]).stem}_gradcam.png"
                    cv2.imwrite(str(filename), superimposed)
                    
                    class_counts[true_label_name] += 1
        
        grad_cam.remove_hooks()
        logger.info(f"GradCAM images saved to: {gradcam_dir}")


class Trainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: TrainingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
    
    def train_epoch(
        self, 
        model: nn.Module, 
        dataloader: DataLoader, 
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scaler: torch.amp.GradScaler
    ) -> float:
        """Train model for one epoch."""
        model.train()
        running_loss = 0.0
        
        for inputs, targets, _ in tqdm(dataloader, desc="Training"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
        
        return running_loss / len(dataloader.dataset)
    
    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> Tuple[List[int], List[int]]:
        """Evaluate model on given dataloader."""
        model.eval()
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for inputs, targets, _ in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())
        
        return true_labels, predictions
    
    def train_model(
        self, 
        dataloaders: Dict[str, DataLoader], 
        class_weights: np.ndarray,
        output_dir: str
    ) -> nn.Module:
        """Complete training pipeline."""
        # Initialize model
        model = EfficientNetVGGEnsemble(
            num_classes=len(self.config.classes),
            dropout=self.config.dropout
        ).to(self.device)
        
        # Initialize optimizer and loss
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        criterion = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(class_weights).to(self.device)
        )
        
        scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        
        # Training loop
        best_val_f1 = 0.0
        model_save_path = Path(output_dir) / "best_model.pth"
        
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.epochs):
            # Train
            train_loss = self.train_epoch(
                model, dataloaders['train'], optimizer, criterion, scaler
            )
            
            # Validate
            val_true, val_pred = self.evaluate(model, dataloaders['val'])
            val_f1 = f1_score(val_true, val_pred, average='macro', zero_division=0)
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs}: "
                f"Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}"
            )
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), model_save_path)
                self.logger.info(f"New best model saved! Val F1: {best_val_f1:.4f}")
        
        # Load best model for final evaluation
        model.load_state_dict(torch.load(model_save_path))
        return model


class MedicalImageClassifier:
    """Main class orchestrating the entire pipeline."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Set up experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = Path(config.data_dir).name
        self.experiment_name = f"{dataset_name}_{timestamp}"
        self.output_dir = Path("experiments") / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        logger_manager = Logger(str(self.output_dir), dataset_name)
        self.logger = logger_manager.get_logger()
        
        # Set random seeds
        RandomSeedManager.set_seeds(config.seed)
        
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Configuration: {config}")
    
    def run(self):
        """Run the complete pipeline."""
        try:
            # Prepare data
            data_manager = DataManager(self.config, self.logger)
            data_splits = data_manager.prepare_data_splits()
            dataloaders, class_weights = data_manager.create_data_loaders(data_splits)
            
            # Train model
            trainer = Trainer(self.config, self.logger)
            model = trainer.train_model(dataloaders, class_weights, str(self.output_dir))
            
            # Final evaluation
            self.logger.info("Evaluating on test set...")
            test_true, test_pred = trainer.evaluate(model, dataloaders['test'])
            
            # Generate report
            report = classification_report(
                test_true, test_pred, 
                target_names=self.config.classes,
                zero_division=0
            )
            self.logger.info(f"Test Classification Report:\n{report}")
            
            # Save visualizations
            visualizer = Visualizer(str(self.output_dir), self.config.classes)
            
            # Confusion matrix
            cm_path = visualizer.save_confusion_matrix(
                test_true, test_pred, 
                f"Confusion Matrix - {Path(self.config.data_dir).name}"
            )
            self.logger.info(f"Confusion matrix saved: {cm_path}")
            
            # GradCAM visualizations
            visualizer.generate_gradcam_visualizations(
                model=model,
                target_layer=model.efficientnet.conv_head,
                dataloader=dataloaders['test'],
                device=trainer.device,
                logger=self.logger,
                image_size=self.config.image_size,
                num_images=2
            )
            
            self.logger.info(f"Experiment completed successfully: {self.experiment_name}")
            
        except Exception as e:
            self.logger.critical(f"Fatal error occurred: {e}", exc_info=True)
            raise
        
        finally:
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Medical Image Classification with Ensemble Learning"
    )
    parser.add_argument(
        '--data_dir', 
        type=str, 
        required=True,
        help='Root directory containing class subdirectories'
    )
    parser.add_argument(
        '--classes', 
        nargs='+', 
        default=['Covid', 'Normal', 'Pneumonia_bacteriana', 'Pneumonia_viral'],
        help='List of class names'
    )
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        data_dir=args.data_dir,
        classes=args.classes,
        image_size=args.image_size,
        dropout=args.dropout,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Run experiment
    classifier = MedicalImageClassifier(config)
    classifier.run()


if __name__ == '__main__':
    main()