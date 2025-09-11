import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg') # Backend não-interativo para salvar figuras
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import argparse
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
import gc
import copy

# --- 1. FUNÇÕES DE CONFIGURAÇÃO ---

def set_seeds(seed=42):
    """Configura seeds para reprodutibilidade completa."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def setup_logging(log_dir, run_name):
    """Configura um logger para salvar em um diretório de execução específico."""
    os.makedirs(log_dir, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_filename = os.path.join(log_dir, f"{run_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# --- 2. PREPARAÇÃO E VALIDAÇÃO DE DADOS ---

class MedicalImageDataset(Dataset):
    """Dataset robusto com tratamento de exceções para imagens médicas."""
    def __init__(self, all_paths, all_labels, transform=None):
        self.image_paths = all_paths
        self.labels = all_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            label = self.labels[idx]
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None: raise ValueError(f"Imagem corrompida: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if self.transform: image = self.transform(image=image)["image"]
            return image, label, str(image_path)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Erro ao carregar {self.image_paths[idx]}: {e}. Retornando placeholder.")
            placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transform: placeholder = self.transform(image=placeholder)["image"]
            return placeholder, self.labels[idx], "FALLBACK_IMAGE"

def get_medical_transforms(image_size=224, is_training=True):
    """Gera transformações com Albumentations."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if is_training:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.7),
            A.RandomBrightnessContrast(p=0.7),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([A.Resize(image_size, image_size), A.Normalize(mean=mean, std=std), ToTensorV2()])

def get_all_data(data_dir, classes):
    """Carrega todos os caminhos de imagem e rótulos do diretório."""
    all_paths, all_labels = [], []
    for class_idx, class_name in enumerate(classes):
        paths = list((Path(data_dir) / class_name).glob('*[.png,.jpg,.jpeg]'))
        all_paths.extend(str(path) for path in paths)
        all_labels.extend([class_idx] * len(paths))
    if not all_paths: raise FileNotFoundError(f"Nenhuma imagem encontrada em: {data_dir}")
    return np.array(all_paths), np.array(all_labels)

# --- 3. ARQUITETURA DO MODELO (ADAPTADA) ---

class AdvancedMedicalClassifier(nn.Module):
    """Modelo híbrido que fusiona DenseNet-121 e ResNet-50."""
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__()
        # --- MUDANÇA AQUI: Backbone 1 agora é DenseNet-121 ---
        self.backbone1 = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        backbone1_features = self.backbone1.classifier.in_features
        self.backbone1.classifier = nn.Identity()
        
        # Backbone 2: ResNet-50 (mantido)
        self.backbone2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        backbone2_features = self.backbone2.fc.in_features
        self.backbone2.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(backbone1_features + backbone2_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        feat1 = self.backbone1(x)
        feat2 = self.backbone2(x)
        return self.classifier(torch.cat([feat1, feat2], dim=1))

# --- 4. LÓGICA DE TREINAMENTO E AVALIAÇÃO ---

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train(); running_loss = 0.0
    for inputs, targets, _ in dataloader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            loss = criterion(model(inputs), targets)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader, device):
    model.eval(); val_preds, val_true = [], []
    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
            val_preds.extend(torch.max(outputs, 1)[1].cpu().numpy())
            val_true.extend(targets.cpu().numpy())
    return f1_score(val_true, val_preds, average='macro', zero_division=0)

def evaluate_with_ensemble(test_loader, fold_models_paths, device, args):
    models_list = []
    for model_path in fold_models_paths:
        if os.path.exists(model_path):
            model = AdvancedMedicalClassifier(len(args.classes), pretrained=False, dropout=args.dropout)
            model.load_state_dict(torch.load(model_path)); model.to(device).eval()
            models_list.append(model)
    if not models_list: raise RuntimeError("Nenhum modelo de fold encontrado para o ensemble.")
    
    all_preds, all_true = [], []
    with torch.no_grad():
        for inputs, targets, _ in tqdm(test_loader, desc="Ensemble Evaluation"):
            inputs = inputs.to(device)
            batch_probs = [F.softmax(m(inputs), dim=1).cpu().numpy() for m in models_list]
            ensemble_preds = np.argmax(np.mean(batch_probs, axis=0), axis=1)
            all_preds.extend(ensemble_preds)
            all_true.extend(targets.numpy())
    return np.array(all_true), np.array(all_preds)

# --- 5. ORQUESTRADOR DE EXPERIMENTO ---

def run_experiment(args, logger, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_paths, all_labels = get_all_data(args.data_dir, args.classes)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=args.seed, stratify=all_labels
    )
    test_dataset = MedicalImageDataset(test_paths, test_labels, transform=get_medical_transforms(args.image_size, False))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    logger.info(f"--- Iniciando Validação Cruzada com {args.num_folds} Folds ---")
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_model_paths = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_paths, train_val_labels), 1):
        logger.info(f"\n{'='*15} FOLD {fold}/{args.num_folds} {'='*15}")
        
        train_ds = MedicalImageDataset(train_val_paths[train_idx], train_val_labels[train_idx], transform=get_medical_transforms(args.image_size, True))
        val_ds = MedicalImageDataset(train_val_paths[val_idx], train_val_labels[val_idx], transform=get_medical_transforms(args.image_size, False))
        
        class_weights = compute_class_weight('balanced', classes=np.unique(train_val_labels[train_idx]), y=train_val_labels[train_idx])
        sampler = WeightedRandomSampler([class_weights[lbl] for lbl in train_val_labels[train_idx]], len(train_idx))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        model = AdvancedMedicalClassifier(len(args.classes), dropout=args.dropout).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
        scaler = torch.amp.GradScaler('cuda')
        best_val_f1, patience_counter = 0.0, 0
        model_save_path = os.path.join(output_dir, 'models', f'model_fold_{fold}.pth')
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_f1 = validate_one_epoch(model, val_loader, device)
            logger.info(f"Fold {fold} Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1, patience_counter = val_f1, 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping ativado na época {epoch+1}.")
                break
        
        fold_model_paths.append(model_save_path)
        del model, optimizer, criterion, scaler, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache(); gc.collect()

    logger.info("\n--- Avaliação Final com Ensemble ---")
    y_true, y_pred = evaluate_with_ensemble(test_loader, fold_model_paths, device, args)
    
    logger.info("\n" + classification_report(y_true, y_pred, target_names=args.classes, zero_division=0))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', xticklabels=args.classes, yticklabels=args.classes)
    report_path = os.path.join(output_dir, 'reports')
    os.makedirs(report_path, exist_ok=True)
    plt.savefig(os.path.join(report_path, 'confusion_matrix.png'))
    plt.close()

# --- 6. BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Framework de Validação Cruzada (DenseNet+ResNet)")
    parser.add_argument('--data_dir', type=str, default='data/', help='Diretório raiz dos datasets.')
    parser.add_argument('--classes', nargs='+', default=['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'])
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=max(os.cpu_count() // 2, 2))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_folds', type=int, default=5)
    
    base_args = parser.parse_args()
    set_seeds(base_args.seed)

    root_data_dir = Path(base_args.data_dir)
    results_base_dir = "training_results_densenet_resnet"

    for dataset_path in [d for d in root_data_dir.iterdir() if d.is_dir()]:
        dataset_name = dataset_path.name
        run_name = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir = os.path.join(results_base_dir, run_name)
        logger = setup_logging(os.path.join(output_dir, 'logs'), run_name)
        
        logger.info(f"\n{'='*25} INICIANDO EXPERIMENTO PARA: {dataset_name.upper()} {'='*25}")
        
        run_args = copy.deepcopy(base_args)
        run_args.data_dir = str(dataset_path)
        
        run_experiment(run_args, logger, output_dir)
        
        logger.info(f"{'='*25} EXPERIMENTO FINALIZADO PARA: {dataset_name.upper()} {'='*25}\n")