# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
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
import timm # Adicionado para carregar o modelo Xception
import gc
import copy
from collections import defaultdict

# --- 1. FUNÇÕES DE CONFIGURAÇÃO ---
def set_seeds(seed=42):
    """Configura seeds para reprodutibilidade."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_dir, dataset_name):
    """Configura um logger específico para cada treinamento."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{log_dir}/{dataset_name}_{timestamp}.log"
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# --- 2. PREPARAÇÃO DE DADOS ---
class MedicalImageDataset(Dataset):
    """Dataset para carregar imagens médicas."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError(f"Imagem corrompida ou não encontrada: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if self.transform: image = self.transform(image=image)["image"]
        return image, label, str(image_path)

def get_medical_transforms(image_size=299, is_training=True):
    """Retorna transformações de imagem (data augmentation)."""
    # Usando a normalização padrão da ImageNet, que funciona bem para a maioria dos modelos
    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if is_training:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.7),
            A.RandomBrightnessContrast(p=0.7),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ])
    else: # Validação e Teste
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2()
        ])

def prepare_data(data_dir, classes, logger, test_size=0.2, val_size=0.2, random_state=42):
    """Divide o dataset em conjuntos de treino, validação e teste."""
    all_paths, all_labels = [], []
    for class_idx, class_name in enumerate(classes):
        class_path = Path(data_dir) / class_name
        if not class_path.is_dir():
            logger.error(f"Diretório da classe não encontrado: {class_path}")
            continue
        paths = list(class_path.glob('*[.png,.jpg,.jpeg]'))
        all_paths.extend(str(p) for p in paths)
        all_labels.extend([class_idx] * len(paths))
    
    if not all_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada no diretório: {data_dir}")

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=train_val_labels)
    
    logger.info(f"Dataset dividido: {len(train_paths)} treino, {len(val_paths)} validação, {len(test_paths)} teste.")
    return {'train': (train_paths, train_labels), 'val': (val_paths, val_labels), 'test': (test_paths, test_labels)}

# --- 3. ARQUITETURA DO MODELO (MODIFICADO PARA XCEPTION) ---
class XceptionClassifier(nn.Module):
    """Classificador baseado na arquitetura Xception."""
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__()
        # Carrega o modelo Xception pré-treinado (sem o classificador final)
        self.backbone = timm.create_model('xception', pretrained=pretrained, num_classes=0)
        
        # Congela os parâmetros do backbone para transfer learning
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Número de características na saída do backbone (para Xception, é 2048)
        num_features = self.backbone.num_features
        
        # Cria um novo classificador customizado
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Passa os dados pelo backbone para extrair características
        # O modelo do timm com num_classes=0 já aplica o Global Average Pooling
        features = self.backbone(x)
        # Passa as características pelo novo classificador
        return self.classifier(features)

# --- 4. VISUALIZAÇÃO E EXPLICABILIDADE (GRAD-CAM) ---
class GradCAM:
    """Implementação do Grad-CAM para visualização de ativações."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        self.hook_handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, input, output): self.activations = output.detach()
    def _backward_hook(self, module, grad_in, grad_out): self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        
        # O forward pass completo dispara os hooks
        output = self.model(x)
        
        if class_idx is None: class_idx = output.argmax(dim=1).item()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Pondera os mapas de ativação pelos gradientes
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        if torch.max(heatmap) > 0: heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().numpy()

    def remove_hooks(self):
        for h in self.hook_handles: h.remove()
        self.hook_handles = []

def unnormalize_image(tensor):
    """Reverte a normalização de um tensor de imagem para visualização."""
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().permute(1, 2, 0).cpu().numpy()
    return np.clip((tensor * std + mean) * 255, 0, 255).astype(np.uint8)

def visualize_and_save_grad_cam(model, target_layer, dataloader, device, classes, logger, output_dir, image_size, num_images=1):
    os.makedirs(output_dir, exist_ok=True)
    grad_cam = GradCAM(model, target_layer)
    class_counts = defaultdict(int)
    num_classes = len(classes)
    
    logger.info(f"Gerando Grad-CAM para {num_images} exemplo(s) de cada classe...")
    
    for inputs, targets, paths in dataloader:
        if len(class_counts) == num_classes and all(c >= num_images for c in class_counts.values()): break
        
        for i in range(inputs.size(0)):
            true_label_idx = targets[i].item()
            true_label_name = classes[true_label_idx]
            
            if class_counts[true_label_name] < num_images:
                img_tensor = inputs[i].unsqueeze(0).to(device)
                heatmap = grad_cam(img_tensor, class_idx=true_label_idx)
                
                original_img_rgb = unnormalize_image(inputs[i])
                heatmap_resized = cv2.resize(heatmap, (image_size, image_size))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                
                superimposed_img = cv2.addWeighted(heatmap_color, 0.5, original_img_rgb, 0.5, 0)
                
                filename = f"{output_dir}/{true_label_name}_{Path(paths[i]).stem}_gradcam.png"
                cv2.imwrite(filename, superimposed_img)
                class_counts[true_label_name] += 1
                
    grad_cam.remove_hooks()
    logger.info(f"Imagens Grad-CAM salvas em '{output_dir}'.")

# --- 5. LÓGICA DE TREINAMENTO E AVALIAÇÃO ---
def train_and_evaluate(args, logger):
    """Função principal que orquestra o treinamento e a avaliação."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Usando dispositivo: {device}")
    
    data_splits = prepare_data(args.data_dir, args.classes, logger, random_state=args.seed)
    
    transforms = {k: get_medical_transforms(args.image_size, is_training=(k == 'train')) for k in ['train', 'val', 'test']}
    datasets = {k: MedicalImageDataset(data_splits[k][0], data_splits[k][1], transform=transforms[k]) for k in ['train', 'val', 'test']}
    
    class_weights = compute_class_weight('balanced', classes=np.unique(data_splits['train'][1]), y=data_splits['train'][1])
    sampler = WeightedRandomSampler(weights=[class_weights[lbl] for lbl in data_splits['train'][1]], num_samples=len(data_splits['train'][1]), replacement=True)

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True),
        'val': DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        'test': DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    }
    
    # **MODIFICAÇÃO: Instancia o modelo Xception**
    model = XceptionClassifier(num_classes=len(args.classes), dropout=args.dropout).to(device)
    
    # Para o Xception, treinamos apenas o classificador novo
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    scaler = torch.amp.GradScaler(device='cuda', enabled=torch.cuda.is_available())
    
    best_val_f1 = 0.0

    logger.info("--- Iniciando Treinamento com Xception ---")
    for epoch in range(args.epochs):
        model.train() # Ativa o treino para o classificador
        model.backbone.eval() # Mantém o backbone em modo de avaliação (importante para BatchNorm)
        
        running_loss = 0.0
        for inputs, targets, _ in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
        
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for inputs, targets, _ in dataloaders['val']:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(targets.cpu().numpy())
        
        val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
        epoch_loss = running_loss / len(datasets['train'])
        logger.info(f"Epoch {epoch+1}/{args.epochs}: Train Loss: {epoch_loss:.4f}, Val F1-Macro: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Salvamos o estado do classificador, que é a única parte treinada
            torch.save(model.state_dict(), args.model_save_path)
            logger.info(f"Modelo salvo! Novo melhor Val F1-Score: {best_val_f1:.4f}")
            
    logger.info("\n--- Avaliação Final no Conjunto de Teste ---")
    model.load_state_dict(torch.load(args.model_save_path))
    model.eval()
    
    test_preds, test_true = [], []
    with torch.no_grad():
        for inputs, targets, _ in dataloaders['test']:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(targets.cpu().numpy())
    
    logger.info("\nRelatório de Classificação:\n" + classification_report(test_true, test_preds, target_names=args.classes, zero_division=0))
    
    cm = confusion_matrix(test_true, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=args.classes, yticklabels=args.classes)
    plt.title(f'Matriz de Confusão - {Path(args.data_dir).name}')
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Prevista')
    cm_path = f"{os.path.dirname(args.model_save_path)}/{Path(args.data_dir).name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Matriz de confusão salva em: {cm_path}")

    # --- CHAMADA PARA O GRAD-CAM ---
    logger.info("\n--- Gerando visualizações Grad-CAM ---")
    # **MODIFICAÇÃO: A camada alvo é a última ativação do último bloco convolucional do Xception**
    target_layer = model.backbone.blocks[-1].rep[-1].act3
    visualize_and_save_grad_cam(
        model=model,
        target_layer=target_layer,
        dataloader=dataloaders['test'],
        device=device,
        classes=args.classes,
        logger=logger,
        output_dir=args.grad_cam_dir,
        image_size=args.image_size,
        num_images=1
    )

# --- 6. BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Framework de Treinamento com Xception e XAI")
    parser.add_argument('--data_dir', type=str, default='./data_originals/QaTa-COV19', help='Diretório raiz contendo as pastas das classes.')
    parser.add_argument('--classes', nargs='+', default=['Covid', 'Normal', 'Pneumonia_bacteriana', 'Pneumonia_viral'])
    # **MODIFICAÇÃO: Tamanho de imagem ideal para Xception**
    parser.add_argument('--image_size', type=int, default=299, help="Tamanho da imagem (Xception usa 299x299)")
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=25, help='Número fixo de épocas.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Taxa de aprendizado para o classificador.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=max(os.cpu_count() // 2, 2))
    parser.add_argument('--seed', type=int, default=42)
    
    base_args = parser.parse_args()
    
    if base_args.data_dir == 'caminho/para/seu/dataset' or not os.path.isdir(base_args.data_dir):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERRO: Por favor, especifique o caminho para o seu dataset.")
        print("!!! Use o argumento --data_dir /caminho/completo/para/o/dataset")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        set_seeds(base_args.seed)
        
        # **MODIFICAÇÃO: Diretório de resultados específico para Xception**
        results_base_dir = "training_results_xception"
        
        dataset_path = Path(base_args.data_dir)
        dataset_name = dataset_path.name
        
        print(f"\n{'='*25} INICIANDO TREINAMENTO PARA: {dataset_name.upper()} {'='*25}")
        
        run_args = copy.deepcopy(base_args)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(results_base_dir) / f"{dataset_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run_args.model_save_path = str(output_dir / f"{dataset_name}_best_model_xception.pth")
        run_args.grad_cam_dir = str(output_dir / "grad_cam_results")
        
        logger = setup_logging(str(output_dir), dataset_name)
        
        try:
            train_and_evaluate(run_args, logger)
        except Exception as e:
            logger.critical(f"Ocorreu um erro fatal: {e}", exc_info=True)

        print(f"{'='*25} TREINAMENTO FINALIZADO PARA: {dataset_name.upper()} {'='*25}\n")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()