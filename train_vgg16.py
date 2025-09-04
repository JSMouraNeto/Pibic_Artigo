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
# import timm # Removido, pois não usaremos mais o EfficientNet
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
    # Limpa handlers anteriores para evitar logs duplicados em execuções sequenciais
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
        # Carrega a imagem em escala de cinza e a converte para RGB (3 canais)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None: raise ValueError(f"Imagem corrompida ou não encontrada: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if self.transform: image = self.transform(image=image)["image"]
        # Retorna o caminho da imagem para uso no Grad-CAM
        return image, label, str(image_path)

def get_medical_transforms(image_size=224, is_training=True):
    """Retorna transformações de imagem (data augmentation)."""
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

    # Divisão estratificada para manter a proporção das classes
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=val_size / (1 - test_size),
        random_state=random_state, stratify=train_val_labels)
    
    logger.info(f"Dataset dividido: {len(train_paths)} treino, {len(val_paths)} validação, {len(test_paths)} teste.")
    return {'train': (train_paths, train_labels), 'val': (val_paths, val_labels), 'test': (test_paths, test_labels)}

# --- 3. ARQUITETURA DO MODELO (MODIFICADO PARA VGG16) ---
class VGG16Classifier(nn.Module):
    """Classificador baseado na arquitetura VGG16."""
    def __init__(self, num_classes, pretrained=True, dropout=0.5):
        super().__init__()
        # Carrega o modelo VGG16 pré-treinado na ImageNet
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # Congela os parâmetros das camadas de extração de características (features)
        # para aproveitar o conhecimento pré-treinado sem alterá-lo.
        for param in self.vgg.features.parameters():
            param.requires_grad = False
            
        # Substitui a camada de classificação final da VGG16 por uma nova,
        # adaptada ao número de classes do nosso problema.
        num_features = self.vgg.classifier[0].in_features # Geralmente 25088 (512*7*7)
        self.vgg.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        # Expõe as camadas de features para fácil acesso pelo Grad-CAM
        self.features = self.vgg.features

    def forward(self, x):
        # O forward passa pela VGG completa (features -> avgpool -> classifier)
        return self.vgg(x)

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
        """Registra os hooks para capturar ativações e gradientes."""
        self.hook_handles.append(self.target_layer.register_forward_hook(self._forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self._backward_hook))

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        """Gera o mapa de calor Grad-CAM."""
        self.model.eval() # Garante que o modelo está em modo de avaliação
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        # Normaliza o heatmap para o intervalo [0, 1]
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().numpy()

    def remove_hooks(self):
        """Remove os hooks para liberar memória."""
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []

def unnormalize_image(tensor):
    """Reverte a normalização de um tensor de imagem para visualização."""
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().permute(1, 2, 0).cpu().numpy()
    return np.clip((tensor * std + mean) * 255, 0, 255).astype(np.uint8)

def visualize_and_save_grad_cam(model, target_layer, dataloader, device, classes, logger, output_dir, image_size, num_images=1):
    """Gera e salva imagens com o heatmap do Grad-CAM sobreposto."""
    os.makedirs(output_dir, exist_ok=True)
    grad_cam = GradCAM(model, target_layer)
    class_counts = defaultdict(int)
    num_classes = len(classes)
    
    logger.info(f"Gerando Grad-CAM para {num_images} exemplo(s) de cada classe...")
    
    # Itera sobre o dataloader até encontrar o número desejado de imagens por classe
    for inputs, targets, paths in dataloader:
        if len(class_counts) == num_classes and all(c >= num_images for c in class_counts.values()):
            break
        
        for i in range(inputs.size(0)):
            true_label_idx = targets[i].item()
            true_label_name = classes[true_label_idx]
            
            if class_counts[true_label_name] < num_images:
                img_tensor = inputs[i].unsqueeze(0).to(device)
                heatmap = grad_cam(img_tensor, class_idx=true_label_idx)
                
                original_img_rgb = unnormalize_image(inputs[i])
                heatmap_resized = cv2.resize(heatmap, (image_size, image_size))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                
                # Sobrepõe o heatmap na imagem original
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
    
    # Define as transformações para cada conjunto
    transforms = {
        'train': get_medical_transforms(args.image_size, is_training=True),
        'val': get_medical_transforms(args.image_size, is_training=False),
        'test': get_medical_transforms(args.image_size, is_training=False)
    }
    datasets = {k: MedicalImageDataset(data_splits[k][0], data_splits[k][1], transform=transforms[k]) for k in ['train', 'val', 'test']}
    
    # Calcula pesos para lidar com classes desbalanceadas no treino
    class_weights = compute_class_weight('balanced', classes=np.unique(data_splits['train'][1]), y=data_splits['train'][1])
    sampler = WeightedRandomSampler(
        weights=[class_weights[lbl] for lbl in data_splits['train'][1]],
        num_samples=len(data_splits['train'][1]),
        replacement=True
    )

    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True),
        'val': DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True),
        'test': DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    }
    
    # **MODIFICAÇÃO: Instancia o modelo VGG16**
    model = VGG16Classifier(num_classes=len(args.classes), dropout=args.dropout).to(device)
    
    # Otimizador e função de perda
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    best_val_f1 = 0.0

    logger.info("--- Iniciando Treinamento com VGG16 ---")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        # Loop de treinamento
        for inputs, targets, _ in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
        
        # Loop de validação
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

        # Salva o melhor modelo baseado no F1-Score da validação
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), args.model_save_path)
            logger.info(f"Modelo salvo! Novo melhor Val F1-Score: {best_val_f1:.4f}")
            
    logger.info("\n--- Avaliação Final no Conjunto de Teste ---")
    # Carrega o melhor modelo salvo para a avaliação final
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
    
    # Imprime o relatório de classificação
    logger.info("\nRelatório de Classificação:\n" + classification_report(test_true, test_preds, target_names=args.classes, zero_division=0))
    
    # Gera e salva a matriz de confusão
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
    visualize_and_save_grad_cam(
        model=model,
        # **MODIFICAÇÃO: Alvo para a última camada de features da VGG16**
        target_layer=model.features[-1],
        dataloader=dataloaders['test'],
        device=device,
        classes=args.classes,
        logger=logger,
        output_dir=args.grad_cam_dir,
        image_size=args.image_size,
        # **MODIFICAÇÃO: Gera apenas um exemplo por classe**
        num_images=1
    )

# --- 6. BLOCO DE EXECUÇÃO PRINCIPAL ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Framework de Treinamento com VGG16 e XAI")
    # Ajuste o 'default' para o caminho do seu dataset
    parser.add_argument('--data_dir', type=str, default='./data_originals/QaTa-COV19', help='Diretório raiz contendo as pastas das classes.')
    parser.add_argument('--classes', nargs='+', default=['Covid', 'Normal', 'Pneumonia_bacteriana', 'Pneumonia_viral'])
    parser.add_argument('--image_size', type=int, default=224, help="Tamanho da imagem (VGG16 usa 224x224)")
    parser.add_argument('--dropout', type=float, default=0.5)
    # **REQUISITO: 25 épocas**
    parser.add_argument('--epochs', type=int, default=25, help='Número fixo de épocas.')
    # **REQUISITO: Taxa de aprendizado fixa**
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Taxa de aprendizado fixa (ajustada para fine-tuning).')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=max(os.cpu_count() // 2, 2))
    parser.add_argument('--seed', type=int, default=42)
    
    base_args = parser.parse_args()
    
    # **IMPORTANTE**: Verifique se o diretório de dados existe
    if base_args.data_dir == 'caminho/para/seu/dataset' or not os.path.isdir(base_args.data_dir):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERRO: Por favor, especifique o caminho para o seu dataset.")
        print("!!! Use o argumento --data_dir /caminho/completo/para/o/dataset")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        set_seeds(base_args.seed)
        
        # Define o diretório base para salvar os resultados
        results_base_dir = "training_results_vgg16"
        
        # O código original iterava sobre sub-datasets, esta versão adaptada
        # executa para o dataset principal especificado em --data_dir
        dataset_path = Path(base_args.data_dir)
        dataset_name = dataset_path.name
        
        print(f"\n{'='*25} INICIANDO TREINAMENTO PARA: {dataset_name.upper()} {'='*25}")
        
        run_args = copy.deepcopy(base_args)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(results_base_dir) / f"{dataset_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define os caminhos de saída para esta execução
        run_args.model_save_path = str(output_dir / f"{dataset_name}_best_model_vgg16.pth")
        run_args.grad_cam_dir = str(output_dir / "grad_cam_results")
        
        logger = setup_logging(str(output_dir), dataset_name)
        
        try:
            train_and_evaluate(run_args, logger)
        except Exception as e:
            if 'logger' in locals():
                logger.critical(f"Ocorreu um erro fatal: {e}", exc_info=True)
            else:
                print(f"Ocorreu um erro fatal antes do logger ser configurado: {e}")

        print(f"{'='*25} TREINAMENTO FINALIZADO PARA: {dataset_name.upper()} {'='*25}\n")
        
        # Limpeza de memória da GPU
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()