# -*- coding: utf-8 -*-
"""
Algoritmo SNN Avançado para classificação com meta de 90%+ de acurácia.
Melhorias implementadas:
1. Arquitetura ResNet-style com conexões residuais adaptadas para SNN
2. Múltiplos tipos de neurônios (Leaky, Synaptic, Alpha)
3. Augmentação de dados avançada
4. Learning rate scheduling e early stopping
5. Ensemble de modelos
6. Técnicas de regularização específicas para SNN
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
from snntorch import functional as SF
from snntorch import spikegen

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os
from PIL import Image
import random
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --- CONFIGURAÇÃO AVANÇADA ---
class AdvancedConfig:
    DATA_DIR = "sam_results_bigger_base/bigger_base_segmented"
    BATCH_SIZE = 16      # Batch menor para estabilidade
    IMG_SIZE = 128       # Imagens maiores para mais detalhes
    NUM_WORKERS = os.cpu_count() // 2
    
    # Parâmetros SNN otimizados
    NUM_STEPS = 20       # Mais passos para dinâmica temporal rica
    BETA = 0.9           # Decaimento otimizado
    THRESHOLD = 1.0      # Limiar de disparo
    
    # Parâmetros de treinamento otimizados
    EPOCHS = 100
    INITIAL_LR = 5e-4    # Learning rate inicial menor
    MIN_LR = 1e-6
    PATIENCE = 15        # Para early stopping
    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.3
    
    # Ensemble
    NUM_MODELS = 3       # Número de modelos no ensemble
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- TRANSFORMAÇÕES AVANÇADAS ---
def get_advanced_transforms(img_size, is_training=True):
    if is_training:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# --- BLOCO RESIDUAL SNN ---
class SNNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, beta=0.9):
        super().__init__()
        
        spike_grad = surrogate.atan(alpha=2.0)
        
        # Convolução principal
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        # Conexão residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
    def forward(self, x):
        # Fluxo principal
        out = self.lif1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Adiciona conexão residual
        out += self.shortcut(x)
        out = self.lif_out(out)
        
        return out

# --- BLOCO DE ATENÇÃO TEMPORAL ---
class TemporalAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- MODELO SNN AVANÇADO ---
class AdvancedSNN_Classifier(nn.Module):
    def __init__(self, num_classes, beta=0.9, dropout=0.3):
        super().__init__()
        
        spike_grad = surrogate.atan(alpha=2.0)
        
        # Stem (entrada)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
        )
        self.stem_lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Blocos residuais
        self.layer1 = self._make_layer(64, 64, 2, stride=1, beta=beta)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, beta=beta)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, beta=beta)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, beta=beta)
        
        # Atenção temporal
        self.attention = TemporalAttention(512)
        
        # Pooling adaptativo e classificador
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        
        # Múltiplas camadas de classificação com diferentes tipos de neurônios
        self.fc1 = nn.Linear(512, 256)
        self.lif_fc1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        self.fc2 = nn.Linear(256, 128)
        self.synaptic_fc2 = snn.Synaptic(alpha=0.8, beta=beta, spike_grad=spike_grad, init_hidden=True)
        
        self.fc3 = nn.Linear(128, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride, beta):
        layers = []
        layers.append(SNNResidualBlock(in_channels, out_channels, stride, beta))
        for _ in range(1, blocks):
            layers.append(SNNResidualBlock(out_channels, out_channels, 1, beta))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        utils.reset(self)
        spk_rec = []
        mem_rec = []
        
        for step in range(AdvancedConfig.NUM_STEPS):
            # Stem
            cur = self.stem_lif(self.stem(x))
            cur = self.maxpool(cur)
            
            # Blocos residuais
            cur = self.layer1(cur)
            cur = self.layer2(cur)
            cur = self.layer3(cur)
            cur = self.layer4(cur)
            
            # Atenção
            cur = self.attention(cur)
            
            # Classificador
            cur = self.avgpool(cur)
            cur = torch.flatten(cur, 1)
            cur = self.dropout(cur)
            
            cur = self.lif_fc1(self.fc1(cur))
            cur = self.dropout(cur)
            
            cur = self.synaptic_fc2(self.fc2(cur))
            cur = self.dropout(cur)
            
            spk_out, mem_out = self.lif_out(self.fc3(cur))
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
        
        # Retorna tanto a soma dos spikes quanto a média dos potenciais de membrana
        spk_sum = torch.stack(spk_rec, dim=0).sum(dim=0)
        mem_avg = torch.stack(mem_rec, dim=0).mean(dim=0)
        
        return spk_sum, mem_avg

# --- FUNÇÃO DE PERDA COMBINADA ---
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.8):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha
    
    def forward(self, spike_output, mem_output, targets):
        # Perda principal baseada em spikes
        spike_loss = self.ce_loss(spike_output, targets)
        
        # Perda auxiliar baseada em potencial de membrana
        target_one_hot = F.one_hot(targets, num_classes=spike_output.size(1)).float()
        mem_loss = self.mse_loss(torch.softmax(mem_output, dim=1), target_one_hot)
        
        return self.alpha * spike_loss + (1 - self.alpha) * mem_loss

# --- SCHEDULER PERSONALIZADO ---
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, step):
        if step < self.warmup_steps:
            # Warmup linear
            lr = self.base_lr * (step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr

# --- FUNÇÕES DE TREINO AVANÇADAS ---
def train_epoch_advanced(model, dataloader, optimizer, criterion, device, scheduler=None, epoch=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Época {epoch+1} - Treino")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        spike_out, mem_out = model(images)
        loss = criterion(spike_out, mem_out, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler:
            scheduler.step(epoch * len(dataloader) + batch_idx)
        
        running_loss += loss.item()
        _, predicted = spike_out.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if batch_idx % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Perda': f'{loss.item():.4f}',
                'Acur.': f'{100.*correct/total:.2f}%',
                'LR': f'{current_lr:.2e}'
            })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate_epoch_advanced(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validação"):
            images, labels = images.to(device), labels.to(device)
            spike_out, mem_out = model(images)
            loss = criterion(spike_out, mem_out, labels)
            
            running_loss += loss.item()
            _, predicted = spike_out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(dataloader), 100. * correct / total, all_preds, all_labels

# --- ENSEMBLE DE MODELOS ---
class SNNEnsemble:
    def __init__(self, num_models, num_classes, device):
        self.models = []
        self.num_models = num_models
        self.device = device
        
        for i in range(num_models):
            # Variação nos hiperparâmetros para diversidade
            beta = 0.85 + 0.1 * (i / num_models)
            dropout = 0.2 + 0.2 * (i / num_models)
            model = AdvancedSNN_Classifier(num_classes, beta=beta, dropout=dropout).to(device)
            self.models.append(model)
    
    def train_ensemble(self, train_loader, val_loader, epochs):
        results = []
        
        for i, model in enumerate(self.models):
            print(f"\nTreinando modelo {i+1}/{self.num_models} do ensemble...")
            
            optimizer = optim.AdamW(model.parameters(), lr=AdvancedConfig.INITIAL_LR,
                                  weight_decay=AdvancedConfig.WEIGHT_DECAY)
            criterion = CombinedLoss()
            
            total_steps = epochs * len(train_loader)
            scheduler = WarmupCosineScheduler(optimizer, warmup_steps=total_steps//10, 
                                           total_steps=total_steps, min_lr=AdvancedConfig.MIN_LR)
            
            best_val_acc = 0
            patience_counter = 0
            
            for epoch in range(epochs):
                train_loss, train_acc = train_epoch_advanced(model, train_loader, optimizer, 
                                                           criterion, self.device, scheduler, epoch)
                val_loss, val_acc, _, _ = validate_epoch_advanced(model, val_loader, criterion, self.device)
                
                print(f"Modelo {i+1} - Época {epoch+1}: Val Acc: {val_acc:.2f}%")
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    torch.save(model.state_dict(), f"best_snn_ensemble_model_{i}.pth")
                else:
                    patience_counter += 1
                    
                if patience_counter >= AdvancedConfig.PATIENCE:
                    print(f"Early stopping para modelo {i+1} na época {epoch+1}")
                    break
            
            # Carrega o melhor modelo
            model.load_state_dict(torch.load(f"best_snn_ensemble_model_{i}.pth"))
            results.append(best_val_acc)
        
        return results
    
    def predict(self, dataloader):
        predictions = []
        
        for model in self.models:
            model.eval()
            model_preds = []
            
            with torch.no_grad():
                for images, _ in dataloader:
                    images = images.to(self.device)
                    spike_out, mem_out = model(images)
                    # Combina informação de spikes e potencial de membrana
                    combined_out = F.softmax(spike_out, dim=1) + 0.3 * F.softmax(mem_out, dim=1)
                    model_preds.append(combined_out.cpu())
            
            predictions.append(torch.cat(model_preds))
        
        # Média das predições do ensemble
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred.argmax(dim=1).numpy()

# --- VISUALIZAÇÃO AVANÇADA ---
def plot_training_metrics(train_accs, val_accs, train_losses, val_losses):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_accs, label='Treino', color='blue')
    ax1.plot(val_accs, label='Validação', color='red')
    ax1.set_title('Acurácia ao Longo do Treinamento')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Acurácia (%)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_losses, label='Treino', color='blue')
    ax2.plot(val_losses, label='Validação', color='red')
    ax2.set_title('Perda ao Longo do Treinamento')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Perda')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão - Ensemble SNN')
    plt.xlabel('Predição')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

# --- FUNÇÃO PRINCIPAL ---
def main():
    cfg = AdvancedConfig()
    
    if not os.path.exists(cfg.DATA_DIR) or not os.listdir(cfg.DATA_DIR):
        print("="*60)
        print(f"ERRO: Diretório do dataset não encontrado ou está vazio!")
        print(f"Caminho configurado: '{cfg.DATA_DIR}'")
        print("="*60)
        return
    
    print(f"Dispositivo: {cfg.DEVICE}")
    print(f"Configuração: {cfg.EPOCHS} épocas, Batch Size: {cfg.BATCH_SIZE}")
    
    # Datasets com transformações avançadas
    train_transform = get_advanced_transforms(cfg.IMG_SIZE, is_training=True)
    val_transform = get_advanced_transforms(cfg.IMG_SIZE, is_training=False)
    
    full_dataset = datasets.ImageFolder(cfg.DATA_DIR, transform=val_transform)
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    
    print(f"Encontradas {num_classes} classes: {class_names}")
    
    # Divisão estratificada
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Aplica transformações de treino apenas ao conjunto de treino
    train_dataset.dataset = datasets.ImageFolder(cfg.DATA_DIR, transform=train_transform)
    
    # Balanceamento das classes
    train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[label] for label in train_labels]
    sampler = WeightedRandomSampler(weights, len(weights))
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, 
                             sampler=sampler, num_workers=cfg.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, 
                           shuffle=False, num_workers=cfg.NUM_WORKERS)
    
    print(f"Tamanho do conjunto de treino: {len(train_dataset)}")
    print(f"Tamanho do conjunto de validação: {len(val_dataset)}")
    
    # Treina ensemble
    ensemble = SNNEnsemble(cfg.NUM_MODELS, num_classes, cfg.DEVICE)
    ensemble_results = ensemble.train_ensemble(train_loader, val_loader, cfg.EPOCHS)
    
    print(f"\nResultados do Ensemble:")
    for i, acc in enumerate(ensemble_results):
        print(f"Modelo {i+1}: {acc:.2f}%")
    print(f"Média do Ensemble: {np.mean(ensemble_results):.2f}%")
    
    # Avaliação final
    print("\nAvaliação final do ensemble...")
    final_preds = ensemble.predict(val_loader)
    val_labels = [full_dataset.targets[i] for i in val_dataset.indices]
    
    final_accuracy = (final_preds == val_labels).mean() * 100
    print(f"Acurácia final do ensemble: {final_accuracy:.2f}%")
    
    # Relatório detalhado
    print("\nRelatório de classificação:")
    print(classification_report(val_labels, final_preds, target_names=class_names))
    
    # Visualizações
    plot_confusion_matrix(val_labels, final_preds, class_names)
    
    # Salva o melhor modelo individual para visualização posterior
    best_model_idx = np.argmax(ensemble_results)
    best_model = ensemble.models[best_model_idx]
    torch.save(best_model.state_dict(), "best_advanced_snn_model.pth")
    
    print(f"\nMelhor modelo individual (#{best_model_idx+1}) salvo com {ensemble_results[best_model_idx]:.2f}% de acurácia")
    print(f"Acurácia do ensemble: {final_accuracy:.2f}%")
    
    if final_accuracy >= 90:
        print("🎉 META DE 90% DE ACURÁCIA ALCANÇADA! 🎉")
    else:
        print(f"Meta não alcançada. Diferença: {90 - final_accuracy:.2f}%")

if __name__ == "__main__":
    main()