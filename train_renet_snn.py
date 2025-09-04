# -*- coding: utf-8 -*-
"""
Script completo para treinamento de uma Rede Neural de Pulso (SNN)
baseada na arquitetura ResNet-50, utilizando snnTorch e PyTorch.

Inclui validação cruzada K-Fold, early stopping, learning rate scheduler,
e geração de mapas de calor Grad-CAM adaptados para SNNs.
"""

# Certifique-se de que as bibliotecas necessárias estão instaladas
# !pip install torch torchvision snntorch scikit-learn seaborn matplotlib opencv-python tqdm

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import cv2
from PIL import Image
import logging
from datetime import datetime
import json
from tqdm import tqdm
import warnings

# Importações específicas para SNN com snnTorch
import snntorch as snn
from snntorch import surrogate
from snntorch import utils

warnings.filterwarnings('ignore')

# Configuração de logging para acompanhar o treinamento
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GradCAM_SNN:
    """
    Classe Grad-CAM adaptada para Redes Neurais de Pulso.
    Lida com a dimensão temporal para gerar os mapas de calor.
    """
    def __init__(self, model, target_layer, num_steps):
        self.model = model
        self.target_layer = target_layer
        self.num_steps = num_steps
        self.gradients = None
        self.activations = None
        
        # Hooks para capturar ativações e gradientes
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        utils.reset(self.model)
        
        final_spk_sum = 0
        for _ in range(self.num_steps):
            spk_out = self.model(input_tensor)
            final_spk_sum += spk_out
        
        if class_idx is None:
            class_idx = final_spk_sum.argmax(dim=1).item()
        elif torch.is_tensor(class_idx):
            class_idx = class_idx.item()

        self.model.zero_grad()
        score = final_spk_sum[0, class_idx]
        score.backward(retain_graph=True)
        
        gradients = self.gradients
        activations = self.activations
        
        weights = torch.mean(gradients, dim=(2, 3))
        
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
        
        cam = torch.relu(cam)
        
        if cam.max() > 0:
            cam_np = cam.cpu().numpy()
            cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np))
            return cam_np
        else:
            return torch.zeros(activations.shape[2:]).cpu().numpy()

class EarlyStopping:
    """Interrompe o treinamento quando a perda de validação para de melhorar."""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            logger.info("Early stopping acionado.")
            if self.restore_best_weights and self.best_weights is not None:
                logger.info("Restaurando os melhores pesos do modelo.")
                device = next(model.parameters()).device
                model.load_state_dict({k: v.to(device) for k, v in self.best_weights.items()})
            return True
        return False

class SNN_ResNet50_Trainer:
    """
    Classe principal para orquestrar o treinamento e validação
    de um modelo SNN-ResNet50.
    """
    def __init__(self, data_dir, batch_size=32, img_size=224, k_folds=5, num_steps=25):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.k_folds = k_folds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_steps = num_steps

        if torch.cuda.is_available():
            gpu_props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU: {gpu_props.name} - VRAM Total: {gpu_props.total_memory / 1024**3:.1f}GB")
        else:
            logger.info("Utilizando CPU. O treinamento será significativamente mais lento.")

        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.full_dataset = datasets.ImageFolder(data_dir)
        self.num_classes = len(self.full_dataset.classes)
        self.class_names = self.full_dataset.classes
        logger.info(f"Detectadas {self.num_classes} classes: {self.class_names}")

    def setup_model(self):
        """Configura a ResNet-50, substituindo ReLU por neurônios SNN."""
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        spike_grad = surrogate.fast_sigmoid()

        def replace_relu_with_snnleaky(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(module, name, snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True))
                else:
                    replace_relu_with_snnleaky(child)
        
        replace_relu_with_snnleaky(model)

        for name, param in model.named_parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        self.model = model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == 'cuda')

    def create_optimizer_scheduler(self):
        """Cria otimizador e scheduler com learning rates diferenciados."""
        params_to_update = [
            {'params': [p for n, p in self.model.named_parameters() if "layer4" in n and p.requires_grad], 'lr': 1e-5},
            {'params': self.model.fc.parameters(), 'lr': 1e-3}
        ]
        
        optimizer = optim.AdamW(params_to_update, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        return optimizer, scheduler

    def train_epoch(self, model, dataloader, optimizer, criterion):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Treinando")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            
            utils.reset(model)
            
            spk_rec_total = 0
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                for _ in range(self.num_steps):
                    spk_out = model(images)
                    spk_rec_total += spk_out
                
                loss = criterion(spk_rec_total / self.num_steps, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(spk_rec_total.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'Perda': f'{loss.item():.4f}', 'Acur.': f'{100.*correct/total:.2f}%'})

        return running_loss / len(dataloader), 100. * correct / total

    def validate_epoch(self, model, dataloader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validando"):
                images, labels = images.to(self.device), labels.to(self.device)
                utils.reset(model)
                
                spk_rec_total = 0
                with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                    for _ in range(self.num_steps):
                        spk_out = model(images)
                        spk_rec_total += spk_out
                
                loss = criterion(spk_rec_total / self.num_steps, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(spk_rec_total.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return running_loss / len(dataloader), 100. * correct / total, all_preds, all_labels
    
    def generate_heatmaps(self, model, save_dir):
        """Gera e salva mapas de calor Grad-CAM para uma amostra de cada classe."""
        logger.info("Gerando mapas de calor Grad-CAM...")
        model.eval()
        os.makedirs(save_dir, exist_ok=True)
        
        target_layer = model.layer4[-1]
        grad_cam = GradCAM_SNN(model, target_layer, self.num_steps)
        
        heatmap_dataset = datasets.ImageFolder(self.data_dir, transform=self.val_transform)
        
        samples_per_class = {}
        for idx, (_, label) in enumerate(heatmap_dataset):
            if label not in samples_per_class:
                samples_per_class[label] = idx
            if len(samples_per_class) == self.num_classes:
                break
        
        fig, axes = plt.subplots(self.num_classes, 3, figsize=(15, 5 * self.num_classes))
        fig.suptitle('Grad-CAM Heatmaps para SNN-ResNet50', fontsize=20)

        for i, (label, idx) in enumerate(samples_per_class.items()):
            class_name = self.class_names[label]
            image, _ = heatmap_dataset[idx]
            image_tensor = image.unsqueeze(0).to(self.device)

            cam = grad_cam.generate_cam(image_tensor, label)
            cam_resized = cv2.resize(cam, (self.img_size, self.img_size))

            img_np = image.permute(1, 2, 0).numpy()
            img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)

            ax_row = axes if self.num_classes == 1 else axes[i]
            ax_row[0].imshow(img_np)
            ax_row[0].set_title(f'Original - {class_name}')
            ax_row[0].axis('off')

            ax_row[1].imshow(cam_resized, cmap='jet')
            ax_row[1].set_title(f'Heatmap - {class_name}')
            ax_row[1].axis('off')

            ax_row[2].imshow(img_np)
            ax_row[2].imshow(cam_resized, cmap='jet', alpha=0.4)
            ax_row[2].set_title(f'Sobreposição - {class_name}')
            ax_row[2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        heatmap_path = os.path.join(save_dir, 'gradcam_heatmaps.png')
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        logger.info(f"Mapas de calor salvos em {heatmap_path}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Matriz de Confusão')
        plt.ylabel('Rótulo Verdadeiro')
        plt.xlabel('Rótulo Previsto')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    def train_kfold(self, epochs=50, save_dir='results'):
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

        targets = [s[1] for s in self.full_dataset.samples]
        kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        fold_results, all_train_losses, all_val_losses, all_train_accs, all_val_accs = [], [], [], [], []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(np.arange(len(self.full_dataset)), targets)):
            logger.info(f"\n{'='*50}\nINICIANDO FOLD {fold + 1}/{self.k_folds}\n{'='*50}")
            self.setup_model()

            train_subset = torch.utils.data.Subset(self.full_dataset, train_ids)
            val_subset = torch.utils.data.Subset(self.full_dataset, val_ids)
            
            train_subset.dataset.transform = self.train_transform
            val_subset.dataset.transform = self.val_transform
            
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_subset, batch_size=self.batch_size, num_workers=4, pin_memory=True)

            optimizer, scheduler = self.create_optimizer_scheduler()
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            early_stopping = EarlyStopping(patience=15, min_delta=0.001)

            fold_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            best_val_acc = 0.0

            for epoch in range(epochs):
                logger.info(f"\n--- Epoch {epoch+1}/{epochs} ---")
                train_loss, train_acc = self.train_epoch(self.model, train_loader, optimizer, criterion)
                val_loss, val_acc, val_preds, val_labels = self.validate_epoch(self.model, val_loader, criterion)
                scheduler.step()

                fold_history['train_loss'].append(train_loss); fold_history['val_loss'].append(val_loss)
                fold_history['train_acc'].append(train_acc); fold_history['val_acc'].append(val_acc)

                logger.info(f"Perda Treino: {train_loss:.4f}, Acur. Treino: {train_acc:.2f}%")
                logger.info(f"Perda Val.: {val_loss:.4f}, Acur. Val.: {val_acc:.2f}%")
                logger.info(f"LR atual: {optimizer.param_groups[0]['lr']:.2e}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    model_path = os.path.join(save_dir, f'best_model_fold_{fold+1}.pth')
                    torch.save(self.model.state_dict(), model_path)
                    logger.info(f"Novo melhor modelo salvo em {model_path} com acurácia de {best_val_acc:.2f}%")

                if early_stopping(val_loss, self.model):
                    break

            fold_results.append({'fold': fold + 1, 'best_val_acc': best_val_acc})
            all_train_losses.append(fold_history['train_loss']); all_val_losses.append(fold_history['val_loss'])
            all_train_accs.append(fold_history['train_acc']); all_val_accs.append(fold_history['val_acc'])
            
            self.plot_confusion_matrix(val_labels, val_preds, os.path.join(save_dir, f'confusion_matrix_fold_{fold+1}.png'))

            if fold == 0:
                best_model_path = os.path.join(save_dir, f'best_model_fold_{fold+1}.pth')
                if os.path.exists(best_model_path):
                    self.model.load_state_dict(torch.load(best_model_path))
                    self.generate_heatmaps(self.model, save_dir)
        
        self.save_final_results(fold_results, all_train_losses, all_val_losses, all_train_accs, all_val_accs, save_dir)

    def save_final_results(self, fold_results, train_losses, val_losses, train_accs, val_accs, save_dir):
        """Salva os resultados consolidados e os gráficos de treinamento."""
        mean_acc = np.mean([f['best_val_acc'] for f in fold_results])
        std_acc = np.std([f['best_val_acc'] for f in fold_results])

        final_stats = {
            'mean_val_accuracy': mean_acc,
            'std_dev_val_accuracy': std_acc,
            'fold_results': fold_results
        }
        
        with open(os.path.join(save_dir, 'training_summary.json'), 'w') as f:
            json.dump(final_stats, f, indent=4)

        logger.info(f"\n{'='*50}\nRESULTADO FINAL K-FOLD\n{'='*50}")
        logger.info(f"Acurácia Média de Validação: {mean_acc:.2f}% ± {std_acc:.2f}%")

        self.plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir)
        logger.info(f"Resultados, logs e gráficos salvos no diretório: {save_dir}")

    def plot_training_history(self, all_train_losses, all_val_losses, all_train_accs, all_val_accs, save_dir):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Histórico de Treinamento por Fold', fontsize=20)
        
        for i in range(len(all_train_losses)):
            ax1.plot(all_train_losses[i], alpha=0.7, label=f'Fold {i+1}')
            ax2.plot(all_val_losses[i], alpha=0.7, label=f'Fold {i+1}')
            ax3.plot(all_train_accs[i], alpha=0.7, label=f'Fold {i+1}')
            ax4.plot(all_val_accs[i], alpha=0.7, label=f'Fold {i+1}')

        ax1.set_title('Perda de Treinamento'); ax1.set_xlabel('Época'); ax1.set_ylabel('Perda'); ax1.legend(); ax1.grid(True)
        ax2.set_title('Perda de Validação'); ax2.set_xlabel('Época'); ax2.set_ylabel('Perda'); ax2.legend(); ax2.grid(True)
        ax3.set_title('Acurácia de Treinamento'); ax3.set_xlabel('Época'); ax3.set_ylabel('Acurácia (%)'); ax3.legend(); ax3.grid(True)
        ax4.set_title('Acurácia de Validação'); ax4.set_xlabel('Época'); ax4.set_ylabel('Acurácia (%)'); ax4.legend(); ax4.grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    # --- CONFIGURAÇÕES DO TREINAMENTO ---
    # !!! ATENÇÃO: ATUALIZE ESTE CAMINHO PARA O SEU DATASET !!!
    DATA_DIR = "data_originals/chest_xray" 

    # Parâmetros de Treinamento
    BATCH_SIZE = 16      # Reduza se tiver erros de memória (OutOfMemoryError)
    IMG_SIZE = 224       # Tamanho da imagem para ResNet
    K_FOLDS = 5          # Número de folds para validação cruzada
    EPOCHS = 25          # Número máximo de épocas por fold
    NUM_STEPS = 20       # Número de passos de tempo para a simulação SNN (15-25 é um bom começo)
    SAVE_DIR = 'resultados_SNN_ResNet50' # Pasta para salvar os resultados

    # --- INÍCIO DA EXECUÇÃO ---
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        print("="*60)
        print(f"ERRO: Diretório do dataset não encontrado ou está vazio!")
        print(f"Caminho configurado: '{DATA_DIR}'")
        print("Por favor, atualize a variável DATA_DIR no final do script.")
        print("="*60)
        exit(1)

    trainer = SNN_ResNet50_Trainer(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        k_folds=K_FOLDS,
        num_steps=NUM_STEPS
    )

    trainer.train_kfold(epochs=EPOCHS, save_dir=SAVE_DIR)

    print("\nTreinamento concluído!")