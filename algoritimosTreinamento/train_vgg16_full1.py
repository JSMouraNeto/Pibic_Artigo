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
warnings.filterwarnings('ignore')

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks (usar register_forward_hook para ativação e register_full_backward_hook para gradientes)
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        score = output[0, class_idx.item() if torch.is_tensor(class_idx) else class_idx]
        score.backward(retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0].detach()
        activations = self.activations[0].detach()
        
        # Pool gradients across spatial dimensions
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = torch.relu(cam)
        
        # Normalize CAM
        if cam.max() != cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = torch.zeros_like(cam)
        
        return cam

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        # Deep copy
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

class CosineAnnealingWarmRestarts(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_i:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
                    for base_lr in self.base_lrs]
        else:
            self.T_cur = self.T_cur - self.T_i
            self.T_i = self.T_i * self.T_mult
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + np.cos(np.pi * self.T_cur / self.T_i)) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.T_i = self.T_0
                else:
                    n = int(np.log((epoch / self.T_0 * (self.T_mult - 1) + 1)) / np.log(self.T_mult))
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** n
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class VGG16Trainer:
    def __init__(self, data_dir, batch_size=32, img_size=224, k_folds=5):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.k_folds = k_folds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Verificar VRAM disponível
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {torch.cuda.get_device_name(0)} - VRAM: {gpu_memory:.1f}GB")

        # Configurar transforms otimizados
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.2))
        ])

        self.val_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Carregar dataset e detectar classes automaticamente
        self.full_dataset = datasets.ImageFolder(data_dir, transform=self.train_transform)
        self.num_classes = len(self.full_dataset.classes)
        self.class_names = self.full_dataset.classes

        logger.info(f"Detectadas {self.num_classes} classes: {self.class_names}")

        # Setup do modelo
        self.setup_model()

        # Histórico de treinamento
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'fold_results': []
        }

    def setup_model(self):
        """Configura o modelo VGG16 otimizado"""
        # Carregar VGG16 pré-treinado
        self.model = models.vgg16(pretrained=True)

        # Modificar classificador com dropout e batch normalization
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.BatchNorm1d(4096),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, self.num_classes)
        )

        # Congelar features layers (transfer learning)
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Descongelar últimas camadas convolucionais para fine-tuning
        for param in list(self.model.features.parameters())[-6:]:
            param.requires_grad = True

        self.model = self.model.to(self.device)

        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

    def create_optimizer_scheduler(self):
        """Cria otimizador e scheduler otimizados"""
        # Diferentes learning rates para diferentes partes do modelo
        params = [
            {'params': self.model.features.parameters(), 'lr': 1e-5},  # Features congeladas
            {'params': self.model.classifier.parameters(), 'lr': 1e-3}  # Classificador
        ]

        optimizer = optim.AdamW(params, weight_decay=1e-4)

        # Scheduler com warm restarts
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

        return optimizer, scheduler

    def train_epoch(self, model, dataloader, optimizer, criterion, scaler):
        """Treina uma época com mixed precision"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Estatísticas
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate_epoch(self, model, dataloader, criterion):
        """Valida uma época"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc, all_preds, all_labels

    def generate_heatmaps(self, model, dataset, save_dir):
        """Gera mapas de calor Grad-CAM para uma amostra de cada classe"""
        model.eval()
        os.makedirs(save_dir, exist_ok=True)

        # Configurar Grad-CAM na última camada convolucional
        target_layer = model.features[-1]
        grad_cam = GradCAM(model, target_layer)

        # Dataset para heatmaps sem augmentação
        heatmap_dataset = datasets.ImageFolder(self.data_dir, transform=self.val_transform)

        # Encontrar uma amostra de cada classe
        samples_per_class = {}
        for idx, (_, label) in enumerate(heatmap_dataset):
            class_name = heatmap_dataset.classes[label]
            if class_name not in samples_per_class:
                samples_per_class[class_name] = idx
                if len(samples_per_class) == self.num_classes:
                    break

        plt.figure(figsize=(15, 5 * self.num_classes))

        for i, (class_name, idx) in enumerate(samples_per_class.items()):
            image, label = heatmap_dataset[idx]
            image_tensor = image.unsqueeze(0).to(self.device)

            # Gerar CAM
            cam = grad_cam.generate_cam(image_tensor, label)
            cam = cam.cpu().numpy()

            # Redimensionar CAM para o tamanho da imagem original
            cam_resized = cv2.resize(cam, (self.img_size, self.img_size))

            # Converter imagem para visualização
            img_np = image.permute(1, 2, 0).numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_np = np.clip(img_np, 0, 1)

            # Plotar
            plt.subplot(self.num_classes, 3, i*3 + 1)
            plt.imshow(img_np)
            plt.title(f'Original - {class_name}')
            plt.axis('off')

            plt.subplot(self.num_classes, 3, i*3 + 2)
            plt.imshow(cam_resized, cmap='jet')
            plt.title(f'Heatmap - {class_name}')
            plt.axis('off')

            plt.subplot(self.num_classes, 3, i*3 + 3)
            plt.imshow(img_np)
            plt.imshow(cam_resized, cmap='jet', alpha=0.4)
            plt.title(f'Overlay - {class_name}')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'gradcam_heatmaps.png'), dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Mapas de calor salvos em {save_dir}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        """Plota e salva matriz de confusão"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Matriz de confusão salva em {save_path}")

    def train_kfold(self, epochs=50, save_dir='results'):
        """Treinamento com K-Fold Cross Validation"""
        os.makedirs(save_dir, exist_ok=True)

        # Configurar logging para arquivo
        log_file = os.path.join(save_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Preparar dados para K-Fold
        targets = [self.full_dataset[i][1] for i in range(len(self.full_dataset))]
        kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        fold_results = []
        all_train_losses = []
        all_val_losses = []
        all_train_accs = []
        all_val_accs = []

        for fold, (train_ids, val_ids) in enumerate(kfold.split(np.arange(len(targets)), targets)):
            logger.info(f"\n{'='*50}")
            logger.info(f"FOLD {fold + 1}/{self.k_folds}")
            logger.info(f"{'='*50}")

            # Reinicializar modelo para cada fold
            self.setup_model()

            # Criar dataloaders para o fold atual
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)

            # Dataset de treino com augmentação
            train_dataset = datasets.ImageFolder(self.data_dir, transform=self.train_transform)
            val_dataset = datasets.ImageFolder(self.data_dir, transform=self.val_transform)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                     sampler=train_sampler, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                   sampler=val_sampler, num_workers=4, pin_memory=True)

            # Configurar otimização
            optimizer, scheduler = self.create_optimizer_scheduler()
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            early_stopping = EarlyStopping(patience=15, min_delta=0.001)

            # Histórico do fold atual
            fold_train_losses = []
            fold_val_losses = []
            fold_train_accs = []
            fold_val_accs = []

            best_val_acc = 0.0

            # Loop de treinamento
            for epoch in range(epochs):
                logger.info(f"\nEpoch {epoch+1}/{epochs}")

                # Treinar
                train_loss, train_acc = self.train_epoch(
                    self.model, train_loader, optimizer, criterion, self.scaler
                )

                # Validar
                val_loss, val_acc, val_preds, val_labels = self.validate_epoch(
                    self.model, val_loader, criterion
                )

                # Atualizar scheduler
                scheduler.step()

                # Salvar métricas
                fold_train_losses.append(train_loss)
                fold_val_losses.append(val_loss)
                fold_train_accs.append(train_acc)
                fold_val_accs.append(val_acc)

                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                logger.info(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

                # Salvar melhor modelo
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save({
                        'fold': fold + 1,
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'val_loss': val_loss,
                        'class_names': self.class_names
                    }, os.path.join(save_dir, f'best_model_fold_{fold+1}.pth'))

                # Early stopping
                if early_stopping(val_loss, self.model):
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # Resultados do fold
            fold_result = {
                'fold': fold + 1,
                'best_val_acc': best_val_acc,
                'final_train_loss': fold_train_losses[-1],
                'final_val_loss': fold_val_losses[-1],
                'final_train_acc': fold_train_accs[-1],
                'final_val_acc': fold_val_accs[-1]
            }

            fold_results.append(fold_result)
            all_train_losses.append(fold_train_losses)
            all_val_losses.append(fold_val_losses)
            all_train_accs.append(fold_train_accs)
            all_val_accs.append(fold_val_accs)

            logger.info(f"Fold {fold + 1} - Best Val Acc: {best_val_acc:.2f}%")

            # Matriz de confusão para o fold atual
            self.plot_confusion_matrix(
                val_labels, val_preds,
                os.path.join(save_dir, f'confusion_matrix_fold_{fold+1}.png')
            )

            # Gerar heatmaps apenas para o primeiro fold
            if fold == 0:
                self.generate_heatmaps(self.model, val_dataset, save_dir)

        # Salvar resultados finais
        self.save_results(fold_results, all_train_losses, all_val_losses,
                          all_train_accs, all_val_accs, save_dir)

        return fold_results

    def save_results(self, fold_results, train_losses, val_losses,
                    train_accs, val_accs, save_dir):
        """Salva todos os resultados e gráficos"""

        # Calcular estatísticas finais
        final_stats = {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'k_folds': self.k_folds,
            'mean_val_acc': np.mean([f['best_val_acc'] for f in fold_results]),
            'std_val_acc': np.std([f['best_val_acc'] for f in fold_results]),
            'fold_results': fold_results
        }

        # Salvar estatísticas em JSON
        with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
            json.dump(final_stats, f, indent=2)

        logger.info(f"\n{'='*50}")
        logger.info("RESULTADOS FINAIS DO K-FOLD CROSS VALIDATION")
        logger.info(f"{'='*50}")
        logger.info(f"Acurácia Média: {final_stats['mean_val_acc']:.2f}% ± {final_stats['std_val_acc']:.2f}%")

        for i, fold_result in enumerate(fold_results):
            logger.info(f"Fold {i+1}: {fold_result['best_val_acc']:.2f}%")

        # Plotar gráficos de treinamento
        self.plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir)

        logger.info(f"\nTodos os resultados salvos em: {save_dir}")

    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs, save_dir):
        """Plota histórico de treinamento"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss por fold
        for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
            ax1.plot(tl, alpha=0.7, label=f'Train Fold {i+1}')
            ax2.plot(vl, alpha=0.7, label=f'Val Fold {i+1}')

        ax1.set_title('Training Loss por Fold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.set_title('Validation Loss por Fold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        # Accuracy por fold
        for i, (ta, va) in enumerate(zip(train_accs, val_accs)):
            ax3.plot(ta, alpha=0.7, label=f'Train Fold {i+1}')
            ax4.plot(va, alpha=0.7, label=f'Val Fold {i+1}')

        ax3.set_title('Training Accuracy por Fold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.legend()
        ax3.grid(True)

        ax4.set_title('Validation Accuracy por Fold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Exemplo de uso
if __name__ == "__main__":
    # Configurar caminho do dataset
    DATA_DIR = "data/bigger_base"  # Substituir pelo caminho real

    # Verificar se o diretório existe
    if not os.path.exists(DATA_DIR):
        print(f"Erro: Diretório {DATA_DIR} não encontrado!")
        print("Por favor, atualize a variável DATA_DIR com o caminho correto do seu dataset.")
        exit(1)

    # Inicializar trainer
    trainer = VGG16Trainer(
        data_dir=DATA_DIR,
        batch_size=16,  # Otimizado para RTX 3070Ti 8GB
        img_size=224,
        k_folds=5
    )

    # Executar treinamento com K-Fold
    results = trainer.train_kfold(epochs=50, save_dir='vgg16_results_big_base')

    print("\nTreinamento concluído!")
    print("Arquivos gerados:")
    print("- best_model_fold_X.pth: Modelos salvos para cada fold")
    print("- training_log_YYYYMMDD_HHMMSS.log: Log detalhado do treinamento")
    print("- training_results.json: Resultados e estatísticas finais")
    print("- confusion_matrix_fold_X.png: Matrizes de confusão")
    print("- gradcam_heatmaps.png: Mapas de calor Grad-CAM")
    print("- training_history.png: Gráficos de perda e acurácia")
