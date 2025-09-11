import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
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

# Dataset customizado para leitura das máscaras e aplicação na imagem
class XrayMaskDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.samples = []  # (img, mask)
        self.labels = []
        
        # Verifica se o diretório existe
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Diretório {root_dir} não encontrado!")
        
        # Assume root_dir/<classe>/images/ e root_dir/<classe>/masks/
        try:
            self.classes = sorted([d for d in os.listdir(root_dir) 
                                 if os.path.isdir(os.path.join(root_dir, d))])
            if not self.classes:
                raise ValueError(f"Nenhuma pasta de classe encontrada em {root_dir}")
                
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            
            for class_name in self.classes:
                class_dir = os.path.join(root_dir, class_name)
                img_dir = os.path.join(class_dir, 'images')
                mask_dir = os.path.join(class_dir, 'masks')
                
                # Verifica se as pastas existem
                if not os.path.exists(img_dir):
                    logger.warning(f"Pasta de imagens não encontrada: {img_dir}")
                    continue
                if not os.path.exists(mask_dir):
                    logger.warning(f"Pasta de máscaras não encontrada: {mask_dir}")
                    continue
                
                image_names = sorted([f for f in os.listdir(img_dir) 
                                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                mask_names = sorted([f for f in os.listdir(mask_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                for img_name in image_names:
                    # Flexibilidade no nome da máscara
                    possible_mask_names = [
                        img_name.replace('.png', '_mask.png'),
                        img_name.replace('.jpg', '_mask.png'),
                        img_name.replace('.jpeg', '_mask.png'),
                        img_name  # caso a máscara tenha o mesmo nome
                    ]
                    
                    mask_found = None
                    for mask_name in possible_mask_names:
                        if mask_name in mask_names:
                            mask_found = mask_name
                            break
                    
                    if mask_found:
                        self.samples.append((os.path.join(img_dir, img_name), 
                                           os.path.join(mask_dir, mask_found)))
                        self.labels.append(self.class_to_idx[class_name])
                    else:
                        logger.warning(f"Máscara não encontrada para {img_name} na classe {class_name}")
        
        except Exception as e:
            logger.error(f"Erro ao carregar dataset: {e}")
            raise
        
        if not self.samples:
            raise ValueError("Nenhuma amostra válida encontrada no dataset!")
        
        self.transform = transform
        self.mask_transform = mask_transform
        logger.info(f"Dataset carregado: {len(self.samples)} amostras, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_path, mask_path = self.samples[idx]
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Redimensiona para o tamanho correto
            image = image.resize((224, 224))
            mask = mask.resize((224, 224))
            
            if self.transform:
                image = self.transform(image)
            if self.mask_transform:
                mask = self.mask_transform(mask)
            
            # CORREÇÃO: Aplica máscara corretamente nos 3 canais
            if mask.dim() == 2:  # Se a máscara tem 2 dimensões
                mask = mask.unsqueeze(0)  # Adiciona uma dimensão
            if mask.size(0) == 1 and image.size(0) == 3:  # Se máscara tem 1 canal e imagem 3
                mask = mask.expand_as(image)  # Expande para 3 canais
            
            image = image * mask
            label = self.labels[idx]
            return image, label
            
        except Exception as e:
            logger.error(f"Erro ao carregar amostra {idx}: {e}")
            # Retorna uma imagem preta como fallback
            image = torch.zeros(3, 224, 224)
            return image, 0

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0] if grad_output[0] is not None else grad_output[1]
    
    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        elif torch.is_tensor(class_idx):
            class_idx = class_idx.item()
        
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)
        
        if self.gradients is None or self.activations is None:
            logger.warning("Gradientes ou ativações não capturados")
            return torch.zeros((224, 224))
        
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # Calcula os pesos
        weights = torch.mean(gradients, dim=(2, 3))  # Média espacial
        
        # Gera o CAM
        cam = torch.zeros(activations.shape[2:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights[0]):  # weights[0] porque batch_size=1
            cam += w * activations[0, i]  # activations[0] porque batch_size=1
        
        cam = torch.relu(cam)
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
        self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

class VGG16Trainer:
    def __init__(self, data_dir, batch_size=32, img_size=224, k_folds=5):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.k_folds = k_folds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Usando device: {self.device}")

        # Transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())  # Threshold mais robusto
        ])
        
        try:
            self.full_dataset = XrayMaskDataset(
                data_dir, transform=self.train_transform, mask_transform=self.mask_transform)
            self.num_classes = len(self.full_dataset.classes)
            self.class_names = self.full_dataset.classes
            logger.info(f"Detectadas {self.num_classes} classes: {self.class_names}")
        except Exception as e:
            logger.error(f"Erro ao inicializar dataset: {e}")
            raise

        self.setup_model()
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    def setup_model(self):
        self.model = models.vgg16(weights='IMAGENET1K_V1')  # Atualizado para novo formato
        
        # Modifica o classificador
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
        
        # Congela features iniciais
        for param in self.model.features.parameters():
            param.requires_grad = False
        
        # Descongela últimas camadas
        for param in list(self.model.features.parameters())[-6:]:
            param.requires_grad = True
        
        self.model = self.model.to(self.device)

    def create_optimizer_scheduler(self, epochs):
        params = [
            {'params': [p for p in self.model.features.parameters() if p.requires_grad], 'lr': 1e-5},
            {'params': self.model.classifier.parameters(), 'lr': 1e-3}
        ]
        optimizer = optim.AdamW(params, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        return optimizer, scheduler

    def train_epoch(self, model, dataloader, optimizer, criterion, scaler=None):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            
            if scaler and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self, model, dataloader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation", leave=False):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.scaler and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
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

    def generate_heatmaps(self, model, save_dir):
        try:
            model.eval()
            os.makedirs(save_dir, exist_ok=True)
            
            target_layer = model.features[-1]
            grad_cam = GradCAM(model, target_layer)
            
            # Usa dataset de validação para heatmaps
            ds = XrayMaskDataset(self.data_dir, transform=self.val_transform, 
                               mask_transform=self.mask_transform)
            
            # Pega uma amostra de cada classe
            samples_per_class = {}
            for idx, (_, label) in enumerate(ds):
                class_name = ds.classes[label]
                if class_name not in samples_per_class:
                    samples_per_class[class_name] = idx
                    if len(samples_per_class) == self.num_classes:
                        break
            
            plt.figure(figsize=(15, 5 * self.num_classes))
            
            for i, (class_name, idx) in enumerate(samples_per_class.items()):
                image, label = ds[idx]
                image_tensor = image.unsqueeze(0).to(self.device)
                
                cam = grad_cam.generate_cam(image_tensor, label)
                cam = cam.cpu().numpy()
                cam_resized = cv2.resize(cam, (self.img_size, self.img_size))
                
                # Desnormaliza a imagem para visualização
                img_np = image.permute(1, 2, 0).numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                
                # Plot original
                plt.subplot(self.num_classes, 3, i*3 + 1)
                plt.imshow(img_np)
                plt.title(f'Original - {class_name}')
                plt.axis('off')
                
                # Plot heatmap
                plt.subplot(self.num_classes, 3, i*3 + 2)
                plt.imshow(cam_resized, cmap='jet')
                plt.title(f'Heatmap - {class_name}')
                plt.axis('off')
                
                # Plot overlay
                plt.subplot(self.num_classes, 3, i*3 + 3)
                plt.imshow(img_np)
                plt.imshow(cam_resized, cmap='jet', alpha=0.4)
                plt.title(f'Overlay - {class_name}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'gradcam_heatmaps.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Mapas de calor salvos em {save_dir}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar heatmaps: {e}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path):
        try:
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
        except Exception as e:
            logger.error(f"Erro ao plotar matriz de confusão: {e}")

    def train_kfold(self, epochs=50, save_dir='results'):
        os.makedirs(save_dir, exist_ok=True)
        
        # Configura logging para arquivo
        log_file = os.path.join(save_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        targets = np.array(self.full_dataset.labels)
        kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        fold_results = []
        all_train_losses, all_val_losses = [], []
        all_train_accs, all_val_accs = [], []
        
        for fold, (train_ids, val_ids) in enumerate(kfold.split(np.arange(len(targets)), targets)):
            logger.info(f"\n{'='*50}\nFOLD {fold + 1}/{self.k_folds}\n{'='*50}")
            
            # Reinicializa o modelo para cada fold
            self.setup_model()
            
            # Cria samplers
            train_sampler = SubsetRandomSampler(train_ids)
            val_sampler = SubsetRandomSampler(val_ids)
            
            train_loader = DataLoader(
                self.full_dataset, batch_size=self.batch_size, 
                sampler=train_sampler, num_workers=2, pin_memory=True
            )
            val_loader = DataLoader(
                self.full_dataset, batch_size=self.batch_size, 
                sampler=val_sampler, num_workers=2, pin_memory=True
            )
            
            optimizer, scheduler = self.create_optimizer_scheduler(epochs)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            early_stopping = EarlyStopping(patience=10, min_delta=0.002)
            
            # Listas para armazenar histórico do fold
            fold_train_losses, fold_val_losses = [], []
            fold_train_accs, fold_val_accs = [], []
            best_val_acc = 0.0
            
            for epoch in range(epochs):
                logger.info(f"\nEpoch {epoch+1}/{epochs}")
                
                # Treina
                train_loss, train_acc = self.train_epoch(
                    self.model, train_loader, optimizer, criterion, self.scaler
                )
                
                # Valida
                val_loss, val_acc, val_preds, val_labels = self.validate_epoch(
                    self.model, val_loader, criterion
                )
                
                scheduler.step()
                
                # Armazena histórico
                fold_train_losses.append(train_loss)
                fold_val_losses.append(val_loss)
                fold_train_accs.append(train_acc)
                fold_val_accs.append(val_acc)
                
                # Logs
                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                logger.info(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
                
                # Salva melhor modelo
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
            
            # Armazena históricos
            all_train_losses.append(fold_train_losses)
            all_val_losses.append(fold_val_losses)
            all_train_accs.append(fold_train_accs)
            all_val_accs.append(fold_val_accs)
            
            logger.info(f"Fold {fold + 1} - Best Val Acc: {best_val_acc:.2f}%")
            
            # Matriz de confusão
            self.plot_confusion_matrix(
                val_labels, val_preds, 
                os.path.join(save_dir, f'confusion_matrix_fold_{fold+1}.png')
            )
            
            # Gera heatmaps apenas no primeiro fold
            if fold == 0:
                self.generate_heatmaps(self.model, save_dir)
        
        # Salva resultados finais
        self.save_results(fold_results, all_train_losses, all_val_losses, 
                         all_train_accs, all_val_accs, save_dir)
        
        return fold_results

    def save_results(self, fold_results, train_losses, val_losses, train_accs, val_accs, save_dir):
        # Calcula estatísticas finais
        final_stats = {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'k_folds': self.k_folds,
            'mean_val_acc': np.mean([f['best_val_acc'] for f in fold_results]),
            'std_val_acc': np.std([f['best_val_acc'] for f in fold_results]),
            'fold_results': fold_results
        }
        
        # Salva em JSON
        with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # Log final
        logger.info(f"\n{'='*50}")
        logger.info("RESULTADOS FINAIS DO K-FOLD CROSS VALIDATION")
        logger.info(f"{'='*50}")
        logger.info(f"Acurácia Média: {final_stats['mean_val_acc']:.2f}% ± {final_stats['std_val_acc']:.2f}%")
        
        for i, fold_result in enumerate(fold_results):
            logger.info(f"Fold {i+1}: {fold_result['best_val_acc']:.2f}%")
        
        # Plota histórico
        self.plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir)
        logger.info(f"Todos os resultados salvos em: {save_dir}")

    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs, save_dir):
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss plots
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
            
            # Accuracy plots
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
            logger.info("Gráficos de histórico salvos")
            
        except Exception as e:
            logger.error(f"Erro ao plotar histórico: {e}")

# Exemplo de uso
if __name__ == "__main__":
    DATA_DIR = "data_originals/COVID-19_Radiography_Dataset"
    
    # Verificação mais robusta do diretório
    if not os.path.exists(DATA_DIR):
        print(f"Erro: Diretório {DATA_DIR} não encontrado!")
        print("Por favor, atualize a variável DATA_DIR com o caminho correto do seu dataset.")
        print("\nEstrutura esperada:")
        print("DATA_DIR/")
        print("├── classe1/")
        print("│   ├── images/")
        print("│   └── masks/")
        print("├── classe2/")
        print("│   ├── images/")
        print("│   └── masks/")
        print("└── ...")
        exit(1)
    
    try:
        # Configurações ajustadas
        trainer = VGG16Trainer(
            data_dir=DATA_DIR,
            batch_size=16,  # Reduzido para evitar problemas de memória
            img_size=224,
            k_folds=5
        )
        
        # Inicia treinamento
        results = trainer.train_kfold(epochs=30, save_dir='vgg16_results_covid_db')
        
        print("\n" + "="*50)
        print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*50)
        print("Arquivos gerados:")
        print("- best_model_fold_X.pth: Modelos salvos para cada fold")
        print("- training_log_YYYYMMDD_HHMMSS.log: Log detalhado do treinamento")
        print("- training_results.json: Resultados e estatísticas finais")
        print("- confusion_matrix_fold_X.png: Matrizes de confusão")
        print("- gradcam_heatmaps.png: Mapas de calor Grad-CAM")
        print("- training_history.png: Gráficos de perda e acurácia")
        
    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        print(f"\nErro: {e}")
        print("\nDicas para resolver:")
        print("1. Verifique se a estrutura de diretórios está correta.")
        print("2. Confirme se há imagens e máscaras correspondentes nas pastas.")
        print("3. Reduza o `batch_size` se estiver ocorrendo um erro de memória (Out of Memory).")
        print("4. Verifique se a GPU está disponível e se o PyTorch foi instalado com suporte a CUDA.")