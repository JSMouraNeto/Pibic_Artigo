import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate, utils
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random

# --- CONFIGURAÇÃO OTIMIZADA PARA CLASSIFICAÇÃO DE PNEUMOPATIA ---
class Hyperparameters:
    # 1. Configuração do Ambiente
    # !!! IMPORTANTE: Altere este caminho para o diretório do seu dataset de raio-x de tórax !!!
    # Ex: ./data/chest_xray/train, com subpastas como 'NORMAL' e 'PNEUMONIA'
    DATA_DIR = "/home/mob4ai-001/Pibic_Artigo/dataset_processado/bigger_base" 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = min(os.cpu_count() // 2, 8)
    
    # 2. Hiperparâmetros do Modelo e SNN
    # Tamanho de imagem maior para capturar detalhes finos em raios-x
    IMG_SIZE = 224
    NUM_STEPS = 20
    BETA = 0.9
    THRESHOLD = 1.0

    # 3. Hiperparâmetros de Treinamento
    EPOCHS = 50 # Reduzido para evitar overfitting em datasets médicos, com Early Stopping
    # Batch size ajustado para imagens maiores
    BATCH_SIZE = 32 
    INITIAL_LR = 3e-4
    MIN_LR = 1e-6

    # 4. Técnicas de Regularização
    WEIGHT_DECAY = 5e-4
    DROPOUT = 0.5
    CUTMIX_MIXUP_PROB = 0.8
    CUTMIX_ALPHA = 1.0
    MIXUP_ALPHA = 1.0
    LABEL_SMOOTHING = 0.1
    # Paciência para Early Stopping
    PATIENCE = 15

# --- TRANSFORMAÇÕES DE DADOS PARA IMAGENS MÉDICAS (RAIO-X) ---
def get_transforms(img_size):
    # Transformações de treino: conservadoras para preservar estruturas patológicas
    train_transform = transforms.Compose([
        # Converte para tons de cinza, pois raios-x não possuem cor
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        # Augmentations mais controladas: rotação, translação e zoom leves
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # Leve ajuste de brilho e contraste
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        # Normalização para imagens de 1 canal (tons de cinza)
        transforms.Normalize([0.5], [0.5]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
    ])
    
    # Transformações de validação: apenas redimensionamento e normalização
    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return train_transform, val_transform

# --- MODELO SNN (ResNet-like) ADAPTADO PARA IMAGENS DE 1 CANAL ---
class SNNResidualBlock(nn.Module):
    # (Nenhuma alteração necessária neste bloco)
    def __init__(self, in_channels, out_channels, stride=1, beta=0.9, spike_grad=surrogate.atan()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

    def forward(self, x):
        out = self.lif1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.lif_out(out)
        return out

class SNN_Pneumopathy_Classifier(nn.Module):
    def __init__(self, num_classes, beta=0.9, dropout=0.5):
        super().__init__()
        spike_grad = surrogate.atan(alpha=2.0)
        
        # *** ALTERAÇÃO CRÍTICA: Camada de entrada agora aceita 1 canal (tons de cinza) ***
        self.stem = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64))
        
        self.stem_lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2, stride=1, beta=beta, spike_grad=spike_grad)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, beta=beta, spike_grad=spike_grad)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, beta=beta, spike_grad=spike_grad)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, beta=beta, spike_grad=spike_grad)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.lif_fc1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc2 = nn.Linear(256, 128)
        self.synaptic_fc2 = snn.Synaptic(alpha=0.8, beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc3 = nn.Linear(128, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)

    def _make_layer(self, in_channels, out_channels, blocks, stride, beta, spike_grad):
        layers = []
        layers.append(SNNResidualBlock(in_channels, out_channels, stride, beta, spike_grad))
        for _ in range(1, blocks):
            layers.append(SNNResidualBlock(out_channels, out_channels, 1, beta, spike_grad))
        return nn.Sequential(*layers)

    def forward(self, x):
        utils.reset(self)
        spk_rec = []
        for step in range(Hyperparameters.NUM_STEPS):
            cur = self.stem_lif(self.stem(x))
            cur = self.maxpool(cur)
            cur = self.layer1(cur)
            cur = self.layer2(cur)
            cur = self.layer3(cur)
            cur = self.layer4(cur)
            cur = self.avgpool(cur)
            cur = torch.flatten(cur, 1)
            cur = self.dropout(cur)
            cur = self.lif_fc1(self.fc1(cur))
            cur = self.dropout(cur)
            cur = self.synaptic_fc2(self.fc2(cur))
            cur = self.dropout(cur)
            spk_out, _ = self.lif_out(self.fc3(cur))
            spk_rec.append(spk_out)
        return torch.stack(spk_rec, dim=0).sum(dim=0)

# --- FUNÇÕES DE MIXUP, CUTMIX, LOSS E LOOPS DE TREINO/VALIDAÇÃO ---
# (Nenhuma alteração necessária nestas funções, elas são robustas como estão)
def mixup_data(x, y, alpha=1.0, device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_w // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1.0, device='cuda'):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def loss_fn_robust(pred, y_a, y_b, lam, label_smoothing):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_one_epoch(model, dataloader, optimizer, scaler, cfg):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc="Treino")
    for images, labels in pbar:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        use_mix = random.random() < cfg.CUTMIX_MIXUP_PROB
        if use_mix:
            if random.random() < 0.5:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, cfg.CUTMIX_ALPHA, cfg.DEVICE)
            else:
                images, targets_a, targets_b, lam = mixup_data(images, labels, cfg.MIXUP_ALPHA, cfg.DEVICE)
        else:
            targets_a, targets_b, lam = labels, labels, 1.0
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            spike_out = model(images)
            loss = loss_fn_robust(spike_out, targets_a, targets_b, lam, cfg.LABEL_SMOOTHING) if use_mix else F.cross_entropy(spike_out, labels, label_smoothing=cfg.LABEL_SMOOTHING)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, predicted = spike_out.max(1)
        total += labels.size(0)
        correct += predicted.eq(targets_a).sum().item()
        pbar.set_postfix({'Perda': f'{loss.item():.4f}', 'Acur.': f'{100.*correct/total:.2f}%', 'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'})
    return running_loss / len(dataloader), 100. * correct / total

@torch.no_grad()
def validate_one_epoch(model, dataloader, cfg):
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    pbar = tqdm(dataloader, desc="Validação")
    for images, labels in pbar:
        images, labels = images.to(cfg.DEVICE), labels.to(cfg.DEVICE)
        with torch.cuda.amp.autocast():
            spike_out = model(images)
        _, predicted = spike_out.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({'Acur. Val.': f'{100.*correct/total:.2f}%'})
    return 100. * correct / total, all_preds, all_labels

# --- FUNÇÃO PRINCIPAL DE TREINAMENTO ---
def main():
    cfg = Hyperparameters()
    torch.backends.cudnn.benchmark = True

    print("="*60)
    print("Iniciando Pipeline de Treinamento SNN para Classificação de Pneumopatia")
    print(f"Dispositivo: {cfg.DEVICE} ({torch.cuda.get_device_name(0) if cfg.DEVICE.type == 'cuda' else 'CPU'})")
    print(f"Tamanho Imagem: {cfg.IMG_SIZE}x{cfg.IMG_SIZE} (Grayscale)")
    print(f"Batch Size: {cfg.BATCH_SIZE}")
    print("="*60)

    train_transform, val_transform = get_transforms(cfg.IMG_SIZE)
    full_dataset = datasets.ImageFolder(cfg.DATA_DIR)
    num_classes = len(full_dataset.classes)
    
    # Divisão estratificada seria ideal, mas random_split é uma boa aproximação
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(range(len(full_dataset)), [train_size, val_size])
    
    train_dataset = torch.utils.data.Subset(datasets.ImageFolder(cfg.DATA_DIR, transform=train_transform), train_indices)
    val_dataset = torch.utils.data.Subset(datasets.ImageFolder(cfg.DATA_DIR, transform=val_transform), val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE * 2, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    print(f"Dataset Carregado: {len(train_dataset)} imagens de treino, {len(val_dataset)} de validação.")
    print(f"Classes Detectadas ({num_classes}): {full_dataset.classes}")
    
    model = SNN_Pneumopathy_Classifier(num_classes, beta=cfg.BETA, dropout=cfg.DROPOUT).to(cfg.DEVICE)
    try:
        model = torch.compile(model)
        print("Modelo otimizado com torch.compile!")
    except Exception:
        print("torch.compile não disponível. Usando modelo padrão.")
        
    optimizer = optim.AdamW(model.parameters(), lr=cfg.INITIAL_LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS, eta_min=cfg.MIN_LR)
    scaler = torch.cuda.amp.GradScaler()

    best_val_acc, patience_counter = 0, 0
    MODEL_SAVE_PATH = "snn_pneumopathy_classifier.pth"

    for epoch in range(cfg.EPOCHS):
        print(f"\n--- Época {epoch+1}/{cfg.EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, cfg)
        val_acc, _, _ = validate_one_epoch(model, val_loader, cfg)
        scheduler.step()
        
        print(f"Fim da Época {epoch+1}: Perda Treino: {train_loss:.4f} | Acur. Treino: {train_acc:.2f}% | Acur. Validação: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"🚀 Novo melhor modelo salvo em '{MODEL_SAVE_PATH}' com {best_val_acc:.2f}% de acurácia!")
        else:
            patience_counter += 1
        
        if patience_counter >= cfg.PATIENCE:
            print(f"Parada antecipada na época {epoch+1}. A acurácia não melhora há {cfg.PATIENCE} épocas.")
            break
            
    print("\n--- Avaliação Final com o Melhor Modelo ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    final_acc, final_preds, final_labels = validate_one_epoch(model, val_loader, cfg)
    print(f"Acurácia final no conjunto de validação: {final_acc:.2f}%")
    print("\nRelatório de Classificação:")
    print(classification_report(final_labels, final_preds, target_names=full_dataset.classes, zero_division=0))
    
    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset.classes, yticklabels=full_dataset.classes)
    plt.title('Matriz de Confusão - Classificador de Pneumopatia')
    plt.xlabel('Predição'); plt.ylabel('Real')
    plt.savefig('confusion_matrix_pneumopathy.png'); plt.show()


if __name__ == "__main__":
    # Verifica se o diretório de dados existe antes de iniciar
    if not os.path.exists(Hyperparameters.DATA_DIR) or not os.listdir(Hyperparameters.DATA_DIR):
        print("="*60)
        print(f"ERRO: Diretório do dataset '{Hyperparameters.DATA_DIR}' não encontrado ou está vazio!")
        print("Por favor, crie o diretório e coloque as imagens de treino,")
        print("organizadas em subpastas por classe (ex: 'NORMAL', 'PNEUMONIA').")
        print("="*60)
    else:
        main()