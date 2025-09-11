# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import cv2
import logging
import argparse
import gc
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

# Imports de bibliotecas externas
from torchvision import models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURAÇÃO E UTILITÁRIOS ---
def set_seeds(seed=42):
    """Configura seeds para reprodutibilidade."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file):
    """Configura um logger para salvar em um arquivo específico."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Limpa handlers para evitar logs duplicados em execuções sequenciais
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

# --- 2. PREPARAÇÃO DE DADOS ---
class MedicalImageDataset(Dataset):
    """Dataset para carregar imagens e seus labels."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Imagem não encontrada ou corrompida: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label

def get_transforms(image_size=224, is_training=True, augmentation_intensity='medium'):
    """Retorna um pipeline de transformações com diferentes intensidades de augmentation."""
    imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    if is_training:
        if augmentation_intensity == 'light':
            return A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.Normalize(mean=imagenet_mean, std=imagenet_std),
                ToTensorV2(),
            ])
        elif augmentation_intensity == 'medium':
            return A.Compose([
                A.Resize(image_size, image_size),
                A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=imagenet_mean, std=imagenet_std),
                ToTensorV2(),
            ])
        elif augmentation_intensity == 'heavy':
            return A.Compose([
                A.Resize(image_size, image_size),
                A.Affine(scale=(0.8, 1.2), rotate=(-25, 25), p=0.7),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
                A.Normalize(mean=imagenet_mean, std=imagenet_std),
                ToTensorV2(),
            ])
    
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2()
    ])

def create_dataloaders(args, logger):
    """Prepara e divide os dados, criando os DataLoaders."""
    all_paths, all_labels = [], []
    for class_idx, class_name in enumerate(args.classes):
        class_path = Path(args.data_dir) / class_name
        if not class_path.is_dir():
            logger.warning(f"Diretório da classe não encontrado: {class_path}")
            continue
        paths = list(class_path.glob('*[.png,.jpg,.jpeg]'))
        all_paths.extend(paths)
        all_labels.extend([class_idx] * len(paths))

    if not all_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {args.data_dir}")

    # --- SEÇÃO MODIFICADA ---
    # Subamostragem estratificada do dataset, se uma porcentagem for especificada.
    if args.dataset_percentage < 1.0:
        logger.info(f"Usando apenas {args.dataset_percentage * 100:.1f}% do dataset total.")
        # Usamos train_test_split para fazer a subamostragem, descartando o resto.
        all_paths, _, all_labels, _ = train_test_split(
            all_paths, all_labels, 
            train_size=args.dataset_percentage, 
            random_state=args.seed, 
            stratify=all_labels
        )
        logger.info(f"Tamanho do dataset após subamostragem: {len(all_paths)} imagens.")
    # --- FIM DA SEÇÃO MODIFICADA ---

    # Divisão estratificada para manter a proporção das classes (60% treino, 20% val, 20% teste)
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=args.seed, stratify=all_labels)
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=0.25, # 0.25 de 80% resulta em 20% do total
        random_state=args.seed, stratify=train_val_labels)
    
    logger.info(f"Dataset dividido em: {len(train_paths)} treino, {len(val_paths)} validação, {len(test_paths)} teste.")

    datasets = {
        'train': MedicalImageDataset(train_paths, train_labels, 
                                     transform=get_transforms(args.image_size, is_training=True, 
                                                              augmentation_intensity=args.augmentation_intensity)),
        'val': MedicalImageDataset(val_paths, val_labels, 
                                   transform=get_transforms(args.image_size, is_training=False)),
        'test': MedicalImageDataset(test_paths, test_labels, 
                                    transform=get_transforms(args.image_size, is_training=False))
    }

    # Sampler para lidar com desbalanceamento de classes no treino
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    sampler_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(weights=sampler_weights, num_samples=len(sampler_weights), replacement=True)

    return {
        'train': DataLoader(datasets['train'], batch_size=args.batch_size, sampler=sampler, 
                            num_workers=args.num_workers, pin_memory=True),
        'val': DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False, 
                          num_workers=args.num_workers, pin_memory=True),
        'test': DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True)
    }, torch.FloatTensor(class_weights)

# --- 3. ARQUITETURA DO MODELO ---
class VGG16Classifier(nn.Module):
    """Classificador customizado baseado na VGG16 pré-treinada."""
    def __init__(self, num_classes, dropout=0.5, hidden_units=4096, freeze_features=True):
        super().__init__()
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Congela as camadas de extração de características (transfer learning)
        if freeze_features:
            for param in self.vgg.features.parameters():
                param.requires_grad = False
        else:
            # Fine-tuning: descongelar últimas camadas convolucionais
            for param in self.vgg.features[-6:].parameters():
                param.requires_grad = True
                
        # Substitui o classificador original por um novo
        num_features = self.vgg.classifier[0].in_features
        self.vgg.classifier = nn.Sequential(
            nn.Linear(num_features, hidden_units),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_units // 2, num_classes),
        )

    def forward(self, x):
        return self.vgg(x)

# --- 4. LÓGICA DE TREINAMENTO E AVALIAÇÃO ---
def run_trial(args, logger):
    """Executa um ciclo completo de treinamento e avaliação."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Parâmetros: LR={args.learning_rate}, Dropout={args.dropout}, "
                f"Batch={args.batch_size}, Optimizer={args.optimizer}, "
                f"Hidden={args.hidden_units}, Aug={args.augmentation_intensity}")
    logger.info(f"Usando dispositivo: {device}")

    start_time = time.time()

    # 1. Preparação dos dados
    dataloaders, class_weights = create_dataloaders(args, logger)
    
    # 2. Construção do modelo
    model = VGG16Classifier(
        num_classes=len(args.classes), 
        dropout=args.dropout,
        hidden_units=args.hidden_units,
        freeze_features=args.freeze_features
    ).to(device)
    
    # 3. Configuração do otimizador
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                              momentum=0.9, weight_decay=args.weight_decay)
    
    # 4. Scheduler de learning rate
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    else:
        scheduler = None
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    best_val_f1 = 0.0
    best_model_state = None
    training_history = {'train_loss': [], 'val_f1': [], 'val_acc': []}

    # 5. Loop de Treinamento
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for inputs, targets in train_loader_tqdm:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader_tqdm))

        epoch_loss = running_loss / len(dataloaders['train'])
        training_history['train_loss'].append(epoch_loss)

        # Validação
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for inputs, targets in dataloaders['val']:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(targets.cpu().numpy())
        
        val_f1 = f1_score(val_true, val_preds, average='macro', zero_division=0)
        val_acc = accuracy_score(val_true, val_preds)
        training_history['val_f1'].append(val_f1)
        training_history['val_acc'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} -> Loss: {epoch_loss:.4f}, Val F1: {val_f1:.4f}, Val Acc: {val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict()
            logger.info(f"🎉 Novo melhor F1-Score de validação: {best_val_f1:.4f}")

        # Atualiza scheduler
        if scheduler:
            if args.lr_scheduler == 'plateau':
                scheduler.step(val_f1)
            else:
                scheduler.step()

    # 6. Avaliação final no teste
    logger.info("\n--- Avaliação Final no Conjunto de Teste ---")
    if best_model_state is None:
        logger.warning("Nenhum modelo foi salvo. Usando modelo da última época.")
        best_model_state = model.state_dict()
        
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, args.model_save_path)
    
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloaders['test'], desc="[Teste]"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(targets.cpu().numpy())
    
    # Métricas finais
    test_f1_macro = f1_score(test_true, test_preds, average='macro', zero_division=0)
    test_f1_weighted = f1_score(test_true, test_preds, average='weighted', zero_division=0)
    test_accuracy = accuracy_score(test_true, test_preds)
    test_precision = precision_score(test_true, test_preds, average='macro', zero_division=0)
    test_recall = recall_score(test_true, test_preds, average='macro', zero_division=0)
    
    # Relatório de classificação
    report = classification_report(test_true, test_preds, target_names=args.classes, 
                                   zero_division=0, output_dict=True)
    
    # Matriz de confusão
    cm = confusion_matrix(test_true, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=args.classes, yticklabels=args.classes)
    plt.title('Matriz de Confusão')
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Prevista')
    cm_path = Path(args.model_save_path).parent / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    training_time = time.time() - start_time
    
    # Retorna resultado estruturado
    return {
        'validation': {
            'best_f1_macro': best_val_f1,
            'best_accuracy': max(training_history['val_acc'])
        },
        'test': {
            'f1_macro': test_f1_macro,
            'f1_weighted': test_f1_weighted,
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall
        },
        'training': {
            'time_seconds': training_time,
            'epochs_completed': args.epochs,
            'final_train_loss': training_history['train_loss'][-1] if training_history['train_loss'] else 0,
            'history': training_history
        },
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

# --- 5. ORQUESTRADOR PRINCIPAL COM GRID SEARCH EXPANDIDO ---
def main():
    parser = argparse.ArgumentParser(description="Framework de Treinamento com VGG16 e Grid Search Expandido")
    parser.add_argument('--data_dir', type=str, default='/home/mob4ai-001/Pibic_Artigo/data/bigger_base_clean')
    parser.add_argument('--classes', nargs='+', default=['COVID', 'Lung_Opacity', 'Normal', 'pneumonia_bacterial', 'Viral Pneumonia'])
    parser.add_argument('--results_dir', type=str, default='training_results_30%')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=os.cpu_count() // 2)
    parser.add_argument('--seed', type=int, default=42)
    
    # --- NOVO ARGUMENTO ADICIONADO ---
    parser.add_argument('--dataset_percentage', type=float, default=0.3, 
                        help="Porcentagem do dataset a ser usado (e.g., 0.05 para 5%). Deve estar entre 0.0 e 1.0.")
    # --- FIM DO NOVO ARGUMENTO ---

    base_args = parser.parse_args()
    
    # Validação do novo argumento
    if not 0.0 < base_args.dataset_percentage <= 1.0:
        raise ValueError("O argumento --dataset_percentage deve ser um valor entre 0.0 (exclusivo) e 1.0 (inclusivo).")
        
    set_seeds(base_args.seed)
    
    # 🚀 GRID SEARCH EXPANDIDO COM MAIS PARÂMETROS
    param_grid = {
        'learning_rate': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
        'dropout': [0.3, 0.4, 0.5, 0.6, 0.7],
        'batch_size': [8, 16, 32, 64],
        'optimizer': ['adamw', 'adam', 'sgd'],
        'weight_decay': [1e-5, 1e-4, 1e-3],
        'lr_scheduler': ['cosine', 'step', 'plateau', None],
        'hidden_units': [2048, 4096, 8192],
        'augmentation_intensity': ['light', 'medium', 'heavy'],
        'freeze_features': [True, False]
    }
    
    # Para testes rápidos, use um grid menor
    if os.environ.get('QUICK_TEST', '0') == '1':
        param_grid = {
            'learning_rate': [1e-3, 5e-4],
            'dropout': [0.4, 0.5],
            'batch_size': [16, 32],
            'optimizer': ['adamw', 'adam'],
            'weight_decay': [1e-4],
            'lr_scheduler': ['cosine', None],
            'hidden_units': [4096],
            'augmentation_intensity': ['medium'],
            'freeze_features': [True]
        }
    
    grid = ParameterGrid(param_grid)
    results = []
    
    # Metadata do experimento
    experiment_metadata = {
        'start_time': datetime.now().isoformat(),
        'total_combinations': len(grid),
        'base_args': vars(base_args),
        'param_grid': param_grid,
        'system_info': {
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'cpu_count': os.cpu_count()
        }
    }
    
    print(f"🚀 Iniciando Grid Search com {len(grid)} combinações de hiperparâmetros.")
    print(f"📊 Parâmetros sendo testados:")
    for key, values in param_grid.items():
        print(f"   {key}: {values}")
    
    for i, params in enumerate(grid):
        run_args = argparse.Namespace(**vars(base_args))
        run_args = argparse.Namespace(**{**vars(run_args), **params})
        
        # Nome único para cada execução
        run_name = (f"run{i+1:03d}_lr{params['learning_rate']}_dr{params['dropout']}_"
                    f"bs{params['batch_size']}_opt{params['optimizer']}_"
                    f"aug{params['augmentation_intensity']}")
        output_dir = Path(run_args.results_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run_args.model_save_path = output_dir / "best_model.pth"
        logger = setup_logging(output_dir / "training.log")

        print(f"\n{'='*50}\n🔄 Trial {i+1}/{len(grid)}: {run_name}\n{'='*50}")
        
        try:
            trial_results = run_trial(run_args, logger)
            
            result_entry = {
                'trial_id': i + 1,
                'run_name': run_name,
                'parameters': params,
                'results': trial_results,
                'status': 'success',
                'error_message': None,
                'output_directory': str(output_dir)
            }
            results.append(result_entry)
            
            print(f"✅ Trial {i+1} concluído com sucesso!")
            print(f"   📈 Melhor F1 Val: {trial_results['validation']['best_f1_macro']:.4f}")
            print(f"   🎯 F1 Teste: {trial_results['test']['f1_macro']:.4f}")
            
        except Exception as e:
            error_msg = str(e)
            logger.critical(f"❌ Erro fatal no trial {run_name}: {error_msg}", exc_info=True)
            
            result_entry = {
                'trial_id': i + 1,
                'run_name': run_name,
                'parameters': params,
                'results': None,
                'status': 'error',
                'error_message': error_msg,
                'output_directory': str(output_dir)
            }
            results.append(result_entry)
            print(f"❌ Trial {i+1} falhou: {error_msg}")
        
        # Limpeza de memória
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # --- RELATÓRIO FINAL EM JSON ---
    experiment_metadata['end_time'] = datetime.now().isoformat()
    experiment_metadata['total_runtime_minutes'] = (
        datetime.fromisoformat(experiment_metadata['end_time']) - 
        datetime.fromisoformat(experiment_metadata['start_time'])
    ).total_seconds() / 60
    
    # Filtra apenas resultados com sucesso para ranking
    successful_results = [r for r in results if r['status'] == 'success']
    successful_results.sort(key=lambda x: x['results']['validation']['best_f1_macro'], reverse=True)
    
    # Estatísticas sumárias
    summary_stats = {
        'total_trials': len(results),
        'successful_trials': len(successful_results),
        'failed_trials': len(results) - len(successful_results),
        'success_rate': len(successful_results) / len(results) * 100 if results else 0
    }
    
    if successful_results:
        best_result = successful_results[0]
        test_f1_scores = [r['results']['test']['f1_macro'] for r in successful_results]
        val_f1_scores = [r['results']['validation']['best_f1_macro'] for r in successful_results]
        
        summary_stats.update({
            'best_validation_f1': max(val_f1_scores),
            'best_test_f1': max(test_f1_scores),
            'mean_validation_f1': np.mean(val_f1_scores),
            'std_validation_f1': np.std(val_f1_scores),
            'mean_test_f1': np.mean(test_f1_scores),
            'std_test_f1': np.std(test_f1_scores),
            'best_parameters': best_result['parameters']
        })
    
    # Relatório final
    final_report = {
        'experiment_metadata': experiment_metadata,
        'summary_statistics': summary_stats,
        'results': results,
        'ranking_by_validation_f1': [
            {
                'rank': idx + 1,
                'trial_id': r['trial_id'],
                'run_name': r['run_name'],
                'validation_f1': r['results']['validation']['best_f1_macro'],
                'test_f1': r['results']['test']['f1_macro'],
                'parameters': r['parameters']
            }
            for idx, r in enumerate(successful_results[:10])  # Top 10
        ]
    }
    
    # Salva o relatório JSON
    results_dir = Path(base_args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"grid_search_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    
    # --- EXIBIÇÃO DOS RESULTADOS ---
    print("\n\n" + "="*80)
    print("🏁 RELATÓRIO FINAL DO GRID SEARCH")
    print("="*80)
    
    print(f"\n📊 Estatísticas Gerais:")
    print(f"   • Total de experimentos: {summary_stats['total_trials']}")
    print(f"   • Experimentos bem-sucedidos: {summary_stats['successful_trials']}")
    print(f"   • Taxa de sucesso: {summary_stats['success_rate']:.1f}%")
    print(f"   • Tempo total: {experiment_metadata['total_runtime_minutes']:.1f} minutos")
    
    if successful_results:
        print(f"\n🎯 Melhores Resultados:")
        print(f"   • Melhor F1 (Validação): {summary_stats['best_validation_f1']:.4f}")
        print(f"   • Melhor F1 (Teste): {summary_stats['best_test_f1']:.4f}")
        print(f"   • Média F1 (Validação): {summary_stats['mean_validation_f1']:.4f} ± {summary_stats['std_validation_f1']:.4f}")
        
        print(f"\n🏆 Top 5 Configurações:")
        for i, entry in enumerate(final_report['ranking_by_validation_f1'][:5]):
            print(f"   {entry['rank']}. Trial {entry['trial_id']} - Val F1: {entry['validation_f1']:.4f}, Test F1: {entry['test_f1']:.4f}")
            print(f"      Parâmetros: {entry['parameters']}")
            
        print(f"\n💾 Relatório completo salvo em: {report_path}")
        print(f"📁 Modelos e logs salvos em: {base_args.results_dir}/")
    else:
        print("\n❌ Nenhum experimento foi concluído com sucesso.")
    
    print("="*80)

if __name__ == '__main__':
    main()