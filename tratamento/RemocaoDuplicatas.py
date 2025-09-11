# -----------------------------------------------------------------------------
# Script: Analisador e Limpador de Dataset de Imagens
# Descrição: Este script utiliza um modelo VGG-16 para extrair características
#            de imagens, identificar e remover duplicatas, e gerar
#            visualizações (t-SNE e animações de convolução).
# Última Atualização: 10 de setembro de 2025
# -----------------------------------------------------------------------------

# --- Importações ---
import argparse
import gc
import hashlib
import io
import os
import shutil
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import imageio.v2 as imageio  # CORREÇÃO: Usar v2 para evitar DeprecationWarning
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm


# --- Classes e Funções Utilitárias ---

def clear_memory() -> None:
    """Força a limpeza da memória GPU (se disponível) e RAM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def get_unique_filename(data_dir: str, base_name: str, extension: str) -> str:
    """Gera um nome de arquivo único combinando nome base, hash do dataset e timestamp."""
    dataset_hash = hashlib.md5(data_dir.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    return f"{base_name}_{dataset_name}_{dataset_hash}_{timestamp}.{extension}"

class CLAHETransform:
    """Transformação customizada que aplica CLAHE para melhorar o contraste local."""
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
        clahe_img_np = self.clahe.apply(img_gray)
        clahe_img_pil = Image.fromarray(clahe_img_np)
        return clahe_img_pil.convert('RGB')


# --- Lógica Principal do Script ---

def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    sample_size: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extrai vetores de características de um dataset usando um modelo pré-treinado."""
    model.eval()
    features_list, labels_list, image_paths_list = [], [], []
    total_processed = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extraindo Características"):
            images = images.to(device)
            features = model(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

            start_idx = total_processed
            actual_batch_size = images.size(0)
            current_dataset = dataloader.dataset
            if isinstance(current_dataset, Subset):
                indices = current_dataset.indices[start_idx : start_idx + actual_batch_size]
                paths_batch = [current_dataset.dataset.samples[i][0] for i in indices]
            else:
                paths_batch = [current_dataset.samples[i][0] for i in range(start_idx, start_idx + actual_batch_size)]
            
            image_paths_list.extend(paths_batch)
            total_processed += actual_batch_size

            if sample_size is not None and total_processed >= sample_size:
                break

    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)

    if sample_size is not None:
        features_array, labels_array, image_paths_list = (
            features_array[:sample_size],
            labels_array[:sample_size],
            image_paths_list[:sample_size],
        )

    return features_array, labels_array, image_paths_list

def plot_scatter(
    points_2d: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str,
    output_base_dir: str
) -> None:
    """Gera e salva um gráfico de dispersão 2D (t-SNE)."""
    plt.close('all')
    plt.figure(figsize=(16, 12))
    sns.scatterplot(
        x=points_2d[:, 0], y=points_2d[:, 1], hue=[class_names[l] for l in labels],
        palette=sns.color_palette("hsv", len(class_names)),
        legend="full", alpha=0.8, s=60
    )
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel("Componente t-SNE 1", fontsize=12)
    plt.ylabel("Componente t-SNE 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_plot_dir = os.path.join(output_base_dir, "analise_similaridade")
    os.makedirs(output_plot_dir, exist_ok=True)
    filepath = os.path.join(output_plot_dir, get_unique_filename(output_base_dir, "grafico_dispersao", "png"))

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de dispersão salvo em '{filepath}'")
    plt.close()

# CORREÇÃO COMPLETA: Função inteira substituída para garantir frames de mesmo tamanho
def visualize_conv_process(
    model: nn.Module,
    dataset: Union[Dataset, Subset],
    class_names: List[str],
    device: torch.device,
    output_base_dir: str
) -> None:
    """Visualiza as ativações das camadas convolucionais para uma imagem de cada classe."""
    print("\n" + "="*80)
    print("🎬 INICIANDO GERAÇÃO DE ANIMAÇÕES DO PROCESSO DE CONVOLUÇÃO")
    print("="*80)
    
    conv_layers = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]
    layer_names = [f"Conv_{i+1}" for i in range(len(conv_layers))]

    feature_maps: Dict[str, torch.Tensor] = {}

    def get_features_hook(name: str):
        def hook(model: nn.Module, input: Any, output: torch.Tensor):
            feature_maps[name] = output.detach()
        return hook

    try:
        underlying_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        targets = underlying_dataset.targets if hasattr(underlying_dataset, 'targets') else [s[1] for s in underlying_dataset.samples]
        _, first_img_indices = np.unique(targets, return_index=True)
    except Exception as e:
        print(f"⚠️ Aviso: Não foi possível obter targets automaticamente ({e}). Usando as primeiras imagens.")
        first_img_indices = list(range(min(len(class_names), 5)))

    model.eval()
    animation_output_dir = os.path.join(output_base_dir, "animations")
    os.makedirs(animation_output_dir, exist_ok=True)

    for class_idx, img_idx in enumerate(first_img_indices):
        if class_idx >= len(class_names): break
        class_name = class_names[class_idx]
        print(f"\n🖼️  Processando imagem para a classe: '{class_name}'...")

        try:
            img_tensor, _ = dataset[img_idx]
            img_tensor = img_tensor.unsqueeze(0).to(device)

            hooks = [layer.register_forward_hook(get_features_hook(name)) for name, layer in zip(layer_names, conv_layers)]

            with torch.no_grad():
                _ = model(img_tensor)
            
            for hook in hooks:
                hook.remove()

            frames = []
            
            original_img_view = img_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
            mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
            original_img_view = np.clip(std * original_img_view + mean, 0, 1)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(original_img_view)
            ax.set_title(f"Imagem Original: {class_name}", fontsize=14)
            ax.axis('off')
            plt.tight_layout()

            with io.BytesIO() as buf:
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                frames.append(imageio.imread(buf))
            plt.close(fig)

            for name in layer_names:
                if name in feature_maps:
                    maps = feature_maps[name].cpu().squeeze(0)
                    avg_feature_map = torch.mean(maps, 0).numpy()
                    
                    fig, ax = plt.subplots(figsize=(8, 8))
                    im = ax.imshow(avg_feature_map, cmap='viridis')
                    ax.set_title(f"Saída Média da Camada: {name}", fontsize=14)
                    ax.axis('off')
                    fig.colorbar(im, ax=ax, shrink=0.85)
                    plt.tight_layout()

                    with io.BytesIO() as buf:
                        plt.savefig(buf, format='png', bbox_inches='tight')
                        buf.seek(0)
                        frames.append(imageio.imread(buf))
                    plt.close(fig)

            if frames:
                gif_filename = get_unique_filename(animation_output_dir, f"animacao_conv_{class_name}", "gif")
                gif_filepath = os.path.join(animation_output_dir, gif_filename)
                imageio.mimsave(gif_filepath, frames, duration=0.8, loop=0)
                print(f"✅ Animação salva em '{gif_filepath}'")

            feature_maps.clear()

        except Exception as e:
            print(f"❌ Erro ao processar a classe '{class_name}': {e}")
            traceback.print_exc()
            continue

def main(args: argparse.Namespace) -> None:
    """Função principal que orquestra a análise e limpeza do dataset."""
    print("="*80)
    print("🔍 ANÁLISE E LIMPEZA DE DUPLICATAS NO DATASET")
    print("="*80)
    
    if not os.path.isdir(args.data_dir):
        print(f"❌ ERRO: Diretório '{args.data_dir}' não encontrado.")
        return
    
    print(f"📁 Dataset Original: {args.data_dir}")
    print(f"🖥️  Dispositivo: {args.device}")
    print(f"📂 Diretório de Saída: {args.output_dir}")
    print(f"🎯 Threshold de De-duplicação: {args.dedup_threshold}")
    print("="*60)

    transform = transforms.Compose([
        CLAHETransform(),
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        full_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
        dataset = full_dataset
        if args.sample_size:
            indices = np.random.choice(len(full_dataset), min(args.sample_size, len(full_dataset)), replace=False)
            dataset = Subset(full_dataset, indices)
            print(f"📊 Usando uma amostra de {len(dataset)} imagens.")
        else:
            print(f"📊 Analisando o dataset completo com {len(dataset)} imagens.")

        class_names = full_dataset.classes
        print(f"📋 Classes encontradas ({len(class_names)}): {', '.join(class_names)}")

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        print("\n🤖 Carregando modelo VGG-16 pré-treinado...")
        model_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model_features.classifier = nn.Identity()
        model_features.to(args.device)
        print("✅ Modelo carregado.")

        features, labels, image_paths = extract_features(model_features, dataloader, args.device, args.sample_size)
        del model_features; clear_memory()

        print("\n🗑️  Iniciando processo de de-duplicação...")
        to_keep_mask = np.ones(len(features), dtype=bool)
        for class_id, class_name in enumerate(tqdm(class_names, desc="De-duplicando por classe")):
            class_indices = np.where(labels == class_id)[0]
            if len(class_indices) < 2: continue
            class_features = features[class_indices]
            distance_matrix = squareform(pdist(class_features, 'euclidean'))
            np.fill_diagonal(distance_matrix, np.inf)
            duplicate_pairs = np.argwhere(distance_matrix < args.dedup_threshold)
            if duplicate_pairs.size > 0:
                indices_to_remove_local = {j for i, j in duplicate_pairs if i < j}
                global_indices_to_remove = class_indices[list(indices_to_remove_local)]
                to_keep_mask[global_indices_to_remove] = False
        
        total_removed = len(features) - int(np.sum(to_keep_mask))
        print(f"✅ De-duplicação concluída. {total_removed} imagens marcadas para remoção.")
        
        features_clean = features[to_keep_mask]
        labels_clean = labels[to_keep_mask]
        image_paths_clean = [path for i, path in enumerate(image_paths) if to_keep_mask[i]]

        if image_paths_clean:
            print("\n💾 Salvando o dataset limpo...")
            output_dataset_path = os.path.join(args.output_dir, "dataset_deduplicado")
            if os.path.exists(output_dataset_path): shutil.rmtree(output_dataset_path)
            
            for path, label in tqdm(zip(image_paths_clean, labels_clean), desc="Copiando imagens limpas", total=len(image_paths_clean)):
                class_name = class_names[label]
                dest_dir = os.path.join(output_dataset_path, class_name)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(path, dest_dir)
            print(f"✅ {len(image_paths_clean)} imagens únicas salvas em '{output_dataset_path}'.")
        else:
            output_dataset_path = "Nenhum diretório criado (nenhuma imagem restou)"

        if len(features_clean) > 1:
            print("\n📊 Gerando visualização t-SNE...")
            n_samples_clean = features_clean.shape[0]

            # CORREÇÃO: Lógica da perplexity robusta para amostras pequenas
            perplexity_value = max(1, min(30, n_samples_clean - 1))
            print(f"ℹ️  Amostra com {n_samples_clean} imagens. Usando perplexity = {perplexity_value}")

            tsne = TSNE(
                n_components=2, verbose=0, perplexity=perplexity_value,
                max_iter=1000, # CORREÇÃO: `n_iter` renomeado para `max_iter`
                random_state=42, n_jobs=-1
            )
            try:
                points_2d_clean = tsne.fit_transform(features_clean)
                title = f"Dataset Limpo ({n_samples_clean} imagens, {total_removed} removidas)"
                plot_scatter(points_2d_clean, labels_clean, class_names, title, args.output_dir)
            except Exception as e:
                print(f"⚠️ Erro na geração do t-SNE: {e}")

        if args.generate_animations:
            print("\n🎬 Recarregando modelo para gerar animações...")
            model_conv = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(args.device)
            visualize_conv_process(model_conv, full_dataset, class_names, args.device, args.output_dir)
            del model_conv; clear_memory()
        
        print("\n" + "="*60)
        print("🎉 ANÁLISE E LIMPEZA DE DATASET CONCLUÍDAS COM SUCESSO!")
        print(f"📊 Dataset original: {len(image_paths)} imagens")
        print(f"🗑️  Duplicatas removidas: {total_removed} imagens")
        print(f"✨ Dataset final: {len(image_paths_clean)} imagens limpas")
        print(f"📁 Salvo em: {output_dataset_path}")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Erro fatal durante a análise: {e}")
        traceback.print_exc()
    finally:
        clear_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analisa, remove duplicatas e visualiza um dataset de imagens.")
    
    # CORREÇÃO: Argumento alterado para opcional (--data_dir) para usar o 'default' corretamente
    parser.add_argument("--data_dir", type=str, default='/home/mob4ai-001/Pibic_Artigo/preprocessado_big_base', help="Caminho para o diretório raiz do dataset.")
    parser.add_argument("--output_dir", type=str, default="dataset_analisado_vfinal", help="Diretório para salvar os resultados.")
    parser.add_argument("--dedup_threshold", type=float, default=1e-5, help="Limiar de distância para considerar imagens como duplicatas.")
    parser.add_argument("--img_size", type=int, default=224, help="Tamanho para redimensionar as imagens.")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamanho do batch para processamento.")
    parser.add_argument("--sample_size", type=int, default=None, help="Processa apenas N imagens para um teste rápido (opcional).")
    parser.add_argument("--no-animations", dest='generate_animations', action='store_false', help="Desativa a geração de animações de convolução.")
    
    default_workers = 0 if os.name == 'nt' else min(os.cpu_count() or 1, 8)
    parser.add_argument("--num_workers", type=int, default=default_workers, help="Número de workers para o DataLoader.")

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    main(args)