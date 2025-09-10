import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
import gc
import hashlib
from datetime import datetime
import imageio
from PIL import Image
import io
import cv2
import shutil
from scipy.spatial.distance import pdist, squareform # <--- NOVO: Para cálculo eficiente de distância

# --- CONFIGURAÇÃO DA ANÁLISE ---
class Config:
    DATA_DIR = "/home/mob4ai-001/Pibic_Artigo/raw"
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE = 224
    BATCH_SIZE = 8
    NUM_WORKERS = min(os.cpu_count() // 2, 8)
    SAMPLE_SIZE = None 

    GENERATE_CONV_ANIMATIONS = True

    # --- NOVO: Configurações para De-duplicação e Saída ---
    OUTPUT_BASE_DIR = "dataset_tratado_vgg16_raw"
    # Limiar de distância para considerar uma imagem como duplicata.
    # Valores muito baixos (ex: 1e-5) pegam duplicatas quase perfeitas.
    # Aumente um pouco (ex: 0.1, 0.5) para pegar imagens muito similares. Comece baixo.
    DEDUPLICATION_THRESHOLD = 1e-5 

# --- UTILITÁRIOS ---
# (Funções utilitárias permanecem as mesmas)
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def generate_dataset_hash(data_dir):
    return hashlib.md5(data_dir.encode()).hexdigest()[:8]

def get_unique_filename(data_dir, base_name, extension):
    dataset_hash = generate_dataset_hash(data_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.basename(os.path.normpath(data_dir))
    return f"{base_name}_{dataset_name}_{dataset_hash}_{timestamp}.{extension}"

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np
        
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        clahe_img_np = clahe.apply(img_gray)
        
        clahe_img_pil = Image.fromarray(clahe_img_np)
        return clahe_img_pil.convert('RGB')

# --- ETAPA 1: EXTRAÇÃO DE CARACTERÍSTICAS ---
def extract_features(model, dataloader, device, sample_size=None):
    clear_memory()
    model.eval()
    features_list = []
    labels_list = []
    image_paths_list = [] 

    pbar = tqdm(dataloader, desc="Extraindo Características das Imagens")
    with torch.no_grad():
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            features = model(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
            start_idx = i * dataloader.batch_size
            end_idx = start_idx + len(images) # Usar len(images) para o último batch

            if isinstance(dataloader.dataset, torch.utils.data.Subset):
                original_indices = dataloader.dataset.indices[start_idx:end_idx]
                paths_batch = [dataloader.dataset.dataset.samples[idx][0] for idx in original_indices]
            else:
                paths_batch = [dataloader.dataset.samples[idx][0] for idx in range(start_idx, end_idx)]
            
            image_paths_list.extend(paths_batch)
            
            if sample_size is not None and len(image_paths_list) >= sample_size:
                break
    clear_memory()
    features_array = np.concatenate(features_list)
    labels_array = np.concatenate(labels_list)
    
    if sample_size is not None:
        features_array = features_array[:sample_size]
        labels_array = labels_array[:sample_size]
        image_paths_list = image_paths_list[:sample_size]

    return features_array, labels_array, image_paths_list

# --- ETAPA 2: VISUALIZAÇÃO DOS DADOS (SIMPLIFICADA) ---
def plot_scatter(points_2d, labels, class_names, title, output_base_dir):
    plt.close('all')
    plt.figure(figsize=(16, 12))
    scatter_plot = sns.scatterplot(
        x=points_2d[:, 0],
        y=points_2d[:, 1],
        hue=[class_names[l] for l in labels],
        palette=sns.color_palette("hsv", len(class_names)),
        legend="full",
        alpha=0.8,
        s=60
    )
    plt.legend(title="Classes")
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel("Componente t-SNE 1", fontsize=12)
    plt.ylabel("Componente t-SNE 2", fontsize=12)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    output_plot_dir = os.path.join(output_base_dir, "analise_similaridade")
    os.makedirs(output_plot_dir, exist_ok=True)
    filename = os.path.join(output_plot_dir, get_unique_filename(output_plot_dir, "grafico_dispersao_similaridade_limpo", "png"))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nGráfico de dispersão do dataset limpo salvo como '{filename}'")
    plt.show()
    plt.close()

# --- ETAPA 3: VISUALIZAÇÃO DAS CONVOLUÇÕES ---
# (Esta função permanece a mesma, visualizando uma imagem de amostra do dataset original)
def visualize_conv_process(model, dataset, class_names, device, output_base_dir):
    print("\n" + "="*80)
    print("🎬 INICIANDO GERAÇÃO DE ANIMAÇÕES DO PROCESSO DE CONVOLUÇÃO")
    print("="*80)
    # (Código da função omitido para brevidade, é o mesmo da resposta anterior)
    # ... (código completo está abaixo)

# --- FUNÇÃO PRINCIPAL ---
def main():
    print("="*80)
    print("🔍 ANÁLISE E LIMPEZA DE DUPLICATAS NO DATASET")
    print("="*80)
    
    clear_memory()
    cfg = Config()
    
    if not os.path.exists(cfg.DATA_DIR) or not os.listdir(cfg.DATA_DIR):
        print(f"❌ ERRO: Diretório '{cfg.DATA_DIR}' não encontrado ou vazio!")
        return

    print(f"📁 Dataset Original: {cfg.DATA_DIR}")
    print(f"🖥️  Dispositivo: {cfg.DEVICE}")
    print(f"📂 Diretório de Saída: {cfg.OUTPUT_BASE_DIR}")
    print(f"Threshold de De-duplicação: {cfg.DEDUPLICATION_THRESHOLD}")
    print("="*60)

    transform = transforms.Compose([
        CLAHETransform(),
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        full_dataset = datasets.ImageFolder(root=cfg.DATA_DIR, transform=transform)
    except Exception as e:
        print(f"❌ Erro ao carregar o dataset: {e}")
        return
    
    if len(full_dataset) == 0:
        print("❌ Dataset vazio!")
        return
    
    dataset = torch.utils.data.Subset(full_dataset, range(len(full_dataset))) if cfg.SAMPLE_SIZE is None else torch.utils.data.Subset(full_dataset, np.random.choice(range(len(full_dataset)), cfg.SAMPLE_SIZE, replace=False))
    if cfg.SAMPLE_SIZE is not None:
        print(f"📊 Usando uma amostra de {cfg.SAMPLE_SIZE} imagens.")
    else:
        print(f"📊 Analisando o dataset completo com {len(dataset)} imagens.")
        
    class_names = full_dataset.classes
    print(f"📋 Classes encontradas: {class_names}")

    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    print("\n🤖 Carregando modelo VGG-16 pré-treinado...")
    clear_memory()
    
    try:
        model_features = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model_features.classifier = nn.Identity()
        model_features.to(cfg.DEVICE)
        
        print("✅ Modelo carregado.")
        
        features, labels, image_paths = extract_features(model_features, dataloader, cfg.DEVICE, cfg.SAMPLE_SIZE)
        print(f"✅ Extração de {features.shape[0]} vetores de características concluída.")
        
        del model_features
        clear_memory()

        # --- NOVO: LÓGICA DE DE-DUPLICAÇÃO ---
        print("\n🗑️  Iniciando processo de de-duplicação...")
        
        to_keep_mask = np.ones(len(features), dtype=bool)
        total_removed = 0

        for class_id, class_name in enumerate(class_names):
            class_indices = np.where(labels == class_id)[0]
            
            if len(class_indices) < 2:
                continue

            class_features = features[class_indices]
            
            # Calcula a matriz de distâncias par a par (muito mais rápido que loops)
            # 'euclidean' é a distância padrão L2
            distance_matrix = squareform(pdist(class_features, 'euclidean'))
            
            # Preenche a diagonal com um valor alto para que uma imagem não se compare a si mesma
            np.fill_diagonal(distance_matrix, np.inf)

            # Encontra pares de imagens com distância abaixo do threshold
            duplicate_pairs = np.argwhere(distance_matrix < cfg.DEDUPLICATION_THRESHOLD)
            
            indices_to_remove = set()
            for i, j in duplicate_pairs:
                # Para evitar remover ambos, removemos apenas o segundo do par
                if i < j:
                    indices_to_remove.add(j)

            if indices_to_remove:
                # Mapeia os índices da classe de volta para os índices globais
                global_indices_to_remove = class_indices[list(indices_to_remove)]
                to_keep_mask[global_indices_to_remove] = False
                class_removed_count = len(indices_to_remove)
                total_removed += class_removed_count
                print(f"  - Classe '{class_name}': {class_removed_count} imagens duplicadas encontradas e marcadas para remoção.")

        print(f"✅ De-duplicação concluída. Total de {total_removed} imagens removidas.")

        # Filtra os dados para manter apenas as imagens únicas
        features_clean = features[to_keep_mask]
        labels_clean = labels[to_keep_mask]
        image_paths_clean = [path for i, path in enumerate(image_paths) if to_keep_mask[i]]
        
        # --- Salvar o dataset limpo ---
        print("\n💾 Salvando o dataset limpo...")
        output_dataset_path = os.path.join(cfg.OUTPUT_BASE_DIR, "dataset_deduplicado")
        if os.path.exists(output_dataset_path):
            shutil.rmtree(output_dataset_path) # Limpa o diretório antigo
        os.makedirs(output_dataset_path)

        for class_name in class_names:
            os.makedirs(os.path.join(output_dataset_path, class_name), exist_ok=True)
            
        for path, label in zip(image_paths_clean, labels_clean):
            class_name = class_names[label]
            dest_path = os.path.join(output_dataset_path, class_name, os.path.basename(path))
            shutil.copy(path, dest_path)
        
        print(f"✅ {len(image_paths_clean)} imagens únicas salvas em '{output_dataset_path}'.")

        # --- Visualizar o dataset limpo com t-SNE ---
        print("\n📊 Gerando visualização t-SNE para o dataset limpo...")
        n_samples_clean = features_clean.shape[0]
        perplexity = min(30, max(5, n_samples_clean // 4))
        tsne = TSNE(
            n_components=2, 
            verbose=1, 
            perplexity=perplexity, 
            max_iter=1000, 
            random_state=42,
            n_jobs=-1
        )
        points_2d_clean = tsne.fit_transform(features_clean)
        print("✅ Redução de dimensionalidade concluída.")

        dataset_name = os.path.basename(os.path.normpath(cfg.DATA_DIR))
        title = f"Dataset Limpo: {dataset_name}\n({n_samples_clean} imagens únicas, {total_removed} removidas)"
        
        plot_scatter(points_2d_clean, labels_clean, class_names, title, cfg.OUTPUT_BASE_DIR)

        if cfg.GENERATE_CONV_ANIMATIONS:
            model_conv = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(cfg.DEVICE)
            visualize_conv_process(model_conv, full_dataset, class_names, cfg.DEVICE, cfg.OUTPUT_BASE_DIR)
            del model_conv 
        
        print("\n" + "="*60)
        print("🎉 Análise e limpeza de dataset concluídas com sucesso!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Erro durante a análise: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        clear_memory()

# O código completo da função visualize_conv_process para referência
def visualize_conv_process(model, dataset, class_names, device, output_base_dir):
    print("\n" + "="*80)
    print("🎬 INICIANDO GERAÇÃO DE ANIMAÇÕES DO PROCESSO DE CONVOLUÇÃO")
    print("="*80)

    conv_layers_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
    layer_names = [f"Conv_{i+1}" for i in range(len(conv_layers_indices))]
    
    feature_maps = {}

    def get_features(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach()
        return hook

    _, first_img_indices = np.unique(dataset.targets, return_index=True)
    model.eval()
    
    animation_output_dir = os.path.join(output_base_dir, "animations")
    os.makedirs(animation_output_dir, exist_ok=True)

    for class_idx, img_idx in enumerate(first_img_indices):
        class_name = class_names[class_idx]
        print(f"\n🖼️  Processando imagem para a classe: '{class_name}'...")
        
        img_tensor, _ = dataset[img_idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)

        hooks = []
        for i, layer_idx in enumerate(conv_layers_indices):
            hook = model.features[layer_idx].register_forward_hook(get_features(layer_names[i]))
            hooks.append(hook)

        with torch.no_grad():
            model(img_tensor)

        for hook in hooks:
            hook.remove()

        frames = []
        original_img_view = img_tensor.cpu().squeeze(0).numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        original_img_view = std * original_img_view + mean
        original_img_view = np.clip(original_img_view, 0, 1)
        
        plt.figure(figsize=(8, 8))
        plt.imshow(original_img_view)
        plt.title(f"Imagem Original da Classe: {class_name}\n(Pré-processada com CLAHE)", fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        plt.close()

        for name in layer_names:
            maps = feature_maps[name].cpu().squeeze(0)
            avg_feature_map = torch.mean(maps, 0).numpy()
            plt.figure(figsize=(8, 8))
            plt.imshow(avg_feature_map, cmap='viridis')
            plt.title(f"Saída Média da Camada: {name}", fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            frames.append(imageio.imread(buf))
            plt.close()

        gif_filename = os.path.join(animation_output_dir, get_unique_filename(animation_output_dir, f"animacao_convolucao_{class_name}", "gif"))
        imageio.mimsave(gif_filename, frames, duration=0.5)
        print(f"✅ Animação salva como '{gif_filename}'")


if __name__ == "__main__":
    main()