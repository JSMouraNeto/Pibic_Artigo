import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import time


INPUT_DIR = Path("dataset_original")
OUTPUT_DIR = Path("dataset_processado")

# 2. Parâmetros de processamento de imagem
IMAGE_SIZE = (224, 224)  # Tamanho final da imagem (altura, largura)
USE_CLAHE = True         # Mude para False se não quiser usar a melhora de contraste

# ---------------------

def process_image(image_path: Path, output_path: Path):
    """
    Aplica o pipeline de pré-processamento a uma única imagem.
    
    Args:
        image_path (Path): Caminho para a imagem original.
        output_path (Path): Caminho onde a imagem processada será salva.
    """
    try:
        # 1. Ler a imagem diretamente em escala de cinza para otimizar
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        # Checa se a imagem foi carregada corretamente
        if image is None:
            print(f"Aviso: Não foi possível ler a imagem {image_path}. Pulando.")
            return

        # 2. (Opcional) Aplicar CLAHE para melhorar o contraste local
        if USE_CLAHE:
            # clipLimit: Limite para o contraste. Valores pequenos (2-3) são bons.
            # tileGridSize: Tamanho da janela para a equalização local.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

        # 3. Redimensionar a imagem para o tamanho alvo
        # cv2.INTER_AREA é eficiente para reduzir o tamanho da imagem (downsampling)
        image_resized = cv2.resize(image, (IMAGE_SIZE[1], IMAGE_SIZE[0]), interpolation=cv2.INTER_AREA)

        # 4. Salvar a imagem processada no diretório de saída
        # O formato PNG é usado por ser sem perdas
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image_resized)
        
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")

def main():
    """
    Função principal que orquestra o pré-processamento de todo o dataset.
    """
    print("Iniciando o pipeline de pré-processamento...")
    start_time = time.time()

    # Validação do diretório de entrada
    if not INPUT_DIR.exists():
        print(f"Erro: O diretório de entrada '{INPUT_DIR}' não foi encontrado.")
        return

    # Criar diretório de saída se não existir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Dataset original em: '{INPUT_DIR}'")
    print(f"Dataset processado será salvo em: '{OUTPUT_DIR}'")

    # Coletar todas as tarefas de processamento de imagem
    tasks = []
    print("Mapeando imagens para processamento...")
    
    # Encontra todos os arquivos de imagem comuns nos subdiretórios
    image_paths = list(INPUT_DIR.glob('**/*.png')) + \
                  list(INPUT_DIR.glob('**/*.jpg')) + \
                  list(INPUT_DIR.glob('**/*.jpeg'))

    for img_path in image_paths:
        # Mantém a estrutura de subpastas (classes)
        relative_path = img_path.relative_to(INPUT_DIR)
        output_image_path = OUTPUT_DIR / relative_path
        tasks.append((img_path, output_image_path))

    if not tasks:
        print("Nenhuma imagem encontrada para processar. Verifique o INPUT_DIR.")
        return
        
    print(f"Total de {len(tasks)} imagens encontradas.")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(lambda p: process_image(*p), tasks), total=len(tasks), desc="Processando Imagens"))
    
    end_time = time.time()
    print("\nPipeline de pré-processamento concluído!")
    print(f"Tempo total de execução: {end_time - start_time:.2f} segundos.")


if __name__ == "__main__":
    main()