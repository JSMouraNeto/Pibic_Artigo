import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2 # Usado para o histograma de cores

# --- 1. Configuração Inicial ---
# Defina o caminho para a pasta principal do seu dataset
DATASET_PATH = "/home/mob4ai-001/Pibic_Artigo/dataset_analisado_vfinal/dataset_deduplicado"

# Verifica se o caminho do dataset existe
if not os.path.exists(DATASET_PATH):
    print(f"ERRO: O diretório '{DATASET_PATH}' não foi encontrado.")
    print("Por favor, atualize a variável 'DATASET_PATH' com o caminho correto.")
else:
    print("Diretório do dataset encontrado com sucesso!")

# --- 2. Carregando os Metadados das Imagens ---
# Vamos percorrer as pastas e coletar o caminho de cada imagem e sua respectiva classe.

def carregar_dados_do_dataset(path):
    """
    Percorre as pastas do dataset e cria um DataFrame do Pandas
    com o caminho da imagem e o nome da classe.
    """
    data = []
    # Lista os diretórios (classes) dentro do caminho do dataset
    classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    for classe_nome in classes:
        classe_path = os.path.join(path, classe_nome)
        for img_nome in os.listdir(classe_path):
            img_path = os.path.join(classe_path, img_nome)
            # Adiciona as informações à lista
            data.append({'caminho': img_path, 'classe': classe_nome})
            
    return pd.DataFrame(data)

# Cria o DataFrame
if os.path.exists(DATASET_PATH):
    df_imagens = carregar_dados_do_dataset(DATASET_PATH)
    print(f"\nTotal de imagens encontradas: {len(df_imagens)}")
    print("Amostra do DataFrame criado:")
    print(df_imagens.head())


# --- 3. Análise Exploratória Geral do Dataset ---

def analise_geral(df):
    """
    Realiza a análise da distribuição das classes no dataset completo.
    """
    if df.empty:
        print("\nDataFrame vazio. Pulando análise geral.")
        return

    print("\n--- ANÁLISE GERAL DO DATASET ---")
    
    # Contagem de imagens por classe
    print("\n1. Distribuição das Classes:")
    contagem_classes = df['classe'].value_counts()
    print(contagem_classes)
    
    # Gráfico de barras da distribuição
    plt.figure(figsize=(10, 6))
    sns.barplot(x=contagem_classes.index, y=contagem_classes.values, palette='viridis')
    plt.title('Distribuição de Imagens por Classe')
    plt.xlabel('Classe')
    plt.ylabel('Número de Imagens')
    plt.xticks(rotation=45)
    plt.show()

# Executa a análise geral
if 'df_imagens' in locals():
    analise_geral(df_imagens)


# --- 4. Análise Individual de Cada Classe ---

def analise_por_classe(df):
    """
    Realiza análises visuais e de metadados para cada classe individualmente.
    """
    if df.empty:
        print("\nDataFrame vazio. Pulando análise por classe.")
        return
        
    print("\n\n--- ANÁLISE INDIVIDUAL POR CLASSE ---")
    classes = df['classe'].unique()
    
    # 1. Visualização de Amostras de Imagens
    print("\n1. Exibindo amostras de imagens de cada classe...")
    fig, axes = plt.subplots(len(classes), 5, figsize=(15, 2 * len(classes)))
    fig.suptitle('Amostras de Imagens por Classe', fontsize=16)

    for i, classe in enumerate(classes):
        imagens_da_classe = df[df['classe'] == classe]['caminho'].tolist()
        amostras = random.sample(imagens_da_classe, min(len(imagens_da_classe), 5))
        
        for j, img_path in enumerate(amostras):
            ax = axes[i, j]
            try:
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(classe if j == 2 else "") # Título no meio
                ax.axis('off')
            except Exception as e:
                print(f"Não foi possível ler a imagem {img_path}: {e}")
                ax.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 2. Análise das Dimensões (Altura e Largura)
    print("\n2. Analisando as dimensões das imagens (altura x largura)...")
    if 'largura' not in df.columns or 'altura' not in df.columns:
        dimensoes = [Image.open(p).size for p in df['caminho']]
        df['largura'] = [d[0] for d in dimensoes]
        df['altura'] = [d[1] for d in dimensoes]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='largura', y='altura', hue='classe', alpha=0.7, palette='deep')
    plt.title('Distribuição das Dimensões das Imagens por Classe')
    plt.xlabel('Largura (pixels)')
    plt.ylabel('Altura (pixels)')
    plt.grid(True)
    plt.show()
    
    print("\nEstatísticas descritivas das dimensões:")
    print(df.groupby('classe')[['largura', 'altura']].describe())

    # 3. Análise da Distribuição de Cores (Histograma)
    print("\n3. Analisando a distribuição de cores (histograma de pixel)...")
    
    fig, axes = plt.subplots(len(classes), 1, figsize=(10, 5 * len(classes)))
    if len(classes) == 1: axes = [axes] # Garante que `axes` seja iterável
    fig.suptitle('Histograma de Cores Médio por Classe', fontsize=16)
    
    cores = ('b', 'g', 'r') # OpenCV usa BGR

    for i, classe in enumerate(classes):
        amostras = df[df['classe'] == classe]['caminho'].sample(min(len(df[df['classe'] == classe]), 20)) # Pega até 20 amostras
        
        hist_total = np.zeros((256, 3))
        
        for img_path in amostras:
            try:
                img = cv2.imread(img_path)
                for j, cor in enumerate(cores):
                    hist = cv2.calcHist([img], [j], None, [256], [0, 256])
                    hist_total[:, j] += hist.flatten()
            except Exception:
                continue # Pula imagens corrompidas

        # Normaliza o histograma
        hist_total /= len(amostras)
        
        ax = axes[i]
        for j, cor in enumerate(cores):
            ax.plot(hist_total[:, j], color=cor, label=f'Canal {cor.upper()}')
        ax.set_title(f'Classe: {classe}')
        ax.set_xlabel('Intensidade do Pixel')
        ax.set_ylabel('Frequência Média')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Executa a análise por classe
if 'df_imagens' in locals():
    analise_por_classe(df_imagens)