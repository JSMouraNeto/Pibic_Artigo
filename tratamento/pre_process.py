import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# --------------------------------------------------------------------------
# --- CONFIGURAÇÃO ---
# --------------------------------------------------------------------------
class Config:
    """
    Classe para centralizar as configurações do script.
    """
    # Caminho para o banco de dados original.
    # Ex: "/home/user/datasets/dataset_bruto"
    INPUT_DIR = "/home/mob4ai-001/Pibic_Artigo/data/bigger_base"

    # Caminho onde o novo banco de dados tratado será salvo.
    # Ex: "/home/user/datasets/dataset_normalizado"
    OUTPUT_DIR = "preprocessado_big_base"

    # Tamanho final da imagem (quadrada). Todas as imagens serão
    # redimensionadas para caber dentro de um canvas deste tamanho.
    IMAGE_SIZE = 256

    # Qualidade da compressão JPEG para as imagens salvas (0 a 100).
    # Um valor maior resulta em maior qualidade e maior tamanho de arquivo.
    JPEG_QUALITY = 95

    # Número de processos paralelos a serem usados.
    # Usar 'os.cpu_count()' utiliza todos os núcleos disponíveis.
    # Recomenda-se 'os.cpu_count() - 1' para manter o sistema responsivo.
    NUM_WORKERS = max(1, os.cpu_count() - 1)

# --------------------------------------------------------------------------
# --- FUNÇÃO PRINCIPAL DE PROCESSAMENTO ---
# --------------------------------------------------------------------------
def processar_imagem(args):
    """
    Processa uma única imagem: lê, redimensiona com padding e salva no destino.

    Esta função foi projetada para ser executada em um processo separado,
    recebendo todos os seus argumentos em uma única tupla para compatibilidade
    com 'ProcessPoolExecutor.map'.

    Args:
        args (tuple): Uma tupla contendo (caminho_entrada, caminho_saida, tamanho_alvo, qualidade_jpeg).

    Returns:
        tuple: Um par (caminho_saida, status), onde status é True para sucesso
               e False para falha.
    """
    caminho_entrada, caminho_saida, tamanho_alvo, qualidade_jpeg = args
    try:
        # 1. Leitura da Imagem
        # Usamos cv2.imread para carregar a imagem. O OpenCV (cv2) é altamente otimizado
        # para operações de imagem e geralmente mais rápido que a PIL para este tipo de tarefa.
        # A imagem é carregada no formato BGR por padrão.
        imagem = cv2.imread(caminho_entrada)

        # Verifica se a imagem foi carregada corretamente.
        if imagem is None:
            # Pula arquivos corrompidos ou não suportados.
            return (caminho_entrada, False, "Não foi possível ler a imagem")

        # 2. Redimensionamento Proporcional com Padding (Letterboxing)
        altura_original, largura_original = imagem.shape[:2]
        
        # Calcula a proporção de redimensionamento e o novo tamanho mantendo a proporção.
        # A imagem será redimensionada para que sua maior dimensão se ajuste ao tamanho_alvo.
        escala = tamanho_alvo / max(altura_original, largura_original)
        nova_largura = int(largura_original * escala)
        nova_altura = int(altura_original * escala)
        
        # Redimensiona a imagem usando a interpolação cv2.INTER_AREA,
        # que é recomendada para diminuir imagens, pois evita artefatos.
        imagem_redimensionada = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_AREA)

        # 3. Criação do Canvas e Adição de Padding
        # Cria um canvas quadrado preto (valor de pixel 0) do tamanho alvo.
        # O formato é (altura, largura, canais de cor).
        canvas = np.zeros((tamanho_alvo, tamanho_alvo, 3), dtype=np.uint8)

        # Calcula as coordenadas para centralizar a imagem redimensionada no canvas.
        y_offset = (tamanho_alvo - nova_altura) // 2
        x_offset = (tamanho_alvo - nova_largura) // 2

        # "Cola" a imagem redimensionada no centro do canvas.
        canvas[y_offset:y_offset + nova_altura, x_offset:x_offset + nova_largura] = imagem_redimensionada

        # 4. Salvamento da Imagem
        # Garante que o diretório de destino exista antes de salvar.
        os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
        
        # Salva a imagem final em formato JPEG com a qualidade especificada.
        cv2.imwrite(caminho_saida, canvas, [int(cv2.IMWRITE_JPEG_QUALITY), qualidade_jpeg])

        return (caminho_entrada, True, None)

    except Exception as e:
        # Captura qualquer erro inesperado durante o processamento.
        return (caminho_entrada, False, str(e))

# --------------------------------------------------------------------------
# --- SCRIPT PRINCIPAL ---
# --------------------------------------------------------------------------
def main():
    """
    Orquestra todo o processo de normalização do banco de dados.
    """
    print("=" * 80)
    print("🚀 INICIANDO SCRIPT DE NORMALIZAÇÃO DE PROPORÇÃO DE IMAGENS 🚀")
    print("=" * 80)

    cfg = Config()

    # --- Validação dos Caminhos ---
    if not os.path.isdir(cfg.INPUT_DIR):
        print(f"❌ ERRO: O diretório de entrada não existe: '{cfg.INPUT_DIR}'")
        return

    if os.path.exists(cfg.OUTPUT_DIR) and os.listdir(cfg.OUTPUT_DIR):
        print(f"⚠️ AVISO: O diretório de saída '{cfg.OUTPUT_DIR}' já existe e não está vazio.")
        resposta = input("Deseja continuar e potencialmente sobrescrever arquivos? (s/n): ").lower()
        if resposta != 's':
            print("Operação cancelada pelo usuário.")
            return
            
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    print(f"📁 Lendo de: {cfg.INPUT_DIR}")
    print(f"💾 Salvando em: {cfg.OUTPUT_DIR}")
    print(f"🖼️  Tamanho Alvo: {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE} pixels")
    print(f"⚙️  Trabalhadores Paralelos: {cfg.NUM_WORKERS}")
    print("-" * 80)

    # --- Mapeamento de Arquivos ---
    print("🔍 Mapeando arquivos de imagem no diretório de entrada...")
    tarefas = []
    for root, _, files in os.walk(cfg.INPUT_DIR):
        for nome_arquivo in files:
            # Verifica se o arquivo tem uma extensão de imagem comum.
            if nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                caminho_entrada = os.path.join(root, nome_arquivo)
                
                # Cria a estrutura de diretórios correspondente no destino.
                # 'os.path.relpath' calcula o caminho relativo (ex: 'classe_A/imagem.jpg').
                caminho_relativo = os.path.relpath(caminho_entrada, cfg.INPUT_DIR)
                caminho_saida = os.path.join(cfg.OUTPUT_DIR, caminho_relativo)
                
                # Adiciona a tarefa à lista.
                tarefas.append((caminho_entrada, caminho_saida, cfg.IMAGE_SIZE, cfg.JPEG_QUALITY))

    if not tarefas:
        print("❌ ERRO: Nenhuma imagem encontrada no diretório de entrada.")
        return

    print(f"✅ Mapeamento concluído. {len(tarefas)} imagens encontradas para processar.")

    # --- Processamento Paralelo ---
    print("\n🔄 Iniciando processamento paralelo das imagens...")
    sucessos = 0
    falhas = 0
    
    # ProcessPoolExecutor gerencia um pool de processos de trabalho.
    # Isso contorna a limitação do GIL (Global Interpreter Lock) do Python,
    # permitindo o verdadeiro paralelismo em máquinas com múltiplos núcleos.
    with ProcessPoolExecutor(max_workers=cfg.NUM_WORKERS) as executor:
        # 'tqdm' cria uma barra de progresso para acompanhar o andamento.
        # 'as_completed' retorna os futuros à medida que são concluídos,
        # o que é mais eficiente em termos de memória do que esperar que todos terminem.
        future_to_task = {executor.submit(processar_imagem, tarefa): tarefa for tarefa in tarefas}
        
        for future in tqdm(as_completed(future_to_task), total=len(tarefas), desc="Processando"):
            try:
                _, sucesso, erro_msg = future.result()
                if sucesso:
                    sucessos += 1
                else:
                    falhas += 1
                    # Opcional: registrar falhas em um arquivo de log
                    # print(f"\nFalha ao processar {caminho}: {erro_msg}")
            except Exception as e:
                falhas += 1
                # print(f"\nErro no worker para a tarefa {future_to_task[future][0]}: {e}")

    # --- Resumo Final ---
    print("\n" + "=" * 80)
    print("🎉 Processamento Concluído! 🎉")
    print(f"✅ Imagens processadas com sucesso: {sucessos}")
    print(f"❌ Imagens com falha: {falhas}")
    print(f"✨ O novo banco de dados normalizado está pronto em: '{cfg.OUTPUT_DIR}'")
    print("=" * 80)


if __name__ == "__main__":
    main()