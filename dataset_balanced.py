import os
import shutil
import random
from collections import Counter
from pathlib import Path
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetBalancer:
    def __init__(self, source_dir, target_dir="raw_balanced"):
        """
        Inicializa o balanceador de dataset.
        
        Args:
            source_dir (str): Diretório do dataset original
            target_dir (str): Diretório onde será salvo o dataset balanceado
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        
        # Verificar se o diretório fonte existe
        if not self.source_dir.exists():
            raise ValueError(f"Diretório fonte não encontrado: {self.source_dir}")
        
        # Extensões de imagem suportadas
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
    def analyze_dataset(self):
        """
        Analisa o dataset e retorna estatísticas das classes.
        
        Returns:
            dict: Dicionário com estatísticas das classes
        """
        logger.info(f"Analisando dataset em: {self.source_dir}")
        
        class_stats = {}
        
        # Iterar sobre cada diretório (classe)
        for class_dir in self.source_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            
            # Contar imagens válidas
            image_files = []
            for file in class_dir.iterdir():
                if file.is_file() and file.suffix.lower() in self.image_extensions:
                    image_files.append(file)
            
            class_stats[class_name] = {
                'count': len(image_files),
                'files': image_files
            }
            
            logger.info(f"Classe '{class_name}': {len(image_files)} imagens")
        
        if not class_stats:
            raise ValueError("Nenhuma classe encontrada no diretório fonte")
        
        return class_stats
    
    def find_min_class_size(self, class_stats):
        """
        Encontra a classe com menor quantidade de imagens.
        
        Args:
            class_stats (dict): Estatísticas das classes
            
        Returns:
            tuple: (nome_da_classe_menor, quantidade_minima)
        """
        min_count = float('inf')
        min_class = None
        
        for class_name, stats in class_stats.items():
            if stats['count'] < min_count:
                min_count = stats['count']
                min_class = class_name
        
        logger.info(f"Classe com menor quantidade: '{min_class}' com {min_count} imagens")
        return min_class, min_count
    
    def create_balanced_dataset(self, random_seed=42):
        """
        Cria o dataset balanceado.
        
        Args:
            random_seed (int): Seed para reprodutibilidade
        """
        # Definir seed para reprodutibilidade
        random.seed(random_seed)
        
        # Analisar dataset original
        class_stats = self.analyze_dataset()
        
        # Encontrar tamanho mínimo
        min_class, min_count = self.find_min_class_size(class_stats)
        
        if min_count == 0:
            raise ValueError(f"A classe '{min_class}' não possui imagens válidas")
        
        logger.info(f"Criando dataset balanceado com {min_count} imagens por classe")
        
        # Remover diretório de destino se existir
        if self.target_dir.exists():
            logger.info(f"Removendo diretório existente: {self.target_dir}")
            shutil.rmtree(self.target_dir)
        
        # Criar diretório de destino
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        total_images_copied = 0
        
        # Processar cada classe
        for class_name, stats in class_stats.items():
            class_target_dir = self.target_dir / class_name
            class_target_dir.mkdir(exist_ok=True)
            
            available_files = stats['files']
            images_to_copy = min_count
            
            if len(available_files) < images_to_copy:
                logger.warning(f"Classe '{class_name}' tem apenas {len(available_files)} imagens, "
                             f"mas precisamos de {images_to_copy}")
                images_to_copy = len(available_files)
            
            # Selecionar imagens aleatoriamente
            selected_files = random.sample(available_files, images_to_copy)
            
            # Copiar imagens selecionadas
            for i, source_file in enumerate(selected_files):
                target_file = class_target_dir / source_file.name
                
                # Se já existe arquivo com mesmo nome, renomear
                if target_file.exists():
                    stem = source_file.stem
                    suffix = source_file.suffix
                    target_file = class_target_dir / f"{stem}_{i}{suffix}"
                
                shutil.copy2(source_file, target_file)
            
            total_images_copied += len(selected_files)
            logger.info(f"Classe '{class_name}': {len(selected_files)} imagens copiadas")
        
        logger.info(f"\nDataset balanceado criado com sucesso!")
        logger.info(f"Diretório: {self.target_dir}")
        logger.info(f"Total de classes: {len(class_stats)}")
        logger.info(f"Imagens por classe: {min_count}")
        logger.info(f"Total de imagens: {total_images_copied}")
        
        return self.get_balanced_stats()
    
    def get_balanced_stats(self):
        """
        Retorna estatísticas do dataset balanceado.
        
        Returns:
            dict: Estatísticas do dataset balanceado
        """
        if not self.target_dir.exists():
            return None
        
        balanced_stats = {}
        total_images = 0
        
        for class_dir in self.target_dir.iterdir():
            if not class_dir.is_dir():
                continue
                
            class_name = class_dir.name
            image_count = len([f for f in class_dir.iterdir() 
                             if f.is_file() and f.suffix.lower() in self.image_extensions])
            
            balanced_stats[class_name] = image_count
            total_images += image_count
        
        return {
            'classes': balanced_stats,
            'total_images': total_images,
            'num_classes': len(balanced_stats)
        }
    
    def print_comparison(self):
        """
        Imprime comparação entre dataset original e balanceado.
        """
        # Estatísticas do dataset original
        original_stats = self.analyze_dataset()
        
        # Estatísticas do dataset balanceado
        balanced_stats = self.get_balanced_stats()
        
        if not balanced_stats:
            logger.error("Dataset balanceado não encontrado")
            return
        
        print("\n" + "="*60)
        print("COMPARAÇÃO: ORIGINAL vs BALANCEADO")
        print("="*60)
        
        print(f"{'Classe':<20} {'Original':<15} {'Balanceado':<15} {'Redução':<15}")
        print("-"*60)
        
        total_original = 0
        total_balanced = 0
        
        for class_name in sorted(original_stats.keys()):
            orig_count = original_stats[class_name]['count']
            balanced_count = balanced_stats['classes'].get(class_name, 0)
            reduction = orig_count - balanced_count if orig_count > 0 else 0
            
            print(f"{class_name:<20} {orig_count:<15} {balanced_count:<15} {reduction:<15}")
            
            total_original += orig_count
            total_balanced += balanced_count
        
        print("-"*60)
        print(f"{'TOTAL':<20} {total_original:<15} {total_balanced:<15} {total_original - total_balanced:<15}")
        
        reduction_percentage = ((total_original - total_balanced) / total_original * 100) if total_original > 0 else 0
        print(f"\nRedução total: {reduction_percentage:.1f}%")


def main():
    """
    Função principal para executar o balanceamento.
    """
    # Configurar caminho do dataset original
    SOURCE_DIR = "data/base_gray_224_padded"  # Atualize com seu caminho
    
    try:
        # Criar balanceador
        balancer = DatasetBalancer(SOURCE_DIR, "base_gray_224_padded_balanced")
        
        # Criar dataset balanceado
        balanced_stats = balancer.create_balanced_dataset(random_seed=42)
        
        # Mostrar comparação
        balancer.print_comparison()
        
        print(f"\n✅ Dataset balanceado criado com sucesso em: raw_balanced/")
        print(f"📊 {balanced_stats['num_classes']} classes com distribuição uniforme")
        print(f"🖼️  Total de imagens: {balanced_stats['total_images']}")
        
    except Exception as e:
        logger.error(f"Erro durante o balanceamento: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main()