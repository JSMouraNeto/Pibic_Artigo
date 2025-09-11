# -*- coding: utf-8 -*-

import os
import cv2
import torch
import numpy as np
import pydicom
from pathlib import Path
from tqdm import tqdm
import argparse
import requests
import logging
import json
from datetime import datetime
from typing import Tuple, Optional, List, Dict
import warnings

warnings.filterwarnings("ignore")

# Dependências de processamento de imagem e métricas
from scipy import ndimage
from scipy.signal import find_peaks
from skimage import measure, morphology
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

# Importa as ferramentas necessárias do SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("ERRO: segment_anything não instalado. Execute: pip install git+https://github.com/facebookresearch/segment-anything.git")
    exit(1)

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('segmentation_improved.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def dice_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o Coeficiente de Similaridade Dice para máscaras binárias.
    É robusto a divisões por zero.
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    intersection = np.sum(y_true & y_pred)
    sum_masks = np.sum(y_true) + np.sum(y_pred)
    if sum_masks == 0:
        return 1.0
    return (2. * intersection) / (sum_masks + 1e-8)


class ImagePreprocessor:
    """Classe para pré-processar imagens médicas, com foco em DICOM."""

    def __init__(self):
        self.lung_window = {'center': -600, 'width': 1500}
        self.mediastinum_window = {'center': 50, 'width': 350}

    @staticmethod
    def normalize_medical_image(image: np.ndarray, window_center: int, window_width: int) -> np.ndarray:
        """
        Normaliza a imagem para 8 bits usando uma janela de visualização (WC/WW).
        O intervalo é cortado (clipped) e depois reescalado para 0-255.
        """
        min_val = window_center - window_width / 2
        max_val = window_center + window_width / 2
        image = np.clip(image, min_val, max_val)

        # Evita divisão por zero
        if max_val == min_val:
            return np.zeros_like(image, dtype=np.uint8)

        image = ((image - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
        return image

    def enhance_contrast(self, image: np.ndarray, method: str = 'clahe') -> np.ndarray:
        """Aplica diferentes métodos de realce de contraste."""
        if method == 'clahe':
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        elif method == 'histogram_eq':
            return cv2.equalizeHist(image)
        elif method == 'adaptive':
            # Combina CLAHE com filtro gaussiano
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            gaussian = cv2.GaussianBlur(enhanced, (3, 3), 0)
            return cv2.addWeighted(enhanced, 0.7, gaussian, 0.3, 0)
        else:
            return image

    def load_dicom_image(self, path: Path) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Carrega uma imagem DICOM, aplica a conversão para Hounsfield Units (HU)
        e extrai metadados relevantes com validações adicionais.
        """
        try:
            ds = pydicom.dcmread(str(path))
            image = ds.pixel_array.astype(np.float64)

            # Validação básica da imagem
            if image.size == 0:
                raise ValueError("Imagem DICOM vazia")

            # Converte para Hounsfield Units (HU)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                image = image * slope + intercept

            # Extrai metadados essenciais com valores padrão seguros
            metadata = {
                'patient_id': str(getattr(ds, 'PatientID', 'Unknown')),
                'study_date': str(getattr(ds, 'StudyDate', 'Unknown')),
                'modality': str(getattr(ds, 'Modality', 'Unknown')),
                'image_shape': image.shape,
                'pixel_spacing': getattr(ds, 'PixelSpacing', [1.0, 1.0]),
                'slice_thickness': float(getattr(ds, 'SliceThickness', 1.0))
            }

            # Determina janela de visualização baseada na modalidade
            if metadata['modality'] == 'CT':
                metadata['window_center'] = int(getattr(ds, 'WindowCenter', self.lung_window['center']))
                metadata['window_width'] = int(getattr(ds, 'WindowWidth', self.lung_window['width']))
            else:
                # Para outras modalidades, usa valores padrão
                metadata['window_center'] = self.lung_window['center']
                metadata['window_width'] = self.lung_window['width']

            return image, metadata

        except Exception as e:
            logger.error(f"Erro ao carregar DICOM {path}: {e}")
            return None, {}

    def load_standard_image(self, path: Path) -> Tuple[Optional[np.ndarray], Dict]:
        """Carrega imagens em formatos padrão (PNG, JPG, etc.)."""
        try:
            image_gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image_gray is None:
                raise ValueError("Falha ao carregar imagem")

            # Simula HU para imagens não-DICOM (aproximação grosseira)
            image_hu = (image_gray.astype(np.float64) - 128) * 8 - 400

            metadata = {
                'patient_id': path.stem,
                'study_date': datetime.now().strftime('%Y%m%d'),
                'modality': 'Unknown',
                'window_center': self.lung_window['center'],
                'window_width': self.lung_window['width'],
                'image_shape': image_gray.shape,
                'pixel_spacing': [1.0, 1.0],
                'slice_thickness': 1.0
            }

            return image_hu, metadata

        except Exception as e:
            logger.error(f"Erro ao carregar imagem {path}: {e}")
            return None, {}


class LungSegmentationValidator:
    """Valida a máscara de segmentação com base em critérios anatômicos aprimorados."""

    def __init__(self):
        self.area_thresholds = {'min': 0.03, 'max': 0.75}
        self.component_thresholds = {'min': 1, 'max': 3}
        self.aspect_ratio_range = (0.3, 3.0)

    def validate_lung_mask(self, mask: np.ndarray, image_shape: Tuple[int, int],
                           metadata: Dict = None) -> Dict[str, any]:
        """Executa uma série de validações aprimoradas na máscara gerada."""
        total_pixels = image_shape[0] * image_shape[1]
        mask_pixels = np.sum(mask > 0)
        area_ratio = mask_pixels / total_pixels

        validation = {
            'area_ratio': area_ratio,
            'valid_area': self.area_thresholds['min'] <= area_ratio <= self.area_thresholds['max'],
            'connected_components': 0,
            'valid_components': False,
            'centroid_separation': False,
            'aspect_ratio_valid': False,
            'symmetry_score': 0.0,
            'compactness_score': 0.0,
            'overall_valid': False
        }

        if mask_pixels == 0:
            return validation

        # Análise de componentes conectados
        labeled_mask = measure.label(mask.astype(bool))
        components = measure.regionprops(labeled_mask)

        # Filtrar componentes muito pequenos
        min_component_size = total_pixels * 0.01  # Pelo menos 1% da imagem
        components = [c for c in components if c.area >= min_component_size]

        # Manter apenas os componentes mais relevantes
        if len(components) > 2:
            components.sort(key=lambda r: r.area, reverse=True)
            components = components[:2]

        num_components = len(components)
        validation['connected_components'] = num_components
        validation['valid_components'] = (
            self.component_thresholds['min'] <= num_components <= self.component_thresholds['max']
        )

        if num_components == 0:
            return validation

        # Validação de forma e geometria
        aspect_ratios = []
        for comp in components:
            minr, minc, maxr, maxc = comp.bbox
            height = maxr - minr
            width = maxc - minc
            if width > 0:
                aspect_ratio = height / width
                aspect_ratios.append(aspect_ratio)

        if aspect_ratios:
            avg_aspect_ratio = np.mean(aspect_ratios)
            validation['aspect_ratio_valid'] = (
                self.aspect_ratio_range[0] <= avg_aspect_ratio <= self.aspect_ratio_range[1]
            )

        # Validação de separação dos centroides para 2 componentes
        if num_components == 2:
            c1, c2 = components[0], components[1]
            c1_x, c1_y = c1.centroid[1], c1.centroid[0]
            c2_x, c2_y = c2.centroid[1], c2.centroid[0]

            # Verifica separação horizontal (pulmões em lados opostos)
            w, h = image_shape[1], image_shape[0]
            horizontal_sep = abs(c1_x - c2_x) > w * 0.2
            vertical_alignment = abs(c1_y - c2_y) < h * 0.3

            validation['centroid_separation'] = horizontal_sep and vertical_alignment

            # Calcula pontuação de simetria
            center_x = w / 2
            dist1 = abs(c1_x - center_x)
            dist2 = abs(c2_x - center_x)
            symmetry = 1.0 - abs(dist1 - dist2) / (center_x + 1e-8)
            validation['symmetry_score'] = max(0, symmetry)

        elif num_components == 1:
            # Para um componente, verifica se está centralizado
            comp = components[0]
            center_x, center_y = image_shape[1] / 2, image_shape[0] / 2
            comp_x, comp_y = comp.centroid[1], comp.centroid[0]

            # Permite centralização com alguma tolerância
            x_tolerance = image_shape[1] * 0.3
            y_tolerance = image_shape[0] * 0.2

            validation['centroid_separation'] = (
                abs(comp_x - center_x) <= x_tolerance and
                abs(comp_y - center_y) <= y_tolerance
            )
            validation['symmetry_score'] = 0.7  # Score neutro para componente único

        # Calcula compacidade (medida de quão "redondo" é o objeto)
        total_area = sum(comp.area for comp in components)
        total_perimeter = sum(comp.perimeter for comp in components)
        if total_perimeter > 0:
            compactness = (4 * np.pi * total_area) / (total_perimeter ** 2)
            validation['compactness_score'] = min(1.0, compactness)

        # Validação geral com pesos
        weights = {
            'area': 0.3,
            'components': 0.2,
            'separation': 0.2,
            'aspect_ratio': 0.15,
            'symmetry': 0.1,
            'compactness': 0.05
        }

        score = 0
        if validation['valid_area']: score += weights['area']
        if validation['valid_components']: score += weights['components']
        if validation['centroid_separation']: score += weights['separation']
        if validation['aspect_ratio_valid']: score += weights['aspect_ratio']
        score += weights['symmetry'] * validation['symmetry_score']
        score += weights['compactness'] * validation['compactness_score']

        validation['overall_valid'] = score >= 0.7  # Threshold mais flexível
        validation['quality_score'] = score

        return validation


def download_file(url: str, destination: Path, force_download: bool = False) -> bool:
    """Faz o download de um arquivo com barra de progresso e verificação de integridade."""
    if destination.exists() and not force_download:
        # Verifica se o arquivo baixado não está corrompido
        if destination.stat().st_size > 100_000_000:  # Arquivo deve ter pelo menos 100MB
            logger.info(f"Modelo já existe em: {destination}")
            return True
        else:
            logger.warning(f"Arquivo existente parece corrompido, baixando novamente...")

    logger.info(f"Baixando modelo de {url}...")
    try:
        # Headers para evitar bloqueios
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        with requests.get(url, stream=True, timeout=120, headers=headers) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))

            # Criar diretório se não existir
            destination.parent.mkdir(parents=True, exist_ok=True)

            with open(destination, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=destination.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # Filtrar chunks vazios
                        f.write(chunk)
                        pbar.update(len(chunk))

        # Verificação de integridade básica
        if destination.stat().st_size < 100_000_000:
             raise ValueError("Arquivo baixado está incompleto ou vazio")

        logger.info(f"Modelo salvo com sucesso em: {destination}")
        return True

    except Exception as e:
        logger.error(f"Falha no download: {e}")
        if destination.exists():
            destination.unlink()
        return False


class MedSAMSegmenter:
    """Classe principal que orquestra a segmentação pulmonar usando MedSAM."""

    def __init__(self, checkpoint_path: str, model_type: str = "vit_b", device: str = 'auto'):
        # Determina o melhor dispositivo disponível
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                # Otimizações para CUDA
                torch.backends.cudnn.benchmark = True
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')  # Apple Silicon
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Inicializando MedSAM no dispositivo: {self.device}")

        # Configurações específicas do dispositivo
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"VRAM disponível: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        try:
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)

            # Otimizações para inferência
            sam.eval()
            if self.device.type == 'cuda':
                sam = sam.half()  # Usar half precision para economizar VRAM

            self.predictor = SamPredictor(sam)
            logger.info("Modelo MedSAM carregado com sucesso.")

        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise

        self.preprocessor = ImagePreprocessor()
        self.validator = LungSegmentationValidator()
        self.stats = {
            'total': 0, 'success': 0, 'fail': 0, 'validation_fail': 0,
            'processing_times': [], 'quality_scores': []
        }

    def _generate_automatic_prompts(self, image_hu: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        """
        Gera prompts automáticos mais robustos usando múltiplas estratégias.
        """
        try:
            # Estratégia 1: Thresholding baseado em HU (principal)
            lung_mask = self._threshold_lungs_hu(image_hu)

            # Se a máscara principal falhar, tenta a estratégia de morfologia (backup)
            if np.sum(lung_mask) == 0:
                logger.warning("Estratégia primária (HU) falhou. Tentando fallback (morfologia).")
                lung_mask = self._detect_lungs_morphology(image_hu)
                if np.sum(lung_mask) == 0:
                    return None
            
            return self._extract_prompts_from_mask(lung_mask, image_hu.shape)

        except Exception as e:
            logger.error(f"Erro ao gerar prompts automáticos: {e}")
            return None

    def _threshold_lungs_hu(self, image_hu: np.ndarray) -> np.ndarray:
        """Segmentação baseada em valores de Hounsfield Units."""
        # Valores padrão clinicamente informados
        air_threshold = -950
        soft_tissue_threshold = -300

        # Cria máscara de pulmão (ar + parênquima)
        lung_mask = np.where(
            (image_hu > air_threshold) & (image_hu < soft_tissue_threshold), 1, 0
        ).astype(np.uint8)

        # Limpeza morfológica mais agressiva
        kernel = morphology.disk(5)
        lung_mask = morphology.binary_opening(lung_mask, kernel)
        lung_mask = morphology.binary_closing(lung_mask, morphology.disk(10))
        lung_mask = morphology.remove_small_objects(lung_mask.astype(bool), min_size=2000)

        return lung_mask.astype(np.uint8)

    def _detect_lungs_morphology(self, image_hu: np.ndarray) -> np.ndarray:
        """Detecção de pulmões usando operações morfológicas."""
        # Normaliza para processamento
        normalized = cv2.normalize(image_hu, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Filtro bilateral para suavizar mantendo bordas
        filtered = cv2.bilateralFilter(normalized, 9, 75, 75)

        # Threshold adaptativo
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # Operações morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

        return (closed / 255).astype(np.uint8)

    def _extract_prompts_from_mask(self, mask: np.ndarray, image_shape: Tuple) -> Dict[str, np.ndarray]:
        """Extrai prompts otimizados da máscara de pulmão."""
        
        # ======================= INÍCIO DA CORREÇÃO =======================
        # Garante que a máscara esteja em um formato de memória limpo, contíguo
        # e com um tipo de dado seguro antes de passar para a biblioteca C subjacente.
        # Isso evita falhas de segmentação (segmentation faults) em casos extremos.
        clean_mask = np.ascontiguousarray(mask, dtype=np.uint8)
        labeled_mask = measure.label(clean_mask)
        # ======================== FIM DA CORREÇÃO =========================
        
        regions = measure.regionprops(labeled_mask)

        if not regions:
            return None

        # Seleciona os componentes mais relevantes
        regions.sort(key=lambda r: r.area, reverse=True)
        main_regions = regions[:min(2, len(regions))]

        # Reconstrói máscara apenas com componentes principais
        main_mask = np.zeros_like(mask)
        points = []

        for region in main_regions:
            main_mask[labeled_mask == region.label] = 1

            # Encontra o centro de massa usando transformada de distância
            region_mask = (labeled_mask == region.label).astype(np.uint8)
            distance_map = ndimage.distance_transform_edt(region_mask)

            if np.max(distance_map) > 0:
                center_coords = np.unravel_index(np.argmax(distance_map), distance_map.shape)
                center_global = (center_coords[1], center_coords[0])  # (x, y)
                points.append(center_global)

        if not points:
            return None

        # Gera bounding box otimizado
        coords = np.where(main_mask > 0)
        if len(coords[0]) == 0:
            return None

        minr, maxr = np.min(coords[0]), np.max(coords[0])
        minc, maxc = np.min(coords[1]), np.max(coords[1])

        # Adiciona margem ao bounding box
        margin_r = max(5, int((maxr - minr) * 0.05))
        margin_c = max(5, int((maxc - minc) * 0.05))

        minr = max(0, minr - margin_r)
        maxr = min(image_shape[0], maxr + margin_r)
        minc = max(0, minc - margin_c)
        maxc = min(image_shape[1], maxc + margin_c)

        bbox = np.array([minc, minr, maxc, maxr])

        # Adiciona pontos negativos estratégicos
        neg_points = self._generate_negative_points(main_mask)

        if neg_points:
            all_points = points + neg_points
            labels = np.array([1] * len(points) + [0] * len(neg_points))
        else:
            all_points = points
            labels = np.array([1] * len(points))

        return {
            "box": bbox,
            "points": np.array(all_points),
            "labels": labels,
            "initial_mask": main_mask.astype(bool)
        }

    def _generate_negative_points(self, mask: np.ndarray) -> List:
        """Gera pontos negativos estratégicos para melhorar a segmentação."""
        neg_points = []
        h, w = mask.shape

        # Ponto no centro da imagem (mediastino)
        center_x, center_y = w // 2, h // 2
        if mask[center_y, center_x] == 0:
            neg_points.append([center_x, center_y])

        # Pontos nas bordas da imagem
        border_points = [
            [w // 4, 10], [3 * w // 4, 10],  # Topo
            [w // 4, h - 10], [3 * w // 4, h - 10],  # Base
            [10, h // 2], [w - 10, h // 2]  # Laterais
        ]

        for point in border_points:
            x, y = point
            if 0 <= x < w and 0 <= y < h and mask[y, x] == 0:
                neg_points.append(point)

        return neg_points[:3]  # Limita o número de pontos negativos

    def _postprocess_mask(self, mask: np.ndarray, original_shape: Tuple) -> np.ndarray:
        """Pós-processamento aprimorado da máscara de segmentação."""
        if np.sum(mask) == 0:
            return mask.astype(np.uint8)

        # 1. Preenchimento de buracos
        mask_filled = ndimage.binary_fill_holes(mask)

        # 2. Remoção de ruído
        mask_denoised = morphology.remove_small_objects(
            mask_filled, min_size=max(1000, int(np.prod(original_shape) * 0.001))
        )

        # 3. Suavização das bordas
        kernel = morphology.disk(3)
        mask_smoothed = morphology.binary_opening(mask_denoised, kernel)
        mask_smoothed = morphology.binary_closing(mask_smoothed, morphology.disk(7))

        # 4. Separação e seleção de componentes
        labeled_mask = measure.label(mask_smoothed)
        regions = measure.regionprops(labeled_mask)

        if not regions:
            return np.zeros_like(mask, dtype=np.uint8)

        # Mantém apenas os componentes mais relevantes (máximo 2 pulmões)
        regions.sort(key=lambda r: r.area, reverse=True)
        valid_regions = []

        for region in regions[:3]:  # Considera até 3 componentes
            # Filtros de validação
            area_valid = region.area >= np.prod(original_shape) * 0.01
            height = region.bbox[2] - region.bbox[0]
            width = region.bbox[3] - region.bbox[1]
            aspect_ratio = height / max(1, width)
            ratio_valid = 0.2 <= aspect_ratio <= 5.0

            if area_valid and ratio_valid:
                valid_regions.append(region)

            if len(valid_regions) >= 2:
                break

        # Reconstrói a máscara final
        final_mask = np.zeros_like(mask, dtype=np.uint8)
        for region in valid_regions:
            final_mask[labeled_mask == region.label] = 1

        return final_mask

    def segment_image(self, image_path: Path, save_debug: bool = False) -> Tuple[Optional[np.ndarray], Dict]:
        """Orquestra o processo completo de segmentação para uma única imagem."""
        start_time = datetime.now()
        result = {
            'success': False, 'validation': {}, 'error': None,
            'metadata': {}, 'processing_time': 0, 'iou_score': 0
        }

        try:
            # Carregamento inteligente baseado na extensão
            if image_path.suffix.lower() in ['.dcm', '.dicom']:
                image_hu, metadata = self.preprocessor.load_dicom_image(image_path)
            else:
                image_hu, metadata = self.preprocessor.load_standard_image(image_path)

            if image_hu is None:
                raise ValueError("Falha ao carregar imagem")

            result['metadata'] = metadata

            # Pré-processamento adaptativo
            processed_image = self.preprocessor.normalize_medical_image(
                image_hu,
                window_center=metadata['window_center'],
                window_width=metadata['window_width']
            )
            processed_image = self.preprocessor.enhance_contrast(processed_image, method='adaptive')
            image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

            # Geração de prompts com múltiplas estratégias
            prompts = self._generate_automatic_prompts(image_hu)
            if prompts is None:
                raise ValueError("Não foi possível gerar prompts automáticos")

            # Predição com MedSAM
            with torch.no_grad():
                self.predictor.set_image(image_rgb)
                masks, scores, logits = self.predictor.predict(
                    box=prompts['box'],
                    point_coords=prompts['points'],
                    point_labels=prompts['labels'],
                    multimask_output=True
                )

            # Seleção inteligente da melhor máscara
            best_mask, best_score = self._select_best_mask(masks, scores, prompts['initial_mask'])

            if best_mask is None:
                raise ValueError("Nenhuma máscara válida foi gerada pelo modelo")

            # Pós-processamento e validação
            refined_mask = self._postprocess_mask(best_mask, processed_image.shape)
            final_validation = self.validator.validate_lung_mask(
                refined_mask, processed_image.shape, metadata
            )

            # Aplicação da máscara na imagem original
            segmented_lung = cv2.bitwise_and(processed_image, processed_image, mask=refined_mask)

            # Salvamento de informações de debug se solicitado
            if save_debug:
                self._save_debug_info(
                    image_path, image_rgb, prompts, best_mask,
                    refined_mask, final_validation
                )

            # Atualização de estatísticas
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['success'] += 1
            self.stats['processing_times'].append(processing_time)
            self.stats['quality_scores'].append(final_validation.get('quality_score', 0))

            if not final_validation['overall_valid']:
                self.stats['validation_fail'] += 1
                logger.warning(
                    f"Validação falhou para {image_path.name}: Score = {final_validation.get('quality_score', 0):.2f}")

            result.update({
                'success': True,
                'validation': final_validation,
                'iou_score': best_score,
                'processing_time': processing_time
            })

            return segmented_lung, result

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.stats['fail'] += 1
            error_msg = f"Erro ao segmentar {image_path.name}: {e}"
            logger.error(error_msg, exc_info=True)
            result.update({
                'error': str(e),
                'processing_time': processing_time
            })
            return None, result

    def _select_best_mask(self, masks: np.ndarray, scores: np.ndarray,
                          initial_mask: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Seleciona a melhor máscara usando múltiplos critérios."""
        if len(masks) == 0:
            return None, 0.0

        best_mask, best_combined_score = None, -1

        for i, mask in enumerate(masks):
            # Calcula IoU com a máscara inicial
            iou = jaccard_score(initial_mask.flatten(), mask.flatten())

            # Calcula Dice score
            dice = dice_score(initial_mask.flatten(), mask.flatten())

            # Score do modelo SAM
            model_score = scores[i] if i < len(scores) else 0.5

            # Score combinado (IoU tem maior peso)
            combined_score = 0.5 * iou + 0.3 * dice + 0.2 * model_score

            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_mask = mask

        return best_mask, best_combined_score

    def _save_debug_info(self, img_path: Path, img_rgb: np.ndarray,
                         prompts: Dict, raw_mask: np.ndarray,
                         final_mask: np.ndarray, validation: Dict):
        """Salva imagens de debug detalhadas para análise."""
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Debug: {img_path.name} | Quality Score: {validation.get('quality_score', 0):.2f}", fontsize=16)

        # Imagem Original com Prompts
        axes[0, 0].imshow(img_rgb)
        box = prompts['box']
        rect = plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                             fill=False, edgecolor='green', linewidth=2)
        axes[0, 0].add_patch(rect)

        pos_points = prompts['points'][prompts['labels'] == 1]
        neg_points = prompts['points'][prompts['labels'] == 0]

        if len(pos_points) > 0:
            axes[0, 0].scatter(pos_points[:, 0], pos_points[:, 1],
                               color='green', marker='*', s=150, label='Positive')
        if len(neg_points) > 0:
            axes[0, 0].scatter(neg_points[:, 0], neg_points[:, 1],
                               color='red', marker='*', s=150, label='Negative')

        axes[0, 0].set_title("Original + Prompts")
        axes[0, 0].axis('off')
        axes[0, 0].legend()

        # Máscara Inicial (Threshold)
        axes[0, 1].imshow(prompts['initial_mask'], cmap='gray')
        axes[0, 1].set_title("Máscara Inicial")
        axes[0, 1].axis('off')

        # Máscara Bruta do MedSAM
        axes[0, 2].imshow(raw_mask, cmap='gray')
        axes[0, 2].set_title("Máscara Bruta (MedSAM)")
        axes[0, 2].axis('off')

        # Máscara Final (Pós-processada)
        axes[1, 0].imshow(final_mask, cmap='gray')
        axes[1, 0].set_title("Máscara Final")
        axes[1, 0].axis('off')

        # Sobreposição da máscara na imagem original
        overlay = img_rgb.copy()
        overlay[final_mask > 0] = [255, 0, 0]  # Vermelho para área segmentada
        blended = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
        axes[1, 1].imshow(blended)
        axes[1, 1].set_title("Sobreposição")
        axes[1, 1].axis('off')

        # Gráfico de validação
        validation_metrics = {
            'Area Ratio': validation.get('area_ratio', 0),
            'Symmetry': validation.get('symmetry_score', 0),
            'Compactness': validation.get('compactness_score', 0),
            'Quality Score': validation.get('quality_score', 0)
        }

        bars = axes[1, 2].bar(validation_metrics.keys(), validation_metrics.values())
        axes[1, 2].set_title("Métricas de Validação")
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].tick_params(axis='x', rotation=45)

        # Colorir barras baseado na qualidade
        for bar, value in zip(bars, validation_metrics.values()):
            if value > 0.7:
                bar.set_color('green')
            elif value > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        plt.tight_layout()
        plt.savefig(debug_dir / f"{img_path.stem}_debug.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Salva também um log JSON detalhado
        debug_info = {
            'image_path': str(img_path),
            'validation': validation,
            'prompts_info': {
                'num_positive_points': int(np.sum(prompts['labels'] == 1)),
                'num_negative_points': int(np.sum(prompts['labels'] == 0)),
                'bbox_area': int((prompts['box'][2] - prompts['box'][0]) * (prompts['box'][3] - prompts['box'][1]))
            },
            'timestamp': datetime.now().isoformat()
        }

        with open(debug_dir / f"{img_path.stem}_debug.json", 'w') as f:
            json.dump(debug_info, f, indent=2)

    def process_dataset(self, input_dir: Path, output_dir: Path,
                        save_debug: bool = False) -> Dict:
        """Processa um diretório completo de imagens de forma sequencial."""
        logger.info(f"Processando dataset: {input_dir} -> {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Busca arquivos de imagem de forma recursiva
        supported_formats = ['.png', '.jpg', '.jpeg', '.dcm', '.dicom', '.tif', '.tiff']
        image_files = []
        for fmt in supported_formats:
            image_files.extend(list(input_dir.rglob(f'*{fmt}')))
            image_files.extend(list(input_dir.rglob(f'*{fmt.upper()}')))

        # Remove duplicatas mantendo a ordem
        image_files = sorted(list(dict.fromkeys(image_files)))

        if not image_files:
            logger.error(f"Nenhuma imagem encontrada em {input_dir}")
            return {}

        logger.info(f"Encontradas {len(image_files)} imagens.")

        # Inicializa estatísticas
        self.stats = {k: 0 if not isinstance(v, list) else [] for k, v in self.stats.items()}
        self.stats['total'] = len(image_files)
        results_log = []

        # Processamento das imagens
        for img_path in tqdm(image_files, desc="Segmentando imagens"):
            relative_path = img_path.relative_to(input_dir)
            output_path = (output_dir / relative_path).with_suffix('.png')
            output_path.parent.mkdir(parents=True, exist_ok=True)

            segmented_image, result = self.segment_image(img_path, save_debug)

            # Salva a imagem segmentada se bem-sucedida
            if segmented_image is not None and result['success']:
                cv2.imwrite(str(output_path), segmented_image)

            # Prepara entrada do log
            log_entry = {
                'input_file': str(img_path),
                'output_file': str(output_path) if result['success'] else None,
                'status': 'SUCCESS' if result['success'] else 'FAIL',
                'error': result.get('error'),
                'validation': result.get('validation', {}),
                'metadata': result.get('metadata', {}),
                'processing_time': result.get('processing_time', 0),
                'iou_score': result.get('iou_score', 0),
                'timestamp': datetime.now().isoformat(),
            }
            results_log.append(log_entry)

            # Limpeza de memória periódica
            if len(results_log) % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Salva log detalhado
        log_file = output_dir / 'segmentation_log.json'
        with open(log_file, 'w') as f:
            json.dump(results_log, f, indent=2, ensure_ascii=False)

        # Gera relatório estatístico
        self._generate_final_report(output_dir, results_log)

        return self.stats

    def _generate_final_report(self, output_dir: Path, results_log: List[Dict]):
        """Gera relatório estatístico detalhado do processamento."""
        total = len(results_log)
        successful = len([r for r in results_log if r['status'] == 'SUCCESS'])
        failed = total - successful

        # Calcula estatísticas de validação
        valid_results = [r for r in results_log if r['status'] == 'SUCCESS']
        validation_passed = len([
            r for r in valid_results
            if r.get('validation', {}).get('overall_valid', False)
        ])

        # Estatísticas de tempo
        processing_times = [r['processing_time'] for r in valid_results if r['processing_time'] > 0]
        avg_time = np.mean(processing_times) if processing_times else 0

        # Estatísticas de qualidade
        quality_scores = [
            r.get('validation', {}).get('quality_score', 0)
            for r in valid_results
        ]
        avg_quality = np.mean(quality_scores) if quality_scores else 0

        # IoU scores
        iou_scores = [r.get('iou_score', 0) for r in valid_results if r.get('iou_score', 0) > 0]
        avg_iou = np.mean(iou_scores) if iou_scores else 0

        report = {
            'summary': {
                'total_images': total,
                'successful_segmentations': successful,
                'failed_segmentations': failed,
                'success_rate': (successful / total * 100) if total > 0 else 0,
                'validation_passed': validation_passed,
                'validation_rate': (validation_passed / successful * 100) if successful > 0 else 0
            },
            'performance': {
                'average_processing_time': round(avg_time, 2),
                'total_processing_time': round(sum(processing_times), 2),
                'average_quality_score': round(avg_quality, 3),
                'average_iou_score': round(avg_iou, 3)
            },
            'error_analysis': {},
            'timestamp': datetime.now().isoformat()
        }

        # Análise de erros
        failed_results = [r for r in results_log if r['status'] == 'FAIL']
        error_types = {}
        for result in failed_results:
            error = result.get('error', 'Unknown error')
            error_type = error.split(':')[0] if ':' in error else error
            error_types[error_type] = error_types.get(error_type, 0) + 1

        report['error_analysis'] = error_types

        # Salva relatório
        report_file = output_dir / 'final_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Log do relatório
        logger.info(f"""
        ===== RELATÓRIO FINAL =====
        Total de imagens: {total}
        Segmentações bem-sucedidas: {successful} ({report['summary']['success_rate']:.1f}%)
        Falhas de segmentação: {failed}
        Validações aprovadas: {validation_passed} ({report['summary']['validation_rate']:.1f}%)
        
        Tempo médio de processamento: {avg_time:.2f}s
        Score de qualidade médio: {avg_quality:.3f}
        IoU médio: {avg_iou:.3f}
        
        Relatório detalhado salvo em: {report_file}
        Log completo salvo em: {output_dir / 'segmentation_log.json'}
        """)


def main():
    parser = argparse.ArgumentParser(
        description="Segmentador pulmonar aprimorado usando MedSAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--datasets_root', type=str, default='data/',
                        help='Diretório raiz das bases de dados')
    parser.add_argument('--dataset_name', type=str, default='bigger_base',
                        help='Nome do subdiretório a ser processado')
    parser.add_argument('--outputs_root', type=str, default='./medsam_results',
                        help='Diretório raiz para salvar os resultados')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu', 'mps'],
                        help='Dispositivo para rodar o modelo')
    parser.add_argument('--debug', action='store_true',
                        help='Salva imagens de debug detalhadas')
    parser.add_argument('--force_download', action='store_true',
                        help='Força o re-download do modelo')
    parser.add_argument('--model_type', type=str, default='vit_b',
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        help='Tipo de modelo SAM a usar')

    args = parser.parse_args()

    # Configuração dos modelos disponíveis
    MODEL_CONFIGS = {
        'vit_b': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
            'filename': 'sam_vit_b_01ec64.pth'
        },
        'vit_l': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'filename': 'sam_vit_l_0b3195.pth'
        },
        'vit_h': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'filename': 'sam_vit_h_4b8939.pth'
        }
    }

    model_config = MODEL_CONFIGS[args.model_type]

    # Preparação de diretórios
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    checkpoint_path = models_dir / model_config['filename']

    input_dir = Path(args.datasets_root) / args.dataset_name
    output_dir = Path(args.outputs_root) / args.dataset_name

    # Validações iniciais
    if not input_dir.exists():
        logger.error(f"Diretório de entrada não encontrado: {input_dir}")
        return 1

    # Download do modelo
    if not download_file(model_config['url'], checkpoint_path, args.force_download):
        logger.error("Falha no download do modelo. Abortando.")
        return 1

    try:
        # Inicialização do segmentador
        segmenter = MedSAMSegmenter(
            checkpoint_path=str(checkpoint_path),
            model_type=args.model_type,
            device=args.device
        )

        # Processamento do dataset
        stats = segmenter.process_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            save_debug=args.debug,
        )

        logger.info("🎉 Processamento concluído com sucesso!")
        return 0

    except Exception as e:
        logger.error(f"Erro fatal durante o processamento: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())