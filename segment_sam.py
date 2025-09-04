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
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, morphology

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
        logging.FileHandler('segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    @staticmethod
    def normalize_medical_image(image: np.ndarray, window_center: int = None, window_width: int = None) -> np.ndarray:
        if window_center and window_width:
            min_val = window_center - window_width // 2
            max_val = window_center + window_width // 2
            image = np.clip(image, min_val, max_val)
        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        return image

    @staticmethod
    def enhance_contrast(image: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        return clahe.apply(image)

    @staticmethod
    def load_dicom_image(path: Path) -> Tuple[np.ndarray, Dict]:
        try:
            ds = pydicom.dcmread(str(path))
            image = ds.pixel_array.astype(float)
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                image = image * ds.RescaleSlope + ds.RescaleIntercept
            metadata = {
                'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                'modality': getattr(ds, 'Modality', 'Unknown'),
                'window_center': getattr(ds, 'WindowCenter', None),
                'window_width': getattr(ds, 'WindowWidth', None)
            }
            return image, metadata
        except Exception as e:
            logger.error(f"Erro ao carregar DICOM {path}: {e}")
            return None, {}

class LungSegmentationValidator:
    @staticmethod
    def validate_lung_mask(mask: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, any]:
        total_pixels = image_shape[0] * image_shape[1]
        mask_pixels = np.sum(mask > 0)
        area_ratio = mask_pixels / total_pixels
        validation = {
            'area_ratio': area_ratio,
            'valid_area': 0.05 <= area_ratio <= 0.6,
            'connected_components': 0,
            'valid_components': False,
            'circularity': 0,
            'valid_shape': False,
            'overall_valid': False
        }
        if mask_pixels == 0:
            return validation
        labeled_mask = measure.label(mask)
        components = measure.regionprops(labeled_mask)
        validation['connected_components'] = len(components)
        validation['valid_components'] = 1 <= len(components) <= 3
        if components:
            areas = [comp.area for comp in components]
            perimeters = [comp.perimeter for comp in components if comp.perimeter > 0]
            if perimeters:
                circularities = [4 * np.pi * area / (perimeter**2) for area, perimeter in zip(areas, perimeters)]
                validation['circularity'] = np.mean(circularities)
                validation['valid_shape'] = 0.1 <= validation['circularity'] <= 0.7
        validation['overall_valid'] = (
            validation['valid_area'] and 
            validation['valid_components'] and 
            validation['valid_shape']
        )
        return validation

def download_file(url: str, destination: Path, force_download: bool = False) -> bool:
    if destination.exists() and not force_download:
        logger.info(f"Modelo já existe em: {destination}")
        return True
    logger.info(f"Baixando modelo de {url}...")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(destination, 'wb') as f, tqdm(
                total=total_size,
                unit='iB',
                unit_scale=True,
                desc=destination.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        logger.info(f"Modelo salvo com sucesso em: {destination}")
        return True
    except Exception as e:
        logger.error(f"Falha no download: {e}")
        if destination.exists():
            destination.unlink()
        return False

class ImprovedMedSAMSegmenter:
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        logger.info(f"Inicializando MedSAM no dispositivo: {self.device}")
        self.model_type = "vit_h"
        try:
            sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            logger.info("Modelo carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
        self.preprocessor = ImagePreprocessor()
        self.validator = LungSegmentationValidator()
        self.stats = {
            'total_images': 0,
            'successful_segmentations': 0,
            'failed_segmentations': 0,
            'validation_failures': 0
        }

    def _get_anatomical_lung_bbox(self, image_shape: Tuple[int, int], image: np.ndarray = None) -> np.ndarray:
        height, width = image_shape[:2]
        x_start = width * 0.1
        x_end = width * 0.9
        y_start = height * 0.15
        y_end = height * 0.85
        return np.array([x_start, y_start, x_end, y_end])

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        mask_cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=1000)
        mask_filled = ndimage.binary_fill_holes(mask_cleaned)
        kernel = morphology.disk(3)
        mask_smooth = morphology.opening(mask_filled, kernel)
        mask_smooth = morphology.closing(mask_smooth, kernel)
        return mask_smooth.astype(np.uint8) * 255

    def segment_image(self, image_path: Path, save_debug: bool = False) -> Tuple[Optional[np.ndarray], Dict]:
        result = {
            'success': False,
            'validation': {},
            'error': None,
            'metadata': {}
        }
        try:
            if image_path.suffix.lower() == '.dcm':
                image_array, metadata = self.preprocessor.load_dicom_image(image_path)
                if image_array is None:
                    raise ValueError("Falha ao carregar DICOM")
                result['metadata'] = metadata
            else:
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError("Falha ao carregar imagem")
                image_array = image.astype(float)
            window_center = result['metadata'].get('window_center')
            window_width = result['metadata'].get('window_width')
            if window_center is None and window_width is None:
                window_center, window_width = 40, 350
            processed_image = self.preprocessor.normalize_medical_image(
                image_array, window_center, window_width
            )
            processed_image = self.preprocessor.enhance_contrast(processed_image)
            if len(processed_image.shape) == 2:
                image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = processed_image
            self.predictor.set_image(image_rgb)
            bbox_prompt = self._get_anatomical_lung_bbox(image_rgb.shape, processed_image)
            best_mask = None
            best_score = -1
            best_validation = {}
            for multimask in [True, False]:
                masks, scores, logits = self.predictor.predict(
                    box=bbox_prompt,
                    multimask_output=multimask
                )
                for i, (mask, score) in enumerate(zip(masks, scores)):
                    validation = self.validator.validate_lung_mask(mask, image_rgb.shape[:2])
                    combined_score = score * (1.0 if validation['overall_valid'] else 0.5)
                    if combined_score > best_score:
                        best_score = combined_score
                        best_mask = mask
                        best_validation = validation
            if best_mask is None:
                raise ValueError("Nenhuma máscara válida gerada")
            refined_mask = self._postprocess_mask(best_mask)
            segmented_lung = cv2.bitwise_and(processed_image, processed_image, mask=refined_mask)
            final_validation = self.validator.validate_lung_mask(refined_mask, processed_image.shape)
            if save_debug:
                self._save_debug_info(image_path, image_rgb, refined_mask, bbox_prompt)
            self.stats['successful_segmentations'] += 1
            if not final_validation['overall_valid']:
                self.stats['validation_failures'] += 1
                logger.warning(f"Segmentação de {image_path.name} falhou na validação: {final_validation}")
            result.update({
                'success': True,
                'validation': final_validation,
                'score': best_score
            })
            return segmented_lung, result
        except Exception as e:
            self.stats['failed_segmentations'] += 1
            error_msg = f"Erro ao segmentar {image_path.name}: {e}"
            logger.error(error_msg)
            result['error'] = str(e)
            return None, result

    def _save_debug_info(self, image_path: Path, original: np.ndarray, mask: np.ndarray, bbox: np.ndarray):
        debug_dir = Path("debug_output")
        debug_dir.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original)
        axes[0].set_title("Original")
        axes[0].axis('off')
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Mask")
        axes[1].axis('off')
        axes[2].imshow(original)
        axes[2].imshow(mask, alpha=0.5, cmap='Reds')
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                           linewidth=2, edgecolor='yellow', facecolor='none')
        axes[2].add_patch(rect)
        axes[2].set_title("Overlay + BBox")
        axes[2].axis('off')
        plt.tight_layout()
        plt.savefig(debug_dir / f"{image_path.stem}_debug.png", dpi=150, bbox_inches='tight')
        plt.close()

    def process_dataset(self, input_dir: Path, output_dir: Path, save_debug: bool = False) -> Dict:
        logger.info(f"Processando dataset: {input_dir} -> {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.dcm', '*.dicom']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.rglob(ext))
        if not image_files:
            logger.error(f"Nenhuma imagem encontrada em {input_dir}")
            return {}
        logger.info(f"Encontradas {len(image_files)} imagens")
        self.stats = {k: 0 for k in self.stats.keys()}
        self.stats['total_images'] = len(image_files)
        results_log = []
        for img_path in tqdm(image_files, desc="Segmentando imagens"):
            relative_path = img_path.relative_to(input_dir)
            output_path = output_dir / relative_path.with_suffix('.png')
            output_path.parent.mkdir(parents=True, exist_ok=True)
            segmented_image, result = self.segment_image(img_path, save_debug)
            if segmented_image is not None and result['success']:
                cv2.imwrite(str(output_path), segmented_image)
                log_entry = {
                    'input_file': str(img_path),
                    'output_file': str(output_path),
                    'timestamp': datetime.now().isoformat(),
                    'validation': result['validation'],
                    'metadata': result.get('metadata', {})
                }
                results_log.append(log_entry)
        log_file = output_dir / 'segmentation_log.json'
        with open(log_file, 'w') as f:
            json.dump(results_log, f, indent=2)
        success_rate = self.stats['successful_segmentations'] / self.stats['total_images'] * 100
        validation_rate = (self.stats['successful_segmentations'] - self.stats['validation_failures']) / self.stats['total_images'] * 100
        logger.info(f"""
        ===== RELATÓRIO FINAL =====
        Total de imagens: {self.stats['total_images']}
        Segmentações bem-sucedidas: {self.stats['successful_segmentations']} ({success_rate:.1f}%)
        Falhas de segmentação: {self.stats['failed_segmentations']}
        Falhas de validação: {self.stats['validation_failures']}
        Taxa de validação aprovada: {validation_rate:.1f}%
        Log detalhado salvo em: {log_file}
        """)
        return self.stats

def main(args):
    MODEL_CONFIGS = {
        'sam_vit_h': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
            'filename': 'sam_vit_h_4b8939.pth'
        },
        'sam_vit_l': {
            'url': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'filename': 'sam_vit_l_0b3195.pth'
        }
    }
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_config = MODEL_CONFIGS[args.model]
    checkpoint_path = models_dir / model_config['filename']
    if not download_file(model_config['url'], checkpoint_path, args.force_download):
        logger.error("Falha no download do modelo. Abortando.")
        return

    # Novo: Escolha de banco de dados conforme argumentos
    input_dir = Path(args.datasets_root) / args.dataset_name
    output_dir = Path(args.outputs_root) / args.dataset_name

    if not input_dir.exists():
        all_bases = [p.name for p in Path(args.datasets_root).iterdir() if p.is_dir()]
        logger.error(f"Base de dados '{args.dataset_name}' não encontrada. Disponíveis: {all_bases}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        segmenter = ImprovedMedSAMSegmenter(
            checkpoint_path=str(checkpoint_path),
            device=args.device
        )
    except Exception as e:
        logger.error(f"Falha ao inicializar segmentador: {e}")
        return
    stats = segmenter.process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        save_debug=args.debug
    )
    logger.info("🎉 Processamento concluído!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmentador de pulmões usando MedSAM")
    parser.add_argument('--datasets_root', type=str, default='./data', help='Diretório raiz das bases de dados')
    parser.add_argument('--dataset_name', type=str, default='bigger_base', help='Nome do subdiretório (base) a processar')
    parser.add_argument('--outputs_root', type=str, default='./sam_results_bigger_base', help='Diretório raiz para resultados')
    parser.add_argument('--model', choices=['sam_vit_h', 'sam_vit_l'],
                      default='sam_vit_h',
                      help='Modelo SAM a usar (padrão: sam_vit_h)')
    parser.add_argument('--device', default='auto',
                      help='Dispositivo (cuda/cpu/auto)')
    parser.add_argument('--debug', action='store_true',
                      help='Salvar informações de debug')
    parser.add_argument('--force_download', action='store_true',
                      help='Forçar re-download do modelo')
    args = parser.parse_args()
    main(args)
