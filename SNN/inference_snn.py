import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import argparse
# Importe as classes da sua arquitetura SNN
from convert_ann_to_snn import SpikingResNet50, SpikingResidualBlock 

# --- FUNÇÃO DE PRÉ-PROCESSAMENTO ---
def preprocess_image(image_path, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3), # Garante 3 canais
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# --- BLOCO DE EXECUÇÃO DA INFERÊNCIA ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inferência com modelo SNN convertido")
    parser.add_argument('--snn_path', type=str, default='/home/mob4ai-001/Pibic_Artigo/SNNReadyResNet50_Results/bigger_base_20250905_184612/bigger_snn_finetuned.pth', help='Caminho para o modelo .pth da SNN convertida.')
    parser.add_argument('--image_path', type=str, default='bigger_base_segmented/bigger_base_segmented/Lung_Opacity/Lung_Opacity-2.png', help='Caminho para a imagem a ser classificada.')
    parser.add_argument('--num_classes', type=int, default=4, help='Número de classes do modelo.')
    # CORREÇÃO DA LISTA DE CLASSES APLICADA
    parser.add_argument('--classes', nargs='+', default=['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'], help='Lista com os nomes das classes.')
    parser.add_argument('--num_steps', type=int, default=25, help='Número de passos de tempo para a inferência SNN.')

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instancie a arquitetura SNN
    snn_model = SpikingResNet50(num_classes=args.num_classes).to(device)
    
    # 2. Carregue os pesos convertidos
    snn_model.load_state_dict(torch.load(args.snn_path, map_location=device))
    snn_model.eval()
    
    print(f"Modelo SNN carregado de '{args.snn_path}'")
    
    # 3. Prepare a imagem de entrada
    input_tensor = preprocess_image(args.image_path).to(device)
    
    # 4. Realize a inferência
    print(f"Realizando inferência com {args.num_steps} passos de tempo...")
    with torch.no_grad():
        output_spikes_sum = snn_model(input_tensor, num_steps=args.num_steps)
    # ADICIONE ESTA LINHA PARA DEBUG
    print(f"DEBUG: Contagem de spikes BRUTA: {output_spikes_sum.cpu().numpy()}")
  
    # --- INÍCIO DA NOVA LÓGICA DE CONFIANÇA ---
    
    # 5. Obtenha a predição final (o índice com mais spikes)
    prediction_idx = output_spikes_sum.argmax().item()
    prediction_class = args.classes[prediction_idx]
    
    # 6. Calcule o "nível de confiança" a partir da contagem de spikes
    # Normaliza a contagem de spikes para que some 1.0 (análogo ao Softmax)
    spike_probabilities = torch.nn.functional.softmax(output_spikes_sum, dim=1).squeeze()
    confidence = spike_probabilities[prediction_idx].item()
    
    # --- FIM DA NOVA LÓGICA DE CONFIANÇA ---
    
    print("\n--- Resultado da Predição ---")
    print(f"Classe Prevista: {prediction_class} (Índice: {prediction_idx})")
    print(f"Nível de Confiança (baseado em spikes): {confidence:.2%}") # Exibe como porcentagem
    
    print("\nDistribuição de Spikes (Análogo a Probabilidades):")
    for i, class_name in enumerate(args.classes):
        prob = spike_probabilities[i].item()
        print(f"  - {class_name}: {prob:.2%}")