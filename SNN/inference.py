import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import snntorch as snn
from snntorch import surrogate
import argparse
from collections import OrderedDict
import os

# --- PASSO 1: DEFINIÇÃO DA ARQUITETURA DO MODELO ---
# É ESSENCIAL que estas classes sejam IDÊNTICAS às do seu script de treino.
# Copie e cole as classes SNNResidualBlock e AdvancedSNN_Classifier aqui.

class SNNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, beta=0.9, spike_grad=surrogate.atan()):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

    def forward(self, x):
        out = self.lif1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.lif_out(out)
        return out

class AdvancedSNN_Classifier(nn.Module):
    def __init__(self, num_classes, beta=0.9, dropout=0.5):
        super().__init__()
        # É necessário importar 'utils' para a função de reset
        from snntorch import utils
        self.utils = utils
        
        spike_grad = surrogate.atan(alpha=2.0)
        self.stem = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64))
        self.stem_lif = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2, stride=1, beta=beta, spike_grad=spike_grad)
        self.layer2 = self._make_layer(64, 128, 2, stride=2, beta=beta, spike_grad=spike_grad)
        self.layer3 = self._make_layer(128, 256, 2, stride=2, beta=beta, spike_grad=spike_grad)
        self.layer4 = self._make_layer(256, 512, 2, stride=2, beta=beta, spike_grad=spike_grad)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.lif_fc1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc2 = nn.Linear(256, 128)
        self.synaptic_fc2 = snn.Synaptic(alpha=0.8, beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc3 = nn.Linear(128, num_classes)
        self.lif_out = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)

    def _make_layer(self, in_channels, out_channels, blocks, stride, beta, spike_grad):
        layers = []
        layers.append(SNNResidualBlock(in_channels, out_channels, stride, beta, spike_grad))
        for _ in range(1, blocks):
            layers.append(SNNResidualBlock(out_channels, out_channels, 1, beta, spike_grad))
        return nn.Sequential(*layers)

    def forward(self, x):
        self.utils.reset(self)
        spk_rec = []
        # O número de passos deve ser o mesmo do treino para consistência
        num_steps = 20 
        for step in range(num_steps):
            cur = self.stem_lif(self.stem(x))
            cur = self.maxpool(cur)
            cur = self.layer1(cur)
            cur = self.layer2(cur)
            cur = self.layer3(cur)
            cur = self.layer4(cur)
            cur = self.avgpool(cur)
            cur = torch.flatten(cur, 1)
            # Durante a inferência, o dropout é desativado por model.eval()
            cur = self.dropout(cur)
            cur = self.lif_fc1(self.fc1(cur))
            cur = self.dropout(cur)
            cur = self.synaptic_fc2(self.fc2(cur))
            cur = self.dropout(cur)
            spk_out, _ = self.lif_out(self.fc3(cur))
            spk_rec.append(spk_out)
        return torch.stack(spk_rec, dim=0).sum(dim=0)


# --- PASSO 2: FUNÇÃO DE PRÉ-PROCESSAMENTO ---
def preprocess_image(image_path, image_size=128):
    """Carrega uma imagem, aplica as transformações de validação e retorna um tensor."""
    # Estas transformações devem ser as mesmas da validação no seu script de treino
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0) # Adiciona a dimensão do lote (batch)
    except FileNotFoundError:
        print(f"ERRO: A imagem em '{image_path}' não foi encontrada.")
        return None

# --- PASSO 3: BLOCO PRINCIPAL DE INFERÊNCIA ---
def main(args):
    # 1. Configurar o ambiente
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Carregar a arquitetura do modelo
    model = AdvancedSNN_Classifier(num_classes=len(args.classes), beta=0.9, dropout=0.5).to(device)
    
    # 3. Carregar os pesos treinados (state_dict)
    if not os.path.exists(args.model_path):
        print(f"ERRO: Arquivo de modelo não encontrado em '{args.model_path}'")
        return
        
    state_dict = torch.load(args.model_path, map_location=device)

    # Lógica para limpar prefixos '_orig_mod.' caso o modelo tenha sido salvo com torch.compile
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace('_orig_mod.', '')] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    # 4. Colocar o modelo em modo de avaliação
    model.eval()
    print(f"✅ Modelo '{args.model_path}' carregado com sucesso.")

    # 5. Carregar e pré-processar a imagem de teste
    input_tensor = preprocess_image(args.image_path)
    if input_tensor is None:
        return
    input_tensor = input_tensor.to(device)

    # 6. Realizar a inferência
    print(f"\n🔎 Realizando inferência na imagem: {args.image_path}...")
    with torch.no_grad(): # Desativa o cálculo de gradientes para acelerar
        output_spikes_sum = model(input_tensor)

    # 7. Calcular a predição e o nível de confiança
    prediction_idx = output_spikes_sum.argmax().item()
    prediction_class = args.classes[prediction_idx]
    
    # Normaliza a contagem de spikes para obter uma distribuição de confiança
    spike_probabilities = torch.nn.functional.softmax(output_spikes_sum, dim=1).squeeze()
    confidence = spike_probabilities[prediction_idx].item()

    # 8. Exibir os resultados
    print("\n--- 🎯 Resultado da Predição ---")
    print(f"Classe Prevista: {prediction_class} (Índice: {prediction_idx})")
    print(f"Nível de Confiança (baseado em spikes): {confidence:.2%}")
    
    print("\n📊 Distribuição de Confiança por Classe:")
    for i, class_name in enumerate(args.classes):
        prob = spike_probabilities[i].item()
        print(f"  - {class_name:<20}: {prob:.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script de Inferência para o modelo SNN Robusto")
    parser.add_argument('--model_path', type=str, default="/home/mob4ai-001/Pibic_Artigo/SNN/best_robust_snn_model.pth",
                        help='Caminho para o arquivo .pth do modelo SNN treinado.')
    parser.add_argument('--image_path', type=str, default='/home/mob4ai-001/Pibic_Artigo/image copy 4.png',
                        help='Caminho para a imagem de entrada que será classificada.')
    parser.add_argument('--classes', nargs='+', required=True,
                        help='Lista com os nomes das classes, na ordem correta. Ex: "COVID" "Normal" ...')
    
    args = parser.parse_args()
    main(args)

    