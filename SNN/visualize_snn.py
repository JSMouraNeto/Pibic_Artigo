import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import snntorch as snn
from snntorch import surrogate, utils
import numpy as np
import os

# --- PASSO 1: COPIE AS CLASSES DO SEU MODELO AQUI ---
# É essencial que estas definições sejam idênticas às do script de treinamento.

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

    def forward_with_probe(self, x, layer_to_probe):
        """
        Uma versão modificada do 'forward' que captura e retorna os spikes
        de uma camada específica ao longo do tempo.
        """
        utils.reset(self)
        
        probed_spikes = []
        num_steps = 20 # Definido com base nos hiperparâmetros do treino

        for step in range(num_steps):
            cur = self.stem_lif(self.stem(x))
            if layer_to_probe == 'stem_lif': probed_spikes.append(cur.clone().detach().cpu())
            
            cur = self.maxpool(cur)
            
            cur = self.layer1(cur)
            if layer_to_probe == 'layer1': probed_spikes.append(cur.clone().detach().cpu())
            
            cur = self.layer2(cur)
            if layer_to_probe == 'layer2': probed_spikes.append(cur.clone().detach().cpu())

            cur = self.layer3(cur)
            if layer_to_probe == 'layer3': probed_spikes.append(cur.clone().detach().cpu())
            
            cur = self.layer4(cur)
            if layer_to_probe == 'layer4': probed_spikes.append(cur.clone().detach().cpu())
            
            # O resto do forward é necessário para a rede funcionar, mas não precisamos mais capturar
            cur = self.avgpool(cur)
            cur = torch.flatten(cur, 1)
            cur = self.dropout(cur)
            cur = self.lif_fc1(self.fc1(cur))
            cur = self.dropout(cur)
            cur = self.synaptic_fc2(self.fc2(cur))
            cur = self.dropout(cur)
            self.lif_out(self.fc3(cur))
        
        return torch.stack(probed_spikes, dim=0)


# --- PASSO 2: FUNÇÃO PARA CARREGAR E PRÉ-PROCESSAR A IMAGEM ---
def load_and_preprocess_image(image_path, img_size=128):
    """Carrega uma imagem e aplica as transformações de validação."""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image).unsqueeze(0) # Adiciona a dimensão do batch
    except FileNotFoundError:
        print(f"ERRO: A imagem em '{image_path}' não foi encontrada.")
        return None

# --- PASSO 3: FUNÇÃO PRINCIPAL PARA GERAR A ANIMAÇÃO ---
def create_snn_animation(model_path, image_path, layer_name, num_classes, output_filename="snn_spikes.gif"):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Verificar se o modelo existe
    if not os.path.exists(model_path):
        print(f"ERRO: O arquivo do modelo '{model_path}' não foi encontrado.")
        return

    # Carregar modelo treinado
    model = AdvancedSNN_Classifier(num_classes=num_classes, beta=0.9, dropout=0.5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Colocar em modo de avaliação

    # Carregar e preparar a imagem
    input_tensor = load_and_preprocess_image(image_path)
    if input_tensor is None:
        return
    input_tensor = input_tensor.to(device)
    
    # Obter os spikes da camada desejada
    print(f"Processando imagem e capturando spikes da camada '{layer_name}'...")
    spikes_over_time = model.forward_with_probe(input_tensor, layer_to_probe=layer_name)
    
    # Formato: [time, batch, channels, height, width]
    # Tira a média dos spikes através dos canais para uma visualização 2D
    spikes_visualization = spikes_over_time.mean(dim=2).squeeze(1) # Remove dimensão de batch e canal
    
    # Configurar a animação
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('black')
    plt.axis('off')

    im = ax.imshow(spikes_visualization[0], cmap='hot', interpolation='nearest')
    title = ax.text(0.5, 1.05, '', ha="center", va="bottom", color="white", fontsize=20, transform=ax.transAxes)
    
    def update(frame):
        im.set_array(spikes_visualization[frame])
        title.set_text(f'Time Step (t) = {frame + 1} / {len(spikes_visualization)}')
        return [im, title]

    print("Gerando animação... Isso pode levar um minuto. ⏳")
    ani = animation.FuncAnimation(fig, update, frames=len(spikes_visualization), interval=100, blit=True)

    # Salvar a animação
    ani.save(output_filename, writer='pillow')
    
    print(f"\nAnimação salva com sucesso como '{output_filename}'! ✨")
    plt.close()


# --- EXECUÇÃO ---
# --- EXECUÇÃO ---
if __name__ == '__main__':
    # --- CONFIGURE AQUI ---
    MODEL_WEIGHTS_PATH = "/home/mob4ai-001/Pibic_Artigo/SNN/best_robust_snn_model_clean.pth"
    IMAGE_TO_VISUALIZE = "/home/mob4ai-001/Pibic_Artigo/image.png" 
    NUM_CLASSES = 4  # Altere para o seu número de classes!

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carregar modelo uma vez
    model = AdvancedSNN_Classifier(num_classes=NUM_CLASSES, beta=0.9, dropout=0.5)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Carregar imagem
    input_tensor = load_and_preprocess_image(IMAGE_TO_VISUALIZE)
    if input_tensor is None:
        exit()
    input_tensor = input_tensor.to(device)

    # Definir camadas em ordem
    layers = ['stem_lif', 'layer1', 'layer2', 'layer3', 'layer4']

    # Guardar todos os frames (camadas + time steps)
    all_frames = []
    all_labels = []

    for layer in layers:
        print(f"Capturando spikes da camada {layer}...")
        spikes = model.forward_with_probe(input_tensor, layer_to_probe=layer)

        # [time, batch, channels, h, w] → média nos canais
        spikes_vis = spikes.mean(dim=2).squeeze(1)  

        for t in range(spikes_vis.shape[0]):
            all_frames.append(spikes_vis[t].cpu().numpy())
            all_labels.append(f"{layer} | t = {t+1}/{spikes_vis.shape[0]}")

    # Criar GIF
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('black')
    plt.axis('off')

    im = ax.imshow(all_frames[0], cmap='hot', interpolation='nearest')
    title = ax.text(0.5, 1.05, '', ha="center", va="bottom", color="white", fontsize=18, transform=ax.transAxes)

    def update(frame_idx):
        im.set_array(all_frames[frame_idx])
        title.set_text(all_labels[frame_idx])
        return [im, title]

    print("Gerando animação final com todas as camadas... ⏳")
    ani = animation.FuncAnimation(fig, update, frames=len(all_frames), interval=150, blit=True)

    output_filename = "snn_all_layers2.gif"
    ani.save(output_filename, writer='pillow')
    plt.close()

    print(f"\nGIF gerado com sucesso: {output_filename} 🎉")
