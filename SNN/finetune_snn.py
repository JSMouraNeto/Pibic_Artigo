import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# --- IMPORTAÇÕES ESSENCIAIS ---
# Importe as classes do modelo SNN e as funções de dados
from convert_ann_to_snn import SpikingResNet50, SpikingResidualBlock 
from R50ToSnn    import get_medical_transforms, MedicalImageDataset, prepare_data 
# ATENÇÃO: Renomeie 'seu_script_de_treino_original.py' para o nome real do arquivo

def finetune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Carregar os dados (usando as mesmas funções do treino original)
    data_splits = prepare_data(args.data_dir, args.classes, logger=None) # Logger simplificado
    train_transform = get_medical_transforms(is_training=True)
    train_dataset = MedicalImageDataset(data_splits['train'][0], data_splits['train'][1], transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 2. Carregar o modelo SNN CONVERTIDO (não o ANN!)
    model = SpikingResNet50(num_classes=len(args.classes)).to(device)
    model.load_state_dict(torch.load(args.snn_path, map_location=device))
    print(f"Modelo SNN carregado de '{args.snn_path}' para ajuste fino.")

    # 3. Configurar o otimizador com TAXA DE APRENDIZADO BAIXA
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    print(f"Iniciando ajuste fino por {args.epochs} épocas...")

    for epoch in range(args.epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, targets, _ in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            # O forward da SNN já inclui os passos de tempo
            outputs = model(inputs, num_steps=args.num_steps)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})

    # 4. Salvar o modelo AJUSTADO
    torch.save(model.state_dict(), args.finetuned_save_path)
    print(f"\n✅ Ajuste fino concluído! Modelo salvo em '{args.finetuned_save_path}'")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ajuste Fino (Fine-Tuning) de SNN Convertida")
    # --- Caminhos ---
    parser.add_argument('--snn_path', type=str, required=True, help='Caminho para o SNN .pth CONVERTIDO.')
    parser.add_argument('--finetuned_save_path', type=str, required=True, help='Caminho para SALVAR o SNN .pth ajustado.')
    parser.add_argument('--data_dir', type=str, required=True, help='Diretório com os dados de treino.')
    
    # --- Parâmetros de Treino (valores baixos são importantes) ---
    parser.add_argument('--epochs', type=int, default=5, help='Número de épocas para o ajuste fino (poucas são necessárias).')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='TAXA DE APRENDIZADO BEM BAIXA para o ajuste fino.')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=15, help='Passos de tempo durante o ajuste fino.')

    # --- Parâmetros do Modelo ---
    parser.add_argument('--classes', nargs='+', required=True, help='Lista com os nomes das classes.')
    
    args = parser.parse_args()
    finetune(args)