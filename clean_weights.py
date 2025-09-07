import torch
from collections import OrderedDict

# Caminho para o seu modelo original e onde o novo modelo limpo será salvo
original_model_path = 'best_robust_snn_model.pth'
clean_model_path = 'best_robust_snn_model_clean.pth'

# Carrega o state_dict do modelo compilado
state_dict = torch.load(original_model_path)

# Cria um novo dicionário ordenado para os pesos limpos
new_state_dict = OrderedDict()

# Itera sobre cada camada no dicionário de estado
for key, value in state_dict.items():
    # Remove o prefixo '_orig_mod.' do nome da chave
    new_key = key.replace('_orig_mod.', '')
    new_state_dict[new_key] = value

# Salva o novo dicionário de estado limpo em um arquivo .pth
torch.save(new_state_dict, clean_model_path)

print(f"Modelo limpo com sucesso! ✨")
print(f"Original: {original_model_path}")
print(f"Novo arquivo salvo em: {clean_model_path}")