import torch

model_sorce_path = '../../checkpoints/MetaQA'

#image = torch.load(model_sorce_path + '/best_score_model.pt', map_location='cpu')
image = torch.load(model_sorce_path + '/best_score_mode-1hop-full.pt', map_location='cpu')
print(" ")