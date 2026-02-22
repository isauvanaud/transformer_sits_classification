'''Test script.
'''

import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
from dataset import PixelSetData, Padding # pad_collate
from models.transformer.transformer import Transformer
from models.classifiers import ShallowClassifier
import torch.optim as optim
import pdb
import os
from tqdm import tqdm
import yaml
import sys
import pickle as pkl
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


results_folder = sys.argv[1]

with open(os.path.join(results_folder, 'train_config.yaml'), 'r') as file:
    cfg = yaml.safe_load(file)

for k, v in cfg.items():
    print(k + ': ' + str(v))

dataset = PixelSetData(cfg['data_folder'], set='test')

n_classes = len(np.unique(dataset.labels))

padding = Padding(pad_value=cfg['pad_value'])

data_loader = torch.utils.data.DataLoader(
    ...
)

encoder = Transformer(
    n_channels=cfg['n_channels'],
    n_pixels=cfg['n_pixels'],
    d_model=cfg['d_model'],
    d_inner=cfg['d_inner'],
    n_head=cfg['n_head'],
    d_k=cfg['d_k'],
    d_v=cfg['d_v'],
    dropout=cfg['dropout'],
    pad_value=cfg['pad_value'],
    scale_emb_or_prj=cfg['scale_emb_or_prj'],
    n_position=cfg['pos_embedding']['n_position'],
    T=cfg['pos_embedding']['T'],
    return_attns=cfg['return_attns'],
    learnable_query=cfg['learnable_query'],
    spectral_indices_embedding=cfg['spectral_indices'],
    channels=cfg['channels'],
    compute_values=cfg['compute_values']
)

classifier = ShallowClassifier(
    d_input=cfg['d_model'],
    d_inner=cfg['classifier']['d_inner'],
    n_classes=n_classes)


if cfg['device'] == 'cuda' and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

encoder = encoder.to(device)
classifier = classifier.to(device)
checkpoint = torch.load(os.path.join(results_folder, 'best_model.pth.tar'), map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
classifier.load_state_dict(checkpoint['classifier'])
del checkpoint

encoder.eval()
classifier.eval()

y_pred = []
y_true = []
for data, doys, labels in tqdm(data_loader, desc=f"Test"):
    data = data.to(device)
    doys = doys.to(device)
    labels = labels.to(device)
    batch_size = data.shape[0]

    with torch.no_grad():
        z, _ = encoder(data, doys)
        logits = classifier(z)
    loss = F.cross_entropy(logits, labels)
    pred = torch.argmax(logits, dim=-1)
    accuracy = (pred == labels).float().mean()

    y_pred.extend(pred.tolist())
    y_true.extend(labels.tolist())


f1_score_ = f1_score(y_true, y_pred, average=None)
f1_score_ = [round(float(x), 2) for x in f1_score_]

metrics = {
    "f1_score": f1_score_,
    "avg_f1_score": f1_score(y_true, y_pred, average="macro"),
    "accuracy": accuracy_score(y_true, y_pred)
}

cm = confusion_matrix(y_true, y_pred)

with open(os.path.join(results_folder, 'test_metrics.yaml'), 'w') as file:
    yaml.dump(metrics, file)

pkl.dump(cm, open(os.path.join(results_folder, 'conf_mat.pkl'), 'wb'))
