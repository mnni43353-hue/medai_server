# interpret.py
import torch
import numpy as np

def ensure_tensor(img, transform):
    return transform(img).unsqueeze(0)  # [1, C, H, W]

def mc_dropout_predictions(model, input_tensor, n_samples=8):
    """تقدير عدم اليقين بالـ Dropout"""
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(input_tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds.append(probs)
    model.eval()
    preds = np.vstack(preds)
    mean = preds.mean(axis=0)[0]
    std = preds.std(axis=0)[0]
    return mean, std