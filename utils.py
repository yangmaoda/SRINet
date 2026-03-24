import os
import sys
import json
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn import functional as F


def read_split_data(excel_file, val_rate=0.2, normalize_mos=False):
    """
    Read data from Excel and split by `split` column.
    """
    data = pd.read_excel(excel_file)
    assert 'split' in data.columns, "Missing 'split' column in the Excel file."
    data['Image'] = data['Image'].str.lower()

    if normalize_mos:
        mos_values = data['MOS']
        min_mos = mos_values.min()
        max_mos = mos_values.max()
        data['MOS'] = (mos_values - min_mos) / (max_mos - min_mos)

    train_data = data[data['split'] == 0]
    val_data = data[data['split'] == 1]
    test_data = data[data['split'] == 2]

    train_images_path = train_data['Image'].tolist()
    train_images_mos = train_data['MOS'].tolist()
    val_images_path = val_data['Image'].tolist()
    val_images_mos = val_data['MOS'].tolist()
    test_images_path = test_data['Image'].tolist()
    test_images_mos = test_data['MOS'].tolist()

    train_hq_images = train_data['hq_Image'].str.lower().tolist() if 'hq_Image' in train_data.columns else [None] * len(
        train_data)
    val_hq_images = val_data['hq_Image'].str.lower().tolist() if 'hq_Image' in val_data.columns else [None] * len(
        val_data)
    test_hq_images = test_data['hq_Image'].str.lower().tolist() if 'hq_Image' in test_data.columns else [None] * len(
        test_data)

    return (
        train_images_path, train_hq_images, train_images_mos,
        val_images_path, val_hq_images, val_images_mos,
        test_images_path, test_hq_images, test_images_mos
    )


def plot_data_loader_image(data_loader):
    """
    Visualize sample images from a dataloader.
    """
    batch = next(iter(data_loader))
    images, masks, mos_scores = batch
    plot_num = min(len(images), 4)

    plt.figure(figsize=(12, 3))
    for i in range(plot_num):
        img = images[i].numpy().transpose(1, 2, 0)
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.subplot(1, plot_num, i + 1)
        plt.imshow(img)
        plt.title(f"MOS: {mos_scores[i]:.2f}")
        plt.axis('off')
    plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def pairwise_ranking_loss(scores, targets):
    """
    Pairwise ranking loss.
    """
    loss = 0.0
    n = scores.size(0)
    count = 0
    for i in range(n):
        for j in range(n):
            if targets[i] > targets[j]:
                count += 1
                diff = scores[j] - scores[i]
                loss += torch.log1p(torch.exp(diff))
    if count > 0:
        loss /= count
    return loss


def nin_loss(pred, target, p=1, q=2):
    # Robustly handle scalar tensors (e.g., batch_size=1).
    if pred.dim() == 0:
        pred = pred.unsqueeze(0)
    if target.dim() == 0:
        target = target.unsqueeze(0)

    if pred.ndim == 1:
        pred = pred.unsqueeze(1)
    if target.ndim == 1:
        target = target.unsqueeze(1)

    batch_size = pred.size(0)
    if batch_size < 2:
        return F.l1_loss(pred, target)

    vx = pred - pred.mean()
    vy = target - target.mean()
    norm_pred = F.normalize(vx, p=q, dim=0)
    norm_target = F.normalize(vy, p=q, dim=0)
    scale = 2 ** p * batch_size ** max(0, 1 - p / q)
    return torch.norm(norm_pred - norm_target, p=p) / scale


def train_one_epoch(model, optimizer, data_loader, device, mse_weight=1.0, nin_weight=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    progress_bar = tqdm(data_loader, desc='Training', leave=True)

    for data in progress_bar:
        images, masks, targets = data
        images, masks, targets = images.to(device), masks.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(images, masks)

        outputs = outputs.squeeze().view(-1)
        targets = targets.float().view(-1)

        mse = F.mse_loss(outputs, targets)
        nin = nin_loss(outputs, targets)
        loss = mse_weight * mse + nin_weight * nin

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Ensure `extend` always receives an iterable.
        preds_np = np.atleast_1d(outputs.detach().cpu().numpy())
        targets_np = np.atleast_1d(targets.detach().cpu().numpy())
        all_preds.extend(preds_np)
        all_targets.extend(targets_np)

        progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    epoch_loss = running_loss / len(data_loader.dataset)
    srocc = spearmanr(all_targets, all_preds).correlation
    plcc = pearsonr(all_targets, all_preds)[0]
    print(f'End of epoch - Loss: {epoch_loss:.4f} SROCC: {srocc:.4f} PLCC: {plcc:.4f}')
    return epoch_loss, srocc, plcc


@torch.no_grad()
def evaluate(model, data_loader, device, mse_weight=1.0, nin_weight=1.0):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    progress_bar = tqdm(data_loader, desc='Validating', leave=True)

    for data in progress_bar:
        images, masks, targets = data
        images, masks, targets = images.to(device), masks.to(device), targets.to(device)

        outputs = model(images, masks)

        outputs = outputs.squeeze().view(-1)
        targets = targets.float().view(-1)

        mse = F.mse_loss(outputs, targets)
        nin = nin_loss(outputs, targets)
        loss = mse_weight * mse + nin_weight * nin

        running_loss += loss.item() * images.size(0)

        # Ensure `extend` always receives an iterable.
        preds_np = np.atleast_1d(outputs.detach().cpu().numpy())
        targets_np = np.atleast_1d(targets.detach().cpu().numpy())
        all_preds.extend(preds_np)
        all_targets.extend(targets_np)

        progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    epoch_loss = running_loss / len(data_loader.dataset)
    srocc = spearmanr(all_targets, all_preds).correlation
    plcc = pearsonr(all_targets, all_preds)[0]
    print(f'End of validation - Loss: {epoch_loss:.4f} SROCC: {srocc:.4f} PLCC: {plcc:.4f}')
    return epoch_loss, srocc, plcc