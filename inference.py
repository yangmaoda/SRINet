
import random
import numpy as np
import argparse
from torchvision import transforms
from my_dataset import MyDataSet
from train import ConditionalCenterCrop
from utils import read_split_data
from model import CustomResNet50_v2
import os
import csv
from tqdm import tqdm
import torch
import pandas as pd
from scipy.stats import spearmanr, pearsonr

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()
def save_predictions_to_csv(image_names, predictions, mos_gt, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Name", "Predicted MOS", "True MOS"])
        for img_name, pred, gt in zip(image_names, predictions, mos_gt):
            writer.writerow([img_name, pred, gt])

def save_output_to_txt(output_lines, output_file):
    with open(output_file, mode='w') as file:
        for line in output_lines:
            file.write(line + "\n")
def val(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    all_split_results = read_split_data(args.data_path, normalize_mos=True)

    val_hq_image_paths = all_split_results[4]

    if args.split == 1:
        image_paths = all_split_results[3]
        mos_scores = all_split_results[5]
    elif args.split == 2:
        image_paths = all_split_results[6]
        mos_scores = all_split_results[8]

    data_for_dataset = {
        'Image': image_paths,
        'MOS': mos_scores
    }

    if val_hq_image_paths and not all(v is None for v in val_hq_image_paths):
        data_for_dataset['hq_Image'] = val_hq_image_paths
    elif args.val_mode == 0:
        print("Warning: FR mode (val_mode=0) requires valid 'hq_Image' values, but none were found.")

    val_df = pd.DataFrame(data_for_dataset)

    if val_df.empty:
        print(f"Error: loaded validation/test dataframe is empty from {args.data_path}. Check the 'split' column.")
        return

    data_transform = {
        "val": {
            "image": transforms.Compose([
                ConditionalCenterCrop(small_crop_size=224, large_crop_size=384),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            "mask": transforms.Compose([
                ConditionalCenterCrop(small_crop_size=224, large_crop_size=384),
                transforms.ToTensor()])
        }
    }

    val_dataset = MyDataSet(
        data_frame=val_df,
        root_dir=args.root_dir,
        mask_dir=args.mask_dir,
        transform=data_transform["val"],
        test_mode=args.val_mode,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers
    )

    model = CustomResNet50_v2(num_classes=1)
    model.to(device)
    model.eval()

    if args.weights and os.path.exists(args.weights):
        weights_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weights_dict, strict=False)
    else:
        print("Pretrained weights file was not found.")
        return

    all_preds = []
    all_mos_gt = []

    with torch.no_grad():
        for data in tqdm(val_loader, desc="Inference", unit="batch", leave=True):
            images, masks, mos_scores_batch = data
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images, masks).squeeze()
            all_preds.extend(outputs.cpu().flatten().tolist())
            all_mos_gt.extend(mos_scores_batch.cpu().flatten().tolist())

    if not all_preds or not all_mos_gt:
        print("Error: failed to collect predictions or MOS targets.")
        return

    try:
        srocc = spearmanr(all_mos_gt, all_preds).correlation
        plcc = pearsonr(all_mos_gt, all_preds)[0]
        print(f"SROCC: {srocc:.4f}")
        print(f"PLCC: {plcc:.4f}")

        save_predictions_to_csv(image_paths, all_preds, all_mos_gt, "predictions.csv")
        save_output_to_txt([f"SROCC: {srocc:.4f}", f"PLCC: {plcc:.4f}"], "inference_output.txt")

    except Exception as e:
        print(f"Error when computing SROCC/PLCC: {e}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, required=True, help='Root directory of distorted images')
    parser.add_argument('--mask-dir', type=str, required=True, help='Root directory of saliency masks')
    parser.add_argument('--data-path', type=str, required=True, help='Excel file containing image paths and MOS labels')
    parser.add_argument('--weights', type=str, required=True, help='Path to pretrained weights')
    parser.add_argument('--batch-size', type=int, default=4, help='Inference batch size')
    parser.add_argument('--device', type=str, default='cuda:0', help='Compute device (e.g. cuda:0 or cpu)')
    parser.add_argument('--val-mode', type=int, default=1, choices=[0, 1], help='0: FR mode, 1: NR mode')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of DataLoader workers; use 0 for debugging')
    parser.add_argument('--split', type=int, default=1, choices=[1, 2], help='Split for inference: 1 validation, 2 test')

    args = parser.parse_args()
    print("Run arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print("-" * 30)

    val(args)