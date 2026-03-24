import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from model import CustomResNet50_v2
from train import ConditionalCenterCrop

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PredictDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_list = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            img = self.transform["image"](img)
            mask = self.transform["mask"](mask)

        return img, mask, img_name

def main(args):
    seed_everything()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Keep preprocessing consistent with training.
    data_transform = {
        "image": transforms.Compose([
            ConditionalCenterCrop(small_crop_size=224, large_crop_size=384),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "mask": transforms.Compose([
            ConditionalCenterCrop(small_crop_size=224, large_crop_size=384),
            transforms.ToTensor(),
        ])
    }

    dataset = PredictDataset(args.image_dir, args.mask_dir, transform=data_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = CustomResNet50_v2(num_classes=1)
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)
    model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for imgs, masks, img_names in tqdm(dataloader, desc="Predicting"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds, *_ = model(imgs, masks)
            outputs = preds.view(-1)

            for name, mos in zip(img_names, outputs.cpu().numpy()):
                predictions.append({"Image": name, "Pred_MOS": mos})

    result_df = pd.DataFrame(predictions)
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "predictions_koniq.csv")
    result_df.to_csv(save_path, index=False)
    print(f"Saved predictions to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-dir', type=str, required=True, help='Input image directory')
    parser.add_argument('--mask-dir', type=str, required=True, help='Saliency mask directory')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--output-dir', type=str, default='./prediction_results_cat', help='Prediction output directory')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()
    main(args)
