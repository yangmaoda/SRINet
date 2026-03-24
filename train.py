import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
import pandas as pd
from model import CustomResNet50_v2
import torchvision.transforms.functional as F
import random
class ConditionalCenterCrop:
    def __init__(self, small_crop_size=224, large_crop_size=384):
        self.small_crop_size = small_crop_size
        self.large_crop_size = large_crop_size

    def __call__(self, img):
        width, height = img.size
        short_side = min(width, height)

        if short_side >= self.large_crop_size:
            # Resize short side to large_crop_size.
            scale_factor = self.large_crop_size / short_side
        else:
            # Resize short side to small_crop_size.
            scale_factor = self.small_crop_size / short_side

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        img = F.resize(img, (new_height, new_width))

        width, height = img.size
        short_side = min(width, height)

        if short_side >= self.large_crop_size:
            crop_size = self.large_crop_size
        else:
            crop_size = self.small_crop_size

        if short_side >= crop_size:
            return F.center_crop(img, crop_size)
        else:
            pad_width = max(crop_size - width, 0)
            pad_height = max(crop_size - height, 0)
            padding = (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)
            img = F.pad(img, padding, fill=0)
            return F.center_crop(img, crop_size)

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU.
    torch.backends.cudnn.deterministic = True  # Make results deterministic.
    torch.backends.cudnn.benchmark = False  # Disable cuDNN auto-tuner optimization.

def main(args):
    fix_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists("./weights"):
        os.makedirs("./weights")

    tb_writer = SummaryWriter()
    (
        train_images_path, train_hq_images, train_images_mos,
        val_images_path, val_hq_images, val_images_mos,
        test_images_path, test_hq_images, test_images_mos
    ) = read_split_data(args.data_path, normalize_mos=args.normalize_mos)

    train_data = pd.DataFrame({'Image': train_images_path, 'hq_Image': train_hq_images, 'MOS': train_images_mos})
    val_data = pd.DataFrame({'Image': val_images_path, 'hq_Image': val_hq_images, 'MOS': val_images_mos})
    test_data = pd.DataFrame({'Image': test_images_path, 'hq_Image': test_hq_images, 'MOS': test_images_mos})

    data_transform = {
        "train": {
            "image": transforms.Compose([
                ConditionalCenterCrop(small_crop_size=224, large_crop_size=384),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            "mask": transforms.Compose([
                ConditionalCenterCrop(small_crop_size=224, large_crop_size=384),
                transforms.ToTensor(),
            ])
        },
        "val": {
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
    }

    train_dataset = MyDataSet(data_frame=train_data,
                              root_dir=args.root_dir,
                              mask_dir=args.mask_dir,
                              transform=data_transform["train"],
                              test_mode=args.test_mode,
                              )

    val_dataset = MyDataSet(data_frame=val_data,
                            root_dir=args.root_dir,
                            mask_dir=args.mask_dir,
                            transform=data_transform["val"],
                            test_mode=args.test_mode,
                            )

    test_dataset = MyDataSet(data_frame=test_data,
                             root_dir=args.root_dir,
                             mask_dir=args.mask_dir,
                             transform=data_transform["val"],
                             test_mode=args.test_mode,
                             )

    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                               num_workers=4, collate_fn=train_dataset.collate_fn, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                             num_workers=4, collate_fn=val_dataset.collate_fn, prefetch_factor=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True,
                                              num_workers=4, prefetch_factor=2)

    model = CustomResNet50_v2(num_classes=1)
    model = model.to(device)

    def params_count(net):
        n_parameters = sum(p.numel() for p in net.parameters())
        return 'Parameters Count: {:.5f}M'.format(n_parameters / 1e6)

    print(params_count(model))

    if args.weights:
        assert os.path.exists(args.weights), f"weights file: '{args.weights}' not exist."
        weights_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(weights_dict, strict=True)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training", name)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)

    log_interval = 5
    data_path_parts = os.path.normpath(args.data_path).split(os.sep)
    dataset_name = data_path_parts[-2] if len(data_path_parts) >= 2 else "unknown_dataset"

    experiment_root = "experiment"
    dataset_experiment_dir = os.path.join(experiment_root, dataset_name)
    os.makedirs(dataset_experiment_dir, exist_ok=True)

    log_filename = f"training_{dataset_name}_log.txt"
    log_file = open(log_filename, 'a')
    log_file.write("========== Training Configuration ==========\n")
    for arg_name, arg_value in vars(args).items():
        log_file.write(f"{arg_name}: {arg_value}\n")
    log_file.write("============================================\n\n")
    log_file.flush()
    best_weight_path = os.path.join(dataset_experiment_dir, f"{args.weight_name}_best.pth")
    last_weight_path = best_weight_path.replace("_best.pth", "_last.pth")
    print(f"Current experiment weight directory: {dataset_experiment_dir}")

    best_val_loss = float('inf')
    best_val_srocc = 0
    best_val_plcc = 0
    best_epoch = -1

    for epoch in range(args.epochs):
        print(f"Training epoch: {epoch + 1}/{args.epochs}")

        train_loss, train_srocc, train_plcc = train_one_epoch(model, optimizer, train_loader, device, args.mse_weight,
                                                              args.nin_weight)
        val_loss, val_srocc, val_plcc = evaluate(model, val_loader, device, args.mse_weight, args.nin_weight)

        tb_writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
        tb_writer.add_scalars('SROCC', {'Train': train_srocc, 'Val': val_srocc}, epoch)
        tb_writer.add_scalars('PLCC', {'Train': train_plcc, 'Val': val_plcc}, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_srocc = val_srocc
            best_val_plcc = val_plcc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_weight_path)
            print(f"Saved new best model at epoch {best_epoch}, val loss: {val_loss:.4f}")

        if epoch % log_interval == 0:
            log_msg = (
                f"Epoch {epoch + 1}\n"
                f"Train - Loss: {train_loss:.4f} | SROCC: {train_srocc:.4f} | PLCC: {train_plcc:.4f}\n"
                f"Val   - Loss: {val_loss:.4f} | SROCC: {val_srocc:.4f} | PLCC: {val_plcc:.4f}\n"
                f"Best  - Val Loss: {best_val_loss:.4f} | SROCC: {best_val_srocc:.4f} | PLCC: {best_val_plcc:.4f}\n"
            )
            print(log_msg)
            log_file.write(log_msg + "\n")
            log_file.flush()

    torch.save(model.state_dict(), last_weight_path)
    print(f"Saved last-epoch model at epoch {args.epochs}")
    log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str, default=r"D:\dataset\CSIQ\distorted_images", help='Dataset root path')
    parser.add_argument('--mask-dir', type=str, default=r"D:/dataset/mask/BiRefNet-general-epoch_244/CSIQrf",
                        help='Saliency mask root path')
    parser.add_argument('--data-path', type=str, default="D:/dataset/mos/CSIQ/mos_values.xlsx")
    parser.add_argument('--weight-name', type=str, default='SRINet', help='Output weight filename stem')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 01 or cpu)')
    parser.add_argument('--mse-weight', type=float, default=1.0, help='Weight for MSE loss')
    parser.add_argument('--nin-weight', type=float, default=1.0, help='Weight for NiN loss')
    parser.add_argument('--normalize-mos', type=int, default=1, help='Enable MOS normalization: 1 on, 0 off')
    parser.add_argument('--test-mode', type=int, default=0, help='0: full-reference (FR), 1: no-reference (NR)')

    args = parser.parse_args()

    args.normalize_mos = bool(args.normalize_mos)
    main(args)