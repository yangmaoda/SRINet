------

# SRINet: Saliency Region Interaction Network for No-Reference Image Quality Assessment

Official PyTorch implementation of the paper: **SRINet: Saliency region interaction network for no-reference image quality assessment**.https://www.sciencedirect.com/science/article/abs/pii/S0141938225003543

## Overview

Image Quality Assessment (IQA) methods often overlook the potential interaction between salient and background regions. To address this, we propose the Saliency Region Interaction Network (SRINet). Our model extracts spatial domain features using a ResNet-50 backbone and processes them through three core modules to accurately predict image quality:

- **Salient Object Chunk Embedding (SCE)**: Differentiates spatial features by partitioning them into salient and background regions based on a saliency mask.
- **Cross Interaction Enhancer (CIE)**: Captures complex dependencies between foreground and background regions using an Interaction Multi-head Self-Attention (IMSA) mechanism.
- **Channel Decoupling Spatial Attention (CDSA)**: Extracts and refines global contextual features along channel and spatial dimensions.

Finally, a Cross-Attention (CSA) module fuses these local interactive features with global representations for comprehensive quality prediction. The model is optimized jointly using Mean Squared Error (MSE) and Normalized Norm (NiN) loss, formulated as $L_{total} = L_{MSE} + L_{NiN}$.

## Requirements

- Python 3.8+
- PyTorch 2.1.0
- torchvision
- pandas
- numpy
- scipy
- Pillow
- tqdm
- tensorboard

## Data Preparation

SRINet requires both the original distorted images and their corresponding saliency masks. In our experiments, masks are generated using the pre-trained BiRefNet model.



You will need an Excel file (e.g., `mos_values.xlsx`) containing the image paths, splits, and ground-truth MOS (Mean Opinion Score) values for training and evaluation.

## Usage

### 1. Training

To train the SRINet model from scratch or fine-tune from existing weights, use `train.py`. The script applies conditional center cropping to adaptively handle images of different resolutions.

```
python train.py \
  --root-dir /path/to/distorted_images \
  --mask-dir /path/to/saliency_masks \
  --data-path /path/to/mos_values.xlsx \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.00005 \
  --device cuda:0
```

### 2. Evaluation (Inference with Ground Truth)

To evaluate a trained model on a specific validation or test split and calculate SROCC and PLCC metrics, use `inference.py`.

```
python inference.py \
  --root-dir /path/to/distorted_images \
  --mask-dir /path/to/saliency_masks \
  --data-path /path/to/mos_values.xlsx \
  --weights /path/to/saved_weights.pth \
```

*Note: Set `--split 1` for the validation set and `--split 2` for the test set.*

### 3. Prediction (Inference in the Wild)

If you want to predict the quality scores for a folder of images without ground-truth MOS data, use `predict.py`. This will output a CSV file containing the predicted scores.

```
python predict.py \
  --image-dir /path/to/wild_images \
  --mask-dir /path/to/wild_masks \
  --weights /path/to/saved_weights.pth \
  --output-dir ./prediction_results \
  --batch-size 1 \
  --device cuda:0
```

## Citation

If you find this code or our paper useful for your research, please consider citing:

```
@article{yang2026srinet,
  title={SRINet: Saliency region interaction network for no-reference image quality assessment},
  author={Yang, Maoda and Li, Qicheng and Guo, Muhan and Ren, Yuening and Zhang, Jun and Deng, Hongxia},
  journal={Displays},
  volume={92},
  pages={103317},
  year={2026},
  publisher={Elsevier}
}
```

------