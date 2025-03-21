# SigLIP Vision Transformer


![Vision Transformer Architecture]([https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTboIz7CcWb4yVUtQc5OxO1ETJYgkxCdxNIDA&s](https://viso.ai/wp-content/uploads/2021/09/vision-transformer-vit.png))


## Overview

This repository contains an implementation of the **SigLIP Vision Transformer** (ViT), a model designed for efficient image representation learning. The model leverages **self-attention mechanisms** to process images as sequences of patches, making it highly effective for various vision tasks such as classification, segmentation, and object detection.

## Features

- **Multi-Head Self-Attention:** Implements attention mechanisms to capture long-range dependencies.
- **Token Embeddings:** Converts image patches into meaningful representations.
- **Positional Encoding:** Adds spatial context to image patches.
- **MLP Head:** Fully connected layers for final predictions.
- **Modular Design:** Easy to modify and extend different components.

## Transformer Encoder Components

The Vision Transformer encoder consists of four main components:

1. **Embedding Patches** - The input image is divided into smaller patches, which are then embedded into a feature space, allowing the transformer to process them as sequential data.
2. **Multi-Head Attention** - This mechanism enables the model to capture long-range dependencies by computing attention scores for different patches.
3. **Encoding Connections** - Residual connections and normalization layers are used to stabilize training.
4. **Multi-Layer Perceptron (MLP)** - Fully connected layers apply transformations to the processed representations, helping in classification or other vision tasks.

## File Structure

- `attentionModule.py` - Implements the self-attention.
- `embeddingModule.py` - Handles image patch embedding and positional embedding.
- `encodingModule.py` - Adds Residual Connection and Normalization Layers between the MLP and Multi-head attention.
- `MLP.py` - Defines the multi-layer perceptron head.
- `mainModule.py` - Combines all modules to create the full Vision Transformer model.
- `testingCode.py` - Checks if state output keys match to pre-trained SigLip Model by Hugging Face Transformers Module.

## Installation

Ensure you have Python 3.8+ installed. Then, install the required dependencies:

```sh
pip install numpy torch torchvision matplotlib
```

## Usage

To run the Vision Transformer model on an image dataset, use the following command:

```sh
python mainModule.py --dataset <path_to_dataset>
```

Example:

```sh
python mainModule.py --dataset ./data/images
```

## Example Code

```python
from mainModule import VisionTransformer
model = VisionTransformer(image_size=224, patch_size=16, num_classes=10)
print(model)
```

## References

- [Dosovitskiy et al., 2020: "An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929)
- [SigLIP: Scaling Vision-Language Models with Sigmoid Loss](https://arxiv.org/abs/2303.15343)

--

!(https://pbs.twimg.com/media/F31g75LXkAAVdEk.jpg:large)


