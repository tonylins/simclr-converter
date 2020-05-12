# A PyTorch Converter for SimCLR Checkpoints

This is a converter to convert TensorFlow checkpoints provided in [SimCLR](https://github.com/google-research/simclr) repo to PyTorch format, to facilitate related research.

### Usage

1. Firstly, download and unzip the checkpoints from [SimCLR](https://github.com/google-research/simclr) repo, you will get 3 folders: `ResNet50_1x`, `ResNet50_2x`, and `ResNet50_4x`.

2. Run the following commands to convert the 3 checkpoints:

   ```bash
   python convert.py ResNet50_1x/model.ckpt-225206 resnet50-1x.pth
   python convert.py ResNet50_2x/model.ckpt-225206 resnet50-2x.pth
   python convert.py ResNet50_4x/model.ckpt-225206 resnet50-4x.pth
   ```

   You will get 3 PyTorch checkpoints, `resnet50-1x.pth`, `resnet50-2x.pth`, `resnet50-4x.pth`. The model definition is in `resent_wider.py`.

### Performance

To validate the correctness of the conversion, I tested the performance of the models using PyTorch standard augmentation on ImageNet (but **without normalization**, as the original TF models were not trained with normalization), using commands:

```
python eval.py /path/to/imagenet -a resnet50-1x/resnet50-2x/resnet50-4x
```

The performance is:

| Model          | TensorFlow Top-1 | PyTorch Top-1 |
| -------------- | ---------------- | ------------- |
| ResNet-50 (1x) | 69.1             | 68.9          |
| ResNet-50 (2x) | 74.2             | 74.1          |
| ResNet-50 (4x) | 76.6             | 76.4          |

There is a slight degradation, which should be due to the difference in data pre-processing (e.g., resize) in two frameworks.

