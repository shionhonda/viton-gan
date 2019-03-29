# VITON-GAN  
Implementation of the paper "VITON-GAN: Virtual Try-on Image Generator Trained with Adversarial Loss" by Shion Honda.

## Installation
### Prerequisites

```
PIL
PyTorch
TorchVision
tqdm
```

In addition, you need OpenPose and Look Into Person (LIP) to get keypoints and segmentation of the human body.

### Download
```
$ git clone https://github.com/shionhonda/viton-gan
```

## Usage

```
$ python train_gmm.py
$ python run_gmm.py
$ python train_tom.py
```