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

## Cite
If you use this repository in your research, please include the paper in your references.

```
@inproceedings{Honda2019VitonGan,
  author = {Honda, Shion},
  title = {VITON-GAN: Virtual Try-on Image Generator Trained with Adversarial Loss},
  booktitle = {Proceedings of EuroGraphics},
  year = {2019},
}
```


## References
[1] BROCK A., DONAHUE J., SIMONYAN K.: Large scale GAN
training for high fidelity natural image synthesis. In International Conference on Learning Representations (2019).  
[2] CAO Z., SIMON T., WEI S.-E., SHEIKH Y.: Realtime multiperson 2d pose estimation using part affinity fields. In Proceedings of the
IEEE Conference on Computer Vision and Pattern Recognition (2017).  
[3] GONG K., LIANG X., ZHANG D., SHEN X., LIN L.: Look
into person: Self-supervised structure-sensitive learning and a new
benchmark for human parsing. In Proceedings of the IEEE Conference
on Computer Vision and Pattern Recognition (2017).  
[4] HAN X., WU Z., WU Z., YU R., DAVIS L. S.: Viton: An
image-based virtual try-on network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2018).  
[4] KARRAS T., LAINE S., AILA T.: A style-based generator architecture for generative adversarial networks. arXiv preprint
arXiv:1812.04948 (2018).  
[5] WANG B., ZHENG H., LIANG X., CHEN Y., LIN L., YANG
M.: Toward characteristic-preserving image-based virtual try-on network. In Proceedings of the European Conference on Computer Vision
(2018).