# VITON-GAN  
Implementation of the paper "VITON-GAN: Virtual Try-on Image Generator Trained with Adversarial Loss" by Shion Honda.  
https://diglib.eg.org/handle/10.2312/egp20191043

## Installation
### Prerequisites

```
PIL
PyTorch
TorchVision
tqdm
```

In addition, you need OpenPose and Look Into Person (LIP) to get keypoints and segmentation of the human body.

### Download repository
```
$ git clone https://github.com/shionhonda/viton-gan
```
### Trained model
You can get trained model [here](https://drive.google.com/drive/folders/1Zcc55A1w6bUNG4cMYnmsIPvOJWvBsmDH?usp=sharing).
## Usage
VITON-GAN requires the keypoints from OpenPose and segmentation labels from Look Into Person.   
First, prepare the following directories in viton-gan/viton_gan/data:

- cloth
- cloth mask
- person
- person-parse
- pose

Second, prepare a file that makes pairs of clothing and human. For example, `test_pairs.txt`:

```
000001_0.jpg 001744_1.jpg
000010_0.jpg 004325_1.jpg
.
.
.
```

You can find more information here: https://github.com/sergeywong/cp-vton

After preparing the data and the list, run the following command:

```
$ python train_gmm.py
$ python run_gmm.py # warp clothing so that it fit on the body
$ python train_tom.py
$ python run_gmm.py # generate virtual try-on image
```

## Cite
If you use this repository in your research, please include the paper in your references.

```
@inproceedings {p.20191043,
booktitle = {Eurographics 2019 - Posters},
editor = {Fusiello, Andrea and Bimber, Oliver},
title = {{VITON-GAN: Virtual Try-on Image Generator Trained with Adversarial Loss}},
author = {Honda, Shion},
year = {2019},
publisher = {The Eurographics Association},
ISSN = {1017-4656},
DOI = {10.2312/egp.20191043}
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
[5] KARRAS T., LAINE S., AILA T.: A style-based generator architecture for generative adversarial networks. arXiv preprint
arXiv:1812.04948 (2018).  
[6] WANG B., ZHENG H., LIANG X., CHEN Y., LIN L., YANG
M.: Toward characteristic-preserving image-based virtual try-on network. In Proceedings of the European Conference on Computer Vision
(2018).
