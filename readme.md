# GANomaly2D 
### The Extended Version of GANomaly with Spatial Clue

[![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)]()
[![Packagist](https://img.shields.io/badge/Python-3.5.0-blue.svg)]()
[![Packagist](https://img.shields.io/badge/Torchvision_sunner-19.4.15-green.svg)]()


Abstraction
---
Anomaly item detection is a critical issue in computer vision. Even though there are some research to solve this problem toward whole patch, these methods doesn't contain the spatial information. The computation is time-consuming if the methods are deployed into practical scenario and check the abnormality patch by patch. In this repository, we purposed **GANomaly2D** to solve the anomaly item recognition problem while preserving the localization information. While the anomaly item occurs in the frame, the anomaly score map will reflect the region rather than only predicting the frame is abnormal or not.    

Install
---
We use `Torchvision_sunner` to deal with data loading. You should install the package from [here](https://github.com/SunnerLi/Torchvision_sunner).    

Structure
---
The network structure is the 2D version of GANomaly [1]. Moreover, the structure of encoder and decoder is revised from the generator in CycleGAN [2]. We also use PatchGAN to replace the original architecture of discriminator.

Dataset

should have the following directory & file structure:
```
Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png

```

Usage
---
* Train:
```
python3 train.py --train dataset/normal/ --n_iter 2000 --record 5 --batch_size 8 --r 2
```
* Demo:
```
python3 demo.py --demo dataset/abnormal/ --batch_size 1 --r 2
```

Reference
---
[0] https://github.com/SunnerLi/GANomaly2D
[1] S. Akcay, A. A. Abarghouei, and T. P. Breckon. Ganomaly: Semi-supervised anomaly detection via adversarial training. CoRR, abs/1805.06725, 2018.    
[2] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint, 2017.    
