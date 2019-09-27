# GANomaly2D
### Improved ganomaly and ganomaly2d network structure
Note: The current software works well with PyTorch >=0.41 python>=3.5

Abstraction
---Ganomaly2d on part of the normal map area might hard to keep the latent feature consistent.Here, the network is improved. By adding resnet, adjusting some parameters makes the network easier to fit the sample data, but the training time needs to be lengthened. In addition, the network training is difficult. Some training skills need to be summarized.

Structure
---
The network structure is the 2D version of GANomaly2d [1]. Moreover, the structure of encoder and decoder is revised from the generator in CycleGAN [3]. We also use PatchGAN to replace the original architecture of discriminator.

Dataset should have the following directory & file structure:

Custom Dataset
├── test
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_n.png
│   ├── 1.abnormal
│   │   └── abnormal_tst_img_0.png
│   │   └── abnormal_tst_img_1.png
│   │   ...
│   │   └── abnormal_tst_img_m.png
├── train
│   ├── 0.normal
│   │   └── normal_tst_img_0.png
│   │   └── normal_tst_img_1.png
│   │   ...
│   │   └── normal_tst_img_t.png
```

Usage(example)
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
[1] SunnerLihttps://github.com/SunnerLi/GANomaly2D.
[2] S. Akcay, A. A. Abarghouei, and T. P. Breckon. Ganomaly: Semi-supervised anomaly detection via adversarial training. CoRR, abs/1805.06725, 2018.
[3] J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. arXiv preprint, 2017.      