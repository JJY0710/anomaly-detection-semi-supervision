from lib.visualize import visualizeEncoderDecoder
from lib.model import GANomaly2D
from parse.parse import parse_args 

import torchvision_sunner.transforms as sunnerTransforms
import torchvision_sunner.data as sunnerData
import torchvision.transforms as transforms

from tqdm import tqdm
import argparse
import torch
import os

"""
    This script defines the training procedure of GANomaly2D

    Author: SunnerLi
"""

def train(args): 
    """
        This function define the training process
        
        Arg:    args    (napmespace) - The arguments
    """
    # Create the data loader
    loader = sunnerData.DataLoader(
        dataset = sunnerData.ImageDataset(
            root = [[args.train]],
            transforms = transforms.Compose([
                 
#                 transforms.RandomCrop(720,720)
#                 transforms.RandomRotation(45)
#                 transforms.RandomHorizontalFlip(),         
#                 transforms.ColorJitter(brightness=0.5, contrast=0.5),
                

                sunnerTransforms.Resize(output_size = (args.H, args.W)),
                #transforms.RandomCrop(512,512)
                sunnerTransforms.ToTensor(),
                sunnerTransforms.ToFloat(),
                # sunnerTransforms.Transpose(),
                sunnerTransforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        ), batch_size = args.batch_size, shuffle = True, num_workers = 2
    )
    loader = sunnerData.IterationLoader(loader, max_iter = args.n_iter)

    # Create the model
    model = GANomaly2D(r = args.r, device = args.device)
    model.IO(args.resume, direction = 'load')
    model.train()
        
    # Train!
    bar = tqdm(loader)
    for i, (normal_img,) in enumerate(bar):
        model.forward(normal_img)
        model.backward()
        loss_G, loss_D = model.getLoss()
        bar.set_description("Loss_G: " + str(loss_G) + " loss_D: " + str(loss_D))
        bar.refresh()
        if i % args.record_iter == 0:
            model.eval()
            with torch.no_grad():
                z, z_ = model.forward(normal_img)
                img, img_ = model.getImg()
                visualizeEncoderDecoder(img, img_, z, z_,i)
            model.train()
            model.IO(args.det, direction = 'save')
    model.IO(args.det, direction = 'save')

if __name__ == '__main__':
    args = parse_args(phase = 'train')
    train(args)
