from torchvision import transforms as torchvision_transforms
from torchvision_sunner.utils import INFO, DEPRECATE
from torchvision_sunner.constant import *
import torchvision_sunner.setting as setting

from torchvision import transforms
from PIL import Image
import numpy as np
import torch
import PIL

"""
    This script define some operation which are rather simple
    The operation only need to call function once (without inherit OP class)

    Author: SunnerLi
"""

class ToTensor():
    def __init__(self):
        """
            Change the tensor into torch.Tensor type
            However, if the input is PIL image, then the original ToTensor will be used

            For the range of output tensor:
                1. [0~255] => [0~1] if the image is PIL object
                2. otherwise the value range doesn't change
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)
        self.official_op_obj = transforms.ToTensor()

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor or other type. The tensor you want to deal with
        """
        if isinstance(tensor, PIL.Image.Image):
            # If the tensor is PIL image, then we use official torchvision ToTensor to deal with
            tensor = self.official_op_obj(tensor)

        elif isinstance(tensor, list):
            # If the tensor is list of PIL image, then we use official torchvision ToTensor iteratively to deal with
            tensor = torch.stack([self.official_op_obj(t) for t in tensor], 0)
        
        elif isinstance(tensor, np.ndarray):
            # Or we only transfer as TorchTensor
            tensor = torch.from_numpy(tensor)

        return tensor

class ToFloat():
    def __init__(self):
        """
            Change the tensor into torch.FloatTensor
        """        
        INFO("Applied << %15s >>" % self.__class__.__name__)

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        tensor = tensor.float()
        return tensor

class Transpose():
    def __init__(self, direction = BHWC2BCHW):
        """
            Transfer the rank of tensor into target one

            Arg:    direction   - The direction you want to do the transpose
        """        
        self.direction = direction
        if self.direction == BHWC2BCHW:
            INFO("Applied << %15s >>, The rank format is BCHW" % self.__class__.__name__)
        elif self.direction == BCHW2BHWC:
            INFO("Applied << %15s >>, The rank format is BHWC" % self.__class__.__name__)
        else:
            raise Exception("Unknown direction symbol: {}".format(self.direction))

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        if self.direction == BHWC2BCHW:
            tensor = tensor.transpose(-1, -2).transpose(-2, -3)
        else:
            tensor = tensor.transpose(-3, -2).transpose(-2, -1)
        return tensor

class RandomHorizontalFlip():
    def __init__(self, p = 0.5):
        """
            Flip the tensor toward horizontal direction randomly

            Arg:    p   - The random probability to filp the tensor
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        if p < 0.0 or p > 1.0:
            raise Exception("The parameter 'p' should in (0, 1], but get {}".format(p))
        self.p = p

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        if setting.random_seed < self.p:
            dim_idx = len(tensor.size()) - 1
            tensor_list = list(torch.split(tensor, 1, dim=dim_idx))
            tensor_list = list(reversed(tensor_list))
            tensor = torch.cat(tensor_list, dim_idx)
        return tensor

class RandomVerticalFlip():
    def __init__(self, p = 0.5):
        """
            Flip the tensor toward vertical direction randomly

            Arg:    p   - The random probability to filp the tensor
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        if p < 0.0 or p > 1.0:
            raise Exception("The parameter 'p' should in (0, 1], but get {}".format(p))
        self.p = p

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        if setting.random_seed < self.p:
            dim_idx = len(tensor.size()) - 2
            tensor_list = list(torch.split(tensor, 1, dim=dim_idx))
            tensor_list = list(reversed(tensor_list))
            tensor = torch.cat(tensor_list, dim_idx)
        return tensor

class GrayStack():
    def __init__(self, direction = BHW2BHWC):
        """
            Stack the gray-scale image for 3 times to become RGB image
            If the input is already RGB image, this function do nothing

            Arg:    direction   - The stack direction you want to conduct
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BHWC'")
        self.direction = direction

    def __call__(self, tensor):
        """
            Arg:    tensor - The torch.Tensor object. The tensor you want to deal with
        """
        if isinstance(tensor, np.ndarray):
            tensor = tensor.from_numpy(tensor)
            back_to_numpy = True
        else:
            back_to_numpy = False
        if self.direction == len(tensor.size()):
            if tensor.size(-1) == 1:
                tensor = torch.cat([tensor, tensor, tensor], -1)
        elif self.direction == (len(tensor.size()) + 1):
            tensor = torch.stack([tensor, tensor, tensor], -1)
        if back_to_numpy:
            tensor = tensor.cpu().numpy()
        return tensor

class Resize():
    def __init__(self, output_size):
        """
            Rescale the tensor to the desired size
            This function only support for nearest-neighbor interpolation
            Since this mechanism can also deal with categorical data

            Arg:    output_size - The tuple (H, W)
        """
        self.output_size = output_size
        self.op = torchvision_transforms.Resize(output_size, interpolation = Image.NEAREST)
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")

    def __call__(self, tensor):
        if isinstance(tensor, Image.Image):
            tensor = self.op(tensor)
        elif isinstance(tensor, list):
            tensor = [self.op(t) for t in tensor]
        return tensor