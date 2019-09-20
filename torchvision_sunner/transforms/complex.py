from torchvision_sunner.transforms.simple import ToFloat
from torchvision_sunner.transforms.base import OP
from torchvision_sunner.utils import INFO, DEPRECATE
from skimage import transform
import numpy as np
import torch

"""
    This script define some complex operations
    These kind of operations should conduct work iteratively (with inherit OP class)

    Author: SunnerLi
"""

class Rescale(OP):
    def __init__(self, output_size):
        """
            Rescale the tensor to the desired size
            This function only support for nearest-neighbor interpolation
            Since this mechanism can also deal with categorical data

            Arg:    output_size - The tuple (H, W)
        """
        self.output_size = output_size
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        
    def work(self, tensor):
        """
            Rescale the tensor
            If the tensor is not in the range of [-1, 1], we will do the normalization automatically

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The resized tensor
        """
        DEPRECATE(func_name = self.__class__.__name__, version = "19.3.15")

        # Normalize the tensor if needed
        mean, std = -1, -1
        min_v = np.min(tensor)
        max_v = np.max(tensor)
        if not (max_v <= 1 and min_v >= -1):
            mean = 0.5 * max_v + 0.5 * min_v
            std  = 0.5 * max_v - 0.5 * min_v
            # print(max_v, min_v, mean, std)
            tensor = (tensor - mean) / std

        # Permute
        if len(tensor.shape) == 3:
            tensor = np.transpose(tensor, (1, 2, 0))
        else:
            raise Exception("This case will be consider in the future...")

        # Work
        tensor = transform.resize(tensor, self.output_size, mode = 'constant', order = 0)

        # Permute back
        if len(tensor.shape) == 3:
            tensor = np.transpose(tensor, (2, 0, 1))
        else:
            raise Exception("This case will be consider in the future...")

        # De-normalize the tensor
        if mean != -1 and std != -1:
            tensor = tensor * std + mean
        return tensor    

class Normalize(OP):
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        """
            Normalize the tensor with given mean and standard deviation
            We recommand you to set mean as [0.5, 0.5, 0.5], and std as [0.5, 0.5, 0.5]
            Then the range will locate in [-1, 1]
            * Notice: If you didn't give mean and std, then we will follow the preprocessing of VGG
                      However, The range is NOT located in [-1, 1]

            Args:
                mean        - The mean of the result tensor
                std         - The standard deviation
        """
        self.mean = mean
        self.std  = std
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        INFO("*****************************************************************")
        INFO("* Notice: You should must call 'ToFloat' before normalization")
        INFO("*****************************************************************")
        if self.mean == [0.485, 0.456, 0.406] and self.std == [0.229, 0.224, 0.225]:
            INFO("* Notice: The result will NOT locate in [-1, 1]")

    def work(self, tensor):
        """
            Normalize the tensor

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The normalized tensor
        """
        if tensor.shape[0] != len(self.mean):
            raise Exception("The channel size should be {}, but the shape is {}".format(len(self.mean), tensor.shape))
        
        # Record the minimun and maximun value (in order to check if the function work normally)
        min_v, max_v = np.min(tensor), np.max(tensor)

        # Normalize with the given mean and std
        result = []
        for t, m, s in zip(tensor, self.mean, self.std):
            result.append((t - m) / s)
        tensor = np.asarray(result)

        # Check if the normalization can really work
        if self.mean != [1.0, 1.0, 1.0] and self.std != [1.0, 1.0, 1.0]:
            if np.min(tensor) == min_v and np.max(tensor) == max_v:
                raise Exception("Normalize can only work with float tensor",
                    "Try to call 'ToFloat()' before normalization")
        return tensor

class UnNormalize(OP):
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        """
            Unnormalize the tensor with given mean and standard deviation
            * Notice: If you didn't give mean and std, the function will assume that the original distribution locates in [-1, 1]

            Args:
                mean    - The mean of the result tensor
                std     - The standard deviation
        """
        self.mean = mean
        self.std = std
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")
        if self.mean == [0.485, 0.456, 0.406] and self.std == [0.229, 0.224, 0.225]:
            INFO("* Notice: The function assume that the normalize method is the same as VGG preprocessing")


    def work(self, tensor):
        """
            Un-normalize the tensor

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The un-normalized tensor
        """
        if tensor.shape[0] != len(self.mean):
            raise Exception("The rank format should be BCHW, but the shape is {}".format(tensor.shape))
        result = []
        for t, m, s in zip(tensor, self.mean, self.std):
            result.append(t * s + m)
        tensor = np.asarray(result)
        return tensor

class ToGray(OP):
    def __init__(self):
        """
            Change the tensor as the gray scale
            The function will turn the BCHW tensor into B1HW gray-scaled tensor
        """
        INFO("Applied << %15s >>" % self.__class__.__name__)
        INFO("* Notice: the rank format of input tensor should be 'BCHW'")

    def work(self, tensor):
        """
            Make the tensor into gray-scale

            Arg:    tensor  - The np.ndarray object. The tensor you want to deal with
            Ret:    The gray-scale tensor, and the rank of the tensor is B1HW
        """
        if tensor.shape[0] == 3:
            result = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
            result = np.expand_dims(result, axis = 0)
        elif tensor.shape[0] != 4:
            result = 0.299 * tensor[:, 0] + 0.587 * tensor[:, 1] + 0.114 * tensor[:, 2]
            result = np.expand_dims(result, axis = 1)
        else:
            raise Exception("The rank format should be BCHW, but the shape is {}".format(tensor.shape))
        return result