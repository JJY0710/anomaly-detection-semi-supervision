from torchvision_sunner.transforms.simple import Transpose
from torchvision_sunner.constant import BCHW2BHWC

from skimage import transform
from skimage import io 
import numpy as np
import torch

"""
    This script define the transform function which can be called directly

    Author: SunnerLi
"""

channel_op = None       # Define the channel op which will be used in 'asImg' function

def asImg(tensor, size = None):
    """
        This function provides fast approach to transfer the image into numpy.ndarray
        This function only accept the output from sigmoid layer or hyperbolic tangent output

        Arg:    tensor  - The torch.Variable object, the rank format is BCHW or BHW
                size    - The tuple object, and the format is (height, width)
        Ret:    The numpy image, the rank format is BHWC
    """
    global channel_op
    result = tensor.detach()

    # 1. Judge the rank first
    if len(tensor.size()) == 3:
        result = torch.stack([result, result, result], 1)

    # 2. Judge the range of tensor (sigmoid output or hyperbolic tangent output)
    min_v = torch.min(result).cpu().data.numpy()
    max_v = torch.max(result).cpu().data.numpy()
    if max_v > 1.0 or min_v < -1.0:
        raise Exception('tensor value out of range...\t range is [' + str(min_v) + ' ~ ' + str(max_v))
    if min_v < 0:
        result = (result + 1) / 2

    # 3. Define the BCHW -> BHWC operation
    if channel_op is None:
        channel_op = Transpose(BCHW2BHWC)

    # 3. Rest               
    result = channel_op(result)
    result = result.cpu().data.numpy()
    if size is not None:
        result_list = []
        for img in result:
            result_list.append(transform.resize(img, (size[0], size[1]), mode = 'constant', order = 0) * 255)
        result = np.stack(result_list, axis = 0)
    else:
        result *= 255.
    result = result.astype(np.uint8)
    return result

def show(tensor, row = 1, column = 1, title = "show_window", sec = -1):
    """
        This function can show the image immediately
        You should be aware that the product of row and column should match the batch size
        If the criterion is not match, this function only show the first image

        Arg:    tensor  - The tensor you want to show
                row     - The number of row you want to extend
                column  - The number of column you want to extend
                title   - The title string in the demonstration image
                sec     - The time you want to pause. Set -1 if you want to wait until press any key
    """
    # Order the demonstration image
    if row * column == tensor.size(0):
        arr = asImg(tensor)
        demonstrate_img = None
        for i in range(row):
            single_row = None
            for j in range(column):
                single_row = arr[i * column + j] if single_row is None else np.concatenate((single_row, arr[i * column + j]), 1)
            single_row = np.array(single_row)
            demonstrate_img = single_row if demonstrate_img is None else np.concatenate((demonstrate_img, single_row), 0)
    else:
        demonstrate_img = asImg(tensor)[0]

    # Show the result
    import cv2
    cv2.imshow(title, demonstrate_img[:, :, ::-1])
    if sec > 0:
        cv2.waitKey(sec)
    else:
        cv2.waitKey()