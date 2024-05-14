import os
import torch
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
from gan_networks import define_G
import torchvision.transforms as transforms


def __transforms2pil_resize(method):
    mapper = {transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
              transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
              transforms.InterpolationMode.NEAREST: Image.NEAREST,
              transforms.InterpolationMode.LANCZOS: Image.LANCZOS, }
    return mapper[method]


def __scale_width(img, target_size, crop_size,
                  method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def cg_transform(load_size=1280, crop_size=224,
                 method=transforms.InterpolationMode.BICUBIC):
    transform_list = [transforms.Lambda(
        lambda img: __scale_width(img, load_size, crop_size, method)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image,
                      torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[
            0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (
            1, 2,
            0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def create_model(model_name, path):
    # Creating model
    input_nc = 3
    output_nc = 3
    ngf = 64
    netG = 'resnet_9blocks'
    norm = 'instance'
    no_dropout = True
    init_type = 'normal'
    init_gain = 0.02
    gpu_ids = []

    netG_A = define_G(input_nc, output_nc, ngf, netG, norm, not no_dropout,
                      init_type, init_gain, gpu_ids)

    pretrained = os.path.join(path, model_name + '.pth')
    chkpntA = torch.load(pretrained)
    netG_A.load_state_dict(chkpntA)
    netG_A.eval()

    return netG_A
