import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo
from utils.training_utils import build_transform
from utils.cyclicGan_utils import cg_transform, create_model
from pathlib import Path
from tqdm.auto import tqdm
import glob
import random


def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def inferece_model(model_name, content_images, args):
    output_dir = args.output_dir + '/' + model_name
    os.makedirs(output_dir, exist_ok=True)

    # initialize the model
    if model_name in ['clear2wet', 'clear2snowy']:

        model = create_model(model_name, args.model_path)
        model.eval()
        T_val = cg_transform()
    else:
        model = CycleGAN_Turbo(pretrained_name=model_name,
                               pretrained_path=args.model_path)
        model.eval()
        T_val = build_transform(args.image_prep)

    for img_path in tqdm(content_images):
        input_image = Image.open(img_path).convert('RGB')

        # translate the image
        inputs = T_val(input_image).unsqueeze(0)
        with torch.no_grad():
            output = model(inputs)

        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize(
            (input_image.width, input_image.height),
            Image.LANCZOS)

        # save the output image
        bname = os.path.basename(img_path)

        # concatenate original and output
        dst = get_concat_h(input_image, output_pil)
        dst.save(os.path.join(output_dir, bname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_imgs', type=str, required=True,
                        help='path to the input image')
    parser.add_argument('--model_name', type=str, default=None,
                        help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None,
                        help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='../output',
                        help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512',
                        help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None,
                        help='the direction of translation. None for '
                             'pretrained models, a2b or b2a for custom paths.')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name is None != args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_name is not None:
        assert args.direction is None, 'direction is not required when ' \
                                       'loading a pretrained model.'

    # Get List of all images
    content_images = glob.glob(args.content_imgs + '/**/*.png', recursive=True)
    random.shuffle(content_images)

    models = ['clear_to_rainy', 'rainy_to_clear',
              'night_to_day', 'day_to_night',
              'clear2wet', 'clear2snowy']
    for model_name in models:
        inferece_model(model_name, content_images[:5], args)
