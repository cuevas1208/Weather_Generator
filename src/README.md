# SynthesisAugmentations
This repository contains the code to generate weather effects synthetically using 

    1. CycleGAN-Turbo
        'clear_to_rainy', 'rainy_to_clear', 'night_to_day', 'day_to_night'
    2. CycliGAN 
        'clear2rainy', 'clear2snowy'

### Weather Effect Generation using Analytical Method
CycleGAN does not
use existing text-to-image models and, as a result, generates artifacts in the outputs,
e.g., the sky regions in the day-to-night translation. In contrast, Instruct-pix2pix uses
a large text-to-image model but does not use the unpaired dataset. So, the Instructpix2pix 
outputs look unnatural and vastly different than the images in our datasets.

# Dataset
### 
you can download [bdd100k](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k/data)
Another dataset to consider Yosemite Summer ↔ Winter

# Dependencies
### install cuda version 11.3 and compatible pytorch 
sudo pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# sources
https://github.com/hgupta01/Weather_Effect_Generator/tree/main
