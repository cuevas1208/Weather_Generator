# SynthesisAugmentations
This repository contains the code to generate weather effects synthetically using 

    1. CycleGAN-Turbo
        'clear_to_rainy', 'rainy_to_clear', 'night_to_day', 'day_to_night'
    2. CycliGAN 
        'clear2rainy', 'clear2snowy'

# run sample code 
```
git clone git@github.com:cuevas1208/Weather_Generator.git

cd Weather_Generator

pip install -r requirements.txt

python src/inference.py --content_imgs td-recon/data_versions/raw/cmi_sample5/ --model_name clear2snowy --model_path ../checkpoints

# model_name all to run all the models 
python src/inference.py --content_imgs td-recon/data_versions/raw/cmi_sample5/ --model_name all --model_path ../checkpoints
```

# Dataset
### 
you can download [bdd100k](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k/data)
Another dataset to consider Yosemite Summer ↔ Winter

# Dependencies
### install cuda version 11.3 and compatible pytorch 
sudo pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# sources
[One-Step Image Translation with
Text-to-Image Models](https://arxiv.org/pdf/2403.12036)
