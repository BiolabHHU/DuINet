# DuINet
## Installation
The model is built in PyTorch 1.12.0 and tested on Ubuntu 20.04 environment (Python3.8, CUDA11.0, cuDNN8.0).

For installing, follow these intructions
```
sudo apt-get install cmake build-essential libjpeg-dev libpng-dev
conda create -n pytorchgpu python=3.8
conda activate pytorchgpu
conda install pytorch=1.10 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```

### Gaussian denoising
#### Training 
1. Download train dataset: Color image: https://drive.google.com/file/d/1S1_QrP-fIXeFl5hYY193lr07KyZV8X8r/view Gray image: https://drive.google.com/file/d/1_miSC9_luoUHSqMG83kqrwYjNoEus6Bj/view

2. Generate image patches and place in '../patches/color/train_patch'
```
python generate_patches_SIDD.py --ps 48 --num_patches 300 --num_cores 10
```
3. Download validation images of CBSD68 and place them in `../patches/color/val_patch`
 
4. Install warmup scheduler

```
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

5. Train your model with default arguments by running

```
Gaussian_denoising/train.py

#add perceptual
Gaussian_denoising/train_perceptual.py
```

#### Test
- Download CBSD68, Kodak24, McMaster and place them in './datasets/'
- pretained blind model 'Gaussian_denoising/pretrained_models/Gaussian-denoising/'
- Run
```
test_denoising.py
```

### Real denoising

#### Training
1. Download the SIDD-Medium dataset from [here](https://www.eecs.yorku.ca/~kamel/sidd/dataset.php)
2. Generate image patches
```
python generate_patches_SIDD.py --ps 128 --num_patches 300 --num_cores 10
```
3. Download validation images of SIDD and place them in `../patches/real/val`

4. Train your model with default arguments by running

```
Real_denoising/train.py

## add perceptual
Real_denoising/train_perc_denoising.py
```

#### Testing on SIDD dataset
- Download sRGB [images] of SIDD and place them in '../datasets/sidd/'
-pretained model 'Real_denoising/pretrained_models/Real_denoising/'
- Run
```
Real_denoising/test_denoising_sidd.py
```


