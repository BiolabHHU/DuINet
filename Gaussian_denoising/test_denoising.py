import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from network.DuINet import DuINet
from dataloaders.data_rgb import get_test_clean_data
import utils
from skimage import img_as_ubyte
import utils.utils_model as utils_model
import pandas as pd
parser = argparse.ArgumentParser(description='denoising evaluation on the testsets')
parser.add_argument('--input_dir', default='../datasets/Kodak24/',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/IFENet/Kodak24/50/',
                    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/Gaussian-denoising/blind.pth',
                    type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', default='True', help='Save denoised images in result directory')
parser.add_argument("--test_noiseL", type=float, default=50, help='noise level used on test set')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_test_clean_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=0, drop_last=False)

model_restoration = DuINet()

utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

model_restoration =nn.DataParallel(model_restoration)

model_restoration.eval()

x8 = 0
with torch.no_grad():
    psnr_val_rgb = []
    ssim_val_rgb = []
    name = []

    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_gt = data_test[0]
        filenames = data_test[1]
        name.append(filenames)
        ### ad noise ###
        noise = torch.FloatTensor(rgb_gt.size()).normal_(mean=0, std=args.test_noiseL / 255.)
        ### over ###

        rgb_noisy = rgb_gt + noise

        rgb_gt = rgb_gt.cuda()
        rgb_noisy = rgb_noisy.cuda()
        if not x8 and rgb_noisy.size(2) // 8 == 0 and rgb_noisy.size(3) // 8 == 0:
            rgb_restored = model_restoration(rgb_noisy)
        elif not x8 and (rgb_noisy.size(2) // 8 != 0 or rgb_noisy.size(3) // 8 != 0):
            rgb_restored = utils_model.test_mode(model_restoration, rgb_noisy, refield=64, mode=5)
        elif x8:
            rgb_restored = utils_model.test_mode(model_restoration, rgb_noisy, mode=3)

        rgb_restored = torch.clamp(rgb_restored ,0 ,1)
        psnr = utils.batch_PSNR(rgb_restored, rgb_gt, 1.)
        ssim = utils.batch_SSIM(rgb_restored, rgb_gt)
        psnr_val_rgb.append(psnr.cpu())
        ssim_val_rgb.append(ssim)

        # print("%s PSNR %f ssim %f " % (filenames, psnr, ssim))

        rgb_gt = rgb_gt.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_noisy = rgb_noisy.permute(0, 2, 3, 1).cpu().detach().numpy()
        rgb_restored = rgb_restored.permute(0, 2, 3, 1).cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(rgb_gt)):
                denoised_img = img_as_ubyte(rgb_restored[batch])
                utils.save_img(args.result_dir + filenames[batch][:-4] + '.png', denoised_img)

psnr_val_rgb_test = sum(psnr_val_rgb ) /len(psnr_val_rgb)
print("PSNR: %.2f " %(psnr_val_rgb_test))

ssim_val_rgb_test = sum(ssim_val_rgb ) /len(ssim_val_rgb)
print("SSIM: %.4f " %(ssim_val_rgb_test))

name.append('Average')
psnr_val_rgb.append(psnr_val_rgb_test)
ssim_val_rgb.append(ssim_val_rgb_test)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('max_colwidth', 100)  # value length 100
df = pd.DataFrame({'name': np.array(name), 'psnr': np.array(psnr_val_rgb), 'ssim': np.array(ssim_val_rgb)})
with open(args.result_dir + 'results.txt', "w+") as f:
    f.write(str(df))
