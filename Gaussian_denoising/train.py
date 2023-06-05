import os
from config import Config

opt = Config('train.yaml')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# gpus = ','.join([str(i) for i in opt.GPU])
# os.environ["CUDA_VISIBLE_DEVICES"] = gpus


import torch

torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np

import utils
# from dataloaders.data_rgb import get_training_data, get_validation_data
from dataloaders.data_rgb import get_train_clean_data, get_val_clean_data
from pdb import set_trace as stx

from network.DuINet import DuINet
from losses import MSELoss

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

result_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'results', session)
model_dir = os.path.join(opt.TRAINING.SAVE_DIR, mode, 'models', session)

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
val_dir = opt.TRAINING.VAL_DIR
save_images = opt.TRAINING.SAVE_IMAGES

noise_blind = opt.TRAINING.NOISE_BLIND
noise_level = opt.TRAINING.NOISE_LEVEL

######### Model ###########
model_restoration = DuINet()
model_restoration.cuda()

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

new_lr = opt.OPTIM.LR_INITIAL

noiseL_B = [0, 55]

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)

######### Scheduler ###########
if opt.TRAINING.WARMUP:
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                            eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

if len(device_ids) > 1:
    model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

######### Loss ###########
criterion = nn.MSELoss(reduction='sum').cuda()

######### DataLoaders ###########
img_options_train = {'patch_size': opt.TRAINING.TRAIN_PS}

# train_dataset = get_training_data(train_dir, img_options_train)
train_dataset = get_train_clean_data(train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16,
                          drop_last=False)

val_dataset = get_val_clean_data(val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8, drop_last=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

mixup = utils.MixUp_AUG()
best_psnr = 0
best_epoch = 0
best_iter = 0

eval_now = len(train_loader) // 4 - 1
print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    for i, data in enumerate(tqdm(train_loader), 0):

        # zero_grad
        for param in model_restoration.parameters():
            param.grad = None

        target = data[0]

        ### add noise ###
        if noise_blind:
            noise = torch.zeros(target.size())
            stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
            for n in range(noise.size()[0]):
                sizeN = noise[0, :, :, :].size()
                noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
        else:
            noise = torch.FloatTensor(target.size()).normal_(mean=0, std=noise_level / 255.)

        ### over ###

        input_ = target + noise

        target = target.cuda()
        input_ = input_.cuda()

        if epoch > 5:
            target, input_ = mixup.aug(target, input_)

        restored = model_restoration(input_)
        restored = torch.clamp(restored, 0, 1)

        loss = criterion(restored, target)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        #### Evaluation ####
        if i % eval_now == 0 and i > 0:
            if save_images:
                utils.mkdir(result_dir + '%d/%d' % (epoch, i))
            model_restoration.eval()
            with torch.no_grad():
                psnr_val_rgb = []
                for ii, data_val in enumerate((val_loader), 0):
                    target = data_val[0]
                    filenames = data_val[1]

                    ### add noise ###
                    if noise_blind:
                        noise = torch.zeros(target.size())
                        stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                        for n in range(noise.size()[0]):
                            sizeN = noise[0, :, :, :].size()
                            noise[n, :, :, :] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n] / 255.)
                    else:
                        noise = torch.FloatTensor(target.size()).normal_(mean=0, std=noise_level / 255.)

                    ### over ###

                    input_ = target + noise

                    target = target.cuda()
                    input_ = input_.cuda()

                    restored = model_restoration(input_)
                    restored = torch.clamp(restored, 0, 1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))

                    if save_images:
                        target = target.permute(0, 2, 3, 1).cpu().detach().numpy()
                        input_ = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
                        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                        for batch in range(input_.shape[0]):
                            temp = np.concatenate((input_[batch] * 255, restored[batch] * 255, target[batch] * 255),
                                                  axis=1)
                            utils.save_img(os.path.join(result_dir, str(epoch), str(i), filenames[batch][:-4] + '.jpg'),
                                           temp.astype(np.uint8))

                psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)

                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model_restoration.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }, os.path.join(model_dir, "model_best.pth"))

                print(
                    "[Ep %d it %d\t PSNR CBSD68: %.4f\t] ----  [best_Ep_CBSD68 %d best_it_CBSD68 %d Best_PSNR_CBSD68 %.4f] " % (
                    epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))

            model_restoration.train()

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

