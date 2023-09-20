import argparse
import math
import os

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
from tqdm import tqdm

from Model import Base_Model
from Model_util import padding_image
from make import getTxt
from test_dataset import dehaze_test_dataset
from utils_test import to_psnr, to_ssim_skimage


# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='Siamese Dehaze Network')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--model_save_dir', type=str, default='./output_result')
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--gpus', default='0,1,2,3', type=str)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--test_dir', type=str, default=' ')
parser.add_argument('--test_name', type=str, default='hazy,clean')
parser.add_argument('--num', type=str, default='9999999', help='')
parser.add_argument('--use_bn', action='store_true', help='if bs>8 please use bn')
parser.add_argument("--type", default=-1, type=int, help="choose a type 012345")
args = parser.parse_args()

predict_result = os.path.join(args.model_save_dir, 'result_pic')
test_batch_size = args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)

if not os.path.exists(predict_result):
    os.makedirs(predict_result)

output_dir = os.path.join(args.model_save_dir, '')

# --- Gpu device --- #
device_ids = [int(i) for i in list(filter(str.isdigit, args.gpus))]
print('use gpus ->', args.gpus)
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
if args.use_bn:
    print('we are using BatchNorm')
else:
    print('we are using InstanceNorm')
#SDN = Base_Model(bn=args.use_bn)
SDN = Base_Model(3,3)
print('SDN parameters:', sum(param.numel() for param in SDN.parameters()))
# --- Multi-GPU --- #
SDN = SDN.to(device)
SDN = torch.nn.DataParallel(SDN, device_ids=device_ids)

tag = 'else'
if args.type == 0:
    args.train_dir = '/data2/ghy/ghy_net/data/thin/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data2/ghy/ghy_net/data/thin/train/"
    args.test_name = 'hazy,clean'
    tag = 'thin'
elif args.type == 1:
    args.train_dir = '/data2/ghy/ghy_net/data/mode/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data2/ghy/ghy_net/data/mode/test/"
    args.test_name = 'hazy,clean'
    tag = 'moderate'
elif args.type == 2:
    args.train_dir = '/data2/ghy/ghy_net/data/thick/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data2/ghy/ghy_net/data/thick/test/"
    args.test_name = 'hazy,clean'
    tag = 'thick'
elif args.type == 3:
    args.train_dir = '/data2/ghy/ghy_net/data/RICE1/train/'
    args.train_name = 'hazy,clean'
    args.test_dir = "/data2/ghy/ghy_net/data/RICE1/test/"
    args.test_name = 'hazy,clean'
    tag = 'rice1'


print('We are testing datasets: ', tag)
getTxt(None, None, args.test_dir, args.test_name)
test_hazy, test_gt = args.test_name.split(',')
if not os.path.exists(os.path.join(predict_result, 'hazy')):
    os.makedirs(os.path.join(predict_result, 'hazy'))
os.system('cp -r {}/* {}'.format(os.path.join(args.test_dir, test_hazy), os.path.join(predict_result, 'hazy')))

if not os.path.exists(os.path.join(predict_result, 'clean')):
    os.makedirs(os.path.join(predict_result, 'clean'))
os.system('cp -r {}/* {}'.format(os.path.join(args.test_dir, test_gt), os.path.join(predict_result, 'clean')))

predict_dir = os.path.join(predict_result, 'predict')
if not os.path.exists(predict_dir):
    os.makedirs(predict_dir)

test_dataset = dehaze_test_dataset(args.test_dir, args.test_name, tag=tag)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

# --- Load the network weight --- #
if args.num != '9999999':
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    name = [i for i in pkl_list if 'epoch_' + str(args.num) + '_' in i][0]
    SDN.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(args.num))
    start_epoch = int(args.num) + 1
else:
    pkl_list = [i for i in os.listdir(args.model_save_dir) if '.pkl' in i]
    num = sorted([int(i.split('.')[0].split('_')[1]) for i in pkl_list])[-1]
    name = [i for i in pkl_list if 'epoch_' + str(num) + '_' in i][0]
    SDN.load_state_dict(
        torch.load(os.path.join(args.model_save_dir, name),
                   map_location="cuda:{}".format(device_ids[0])))
    print('--- {} epoch weight loaded ---'.format(num))

test_txt = open(os.path.join(predict_result, 'result.txt'), 'w+')
# --- Strat testing --- #
with torch.no_grad():
    img_list = []
    psnr_list = []
    ssim_list = []
    SDN.eval()
    imsave_dir = output_dir
    if not os.path.exists(imsave_dir):
        os.makedirs(imsave_dir)
    for (hazy, clean, name) in test_loader:
        hazy = hazy.to(device)
        clean = clean.to(device)
        h, w = hazy.shape[2], hazy.shape[3]
        max_h = int(math.ceil(h / 4)) * 4
        max_w = int(math.ceil(w / 4)) * 4
        hazy, ori_left, ori_right, ori_top, ori_down = padding_image(hazy, max_h, max_w)

        frame_out = SDN(hazy)

        frame_out = frame_out.data[:, :, ori_top:ori_down, ori_left:ori_right]
        

        imwrite(frame_out, os.path.join(predict_dir, name[0]), range=(0, 1))

        psnr_list.extend(to_psnr(frame_out, clean))

        ssim_list.extend(to_ssim_skimage(frame_out, clean))
        print(name[0], to_psnr(frame_out, clean), to_ssim_skimage(frame_out, clean))
        tmp = test_txt.writelines(name[0] + '->\tpsnr: ' + str(to_psnr(frame_out, clean)[0]) + '\tssim:' + str(
            to_ssim_skimage(frame_out, clean)[0]) + '\n')
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    tmp = test_txt.writelines(
        tag + 'datasets ==>>\tpsnr:' + str(avr_psnr) + '\tssim:' + str(avr_ssim) + '\n')
    print('dehazed', avr_psnr, avr_ssim)

test_txt.close()
