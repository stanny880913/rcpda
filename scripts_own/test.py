import argparse
import os
from os.path import join
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import ToTensor
import glob
import skimage.io as io
from skimage.transform import resize

import _init_paths
from pyramidNet import PyramidCNN

# 定义函数加载图像和点云数据
def load_image(file_path):
    image = io.imread(file_path)
    return image

def load_weights(args, model):
    f_checkpoint = join(args.dir_result, 'checkpoint.tar')        
    if os.path.isfile(f_checkpoint):
        print('load best model')        
        model.load_state_dict(torch.load(f_checkpoint)['state_dict_best'])
    else:
        sys.exit('No model found')

def init_env():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")    
    cudnn.benchmark = True if use_cuda else False
    return device

def downsample_im(im, downsample_scale, y_cutoff):
    # resize to [192,400,3]
    h_im, w_im = im.shape[0:2]        
    h_im = int(h_im / downsample_scale)
    w_im = int(w_im / downsample_scale) 
    
    im = resize(im, (h_im, w_im, 3), preserve_range=True, anti_aliasing=False)
    im = im.astype('uint8')
    im = im[y_cutoff:, ...]
    
    return im

def main(args):
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')

    if not args.dir_result:    
        args.dir_result = join(args.dir_data, 'train_result', 'depth_completion')

    # 设置模型和数据的文件夹路径
    img_dir = '/media/stannyho/ssd/rc-pda/data_own/img'
    pcd_dir = '/media/stannyho/ssd/rc-pda/data_own/pcd'

    device = init_env()

    model = PyramidCNN(args.nLevels, args.nPred, args.nPerBlock,
                    args.nChannels, args.inChannels, args.outChannels,
                    args.doRes, args.doBN, doELU=False,
                    predPix=False, predBoxes=False).to(device)
    load_weights(args, model)
    print("finished load model ")
    model.eval()

    downsample_scale = 4
    y_cutoff = 33

    # 读取图像数据并进行推断
    for img_path in glob.glob(os.path.join(img_dir, '*.png')):
        pcd_path = os.path.join(pcd_dir, os.path.basename(
            img_path).split('.')[0] + '.pcd')

        image = load_image(img_path)
        image = resize(image, (900, 1600, 3), preserve_range=True, anti_aliasing=False)

        image = downsample_im(image, downsample_scale, y_cutoff)
        image = ToTensor()(image)  # 转换为Tensor

        # 进行预测
        with torch.no_grad():
            prediction = model(image)

        # 保存预测结果
        result_path = os.path.join(
            '/media/stannyho/ssd/rc-pda/Inference', os.path.basename(img_path))
        prediction = prediction.squeeze().cpu().numpy()
        # 根据需要进行后处理或可视化

        # 保存预测结果为图像文件
        prediction_image = io.imsave(result_path, (prediction * 255).astype('uint8'))

    print("推断完成并保存结果。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()       
    parser.add_argument('--dir_data', type=str)
    parser.add_argument('--dir_result', type=str)

    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    
    parser.add_argument('--nLevels', type=int, default=5)
    parser.add_argument('--nPred', type=int, default=1)
    parser.add_argument('--nPerBlock', type=int, default=4)
    parser.add_argument('--nChannels', type=int, default=64)   
    parser.add_argument('--inChannels', type=int, default=10)
    parser.add_argument('--outChannels', type=int, default=1, help='number of output channel of network; automatically set to 1 if pred_task is foreground_seg')
    parser.add_argument('--doRes', type=bool, default=True)
    parser.add_argument('--doBN', type=bool, default=True) 
    
    parser.add_argument('--d_max', type=float, default=50)
    args = parser.parse_args()
    
    main(args)
