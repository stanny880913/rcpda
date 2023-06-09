import argparse
import os
from os.path import join
import sys
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import numpy as np

import _init_paths
from pyramidNet import PyramidCNN
from data_loader_depth import init_data_loader


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


def prd_one_sample(model, test_loader, device, idx, args,result_path):
    with torch.no_grad():
        for ct, sample in enumerate(test_loader):
            if ct == idx:
                data_in = sample['data_in'].to(device)      
                
                prd = model(data_in)[0]
                
                im = data_in[0][0:3,...].permute(1,2,0).to('cpu').numpy().astype('uint8')
                d_radar = data_in[0][3,...].to('cpu').numpy()
                prd = prd[0][0].cpu().numpy()

                break
            
    d_max = args.d_max
    d_min = 1e-3
    
    prd = prd.clip(d_min, d_max)    
    
        # INFO predict and save result images
    # pre_result = prd

    # plt.close('all')   
    # result = plt.figure()
    # plt.imshow(pre_result, cmap='jet')
    # plt.axis('off')
    
    # print("Saving")
    # result.savefig(os.path.join(result_path,str(idx)) ,bbox_inches='tight', pad_inches = -0.1)
    # print("Save Image")

    plt.close('all')   
    
    plt.figure()
    plt.imshow(im)
    plt.show()
    
    plt.figure()
    plt.imshow(d_radar, cmap='jet')
    plt.title('Radar')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(prd, cmap='jet')
    plt.title('predict')
    plt.colorbar()
    plt.show()

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    
    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    
    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)
    
    mae = np.mean(np.abs(gt - pred))
       
    return silog, log10, mae, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3

def evaluate(model, test_loader, device, d_min=1e-3, d_max=70, eval_low_height = False):
    
    model.eval()
    errors = np.zeros(10)   
    
    with torch.no_grad():
        for sample in tqdm(test_loader, 'Evaluation'):
            data_in, gt = sample['data_in'].to(device), sample['d_lidar']              
            
            prd = ( torch.clamp(model(data_in)[0], min=d_min, max=d_max ) ).to('cpu').numpy()

            
            if eval_low_height:
                gt = gt * sample['msk_lh']
                
            gt = gt.numpy()             
            msk_valid = np.logical_and(gt>d_min, gt<d_max)
            
            errors += compute_errors(gt[msk_valid], prd[msk_valid])           
     
    errors /= len(test_loader) 
       
    print(' \n silog: %f, log10: %f, mae: %f, abs_rel: %f, sq_rel: %f, rmse: %f, rmse_log: %f, d1: %f, d2: %f, d3: %f' \
          % (errors[0], errors[1], errors[2], errors[3], errors[4], errors[5], errors[6], errors[7], errors[8], errors[9]))

def main(args):
    
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')

    if not args.dir_result:
        args.dir_result = join(args.dir_data, 'train_result', 'depth_completion')

    args.path_data_file = join(args.dir_data, 'prepared_data.h5') 
    
    # args.path_radar_file = join(args.dir_data, 'mer_2_30_5_0.5.h5')
    args.path_radar_file = '/media/stannyho/ssd/rc-pda/data/mer_2_30_5_0.5.h5'
                
    device = init_env()
    
    # train_loader = init_data_loader(args, 'train')
    # val_loader = init_data_loader(args, 'val')
    test_loader = init_data_loader(args, 'test')
    
        
    model = PyramidCNN(args.nLevels, args.nPred, args.nPerBlock, 
                        args.nChannels, args.inChannels, args.outChannels, 
                        args.doRes, args.doBN, doELU=False, 
                        predPix=False, predBoxes=False).to(device)
    
    load_weights(args, model)       
    model.eval()
    
    # INFO predict and save result image
    #idx = 1
    # train_len = len(train_loader)
    # val_len = len(val_loader)
    test_len = len(test_loader)

    # train_save_path = "/media/stannyho/ssd/rc-pda/inference/train"  
    # val_save_path = "/media/stannyho/ssd/rc-pda/inference/val"  
    test_save_path = "/media/stannyho/ssd/rc-pda/inference/test"  

    # if not os.path.exists(train_save_path):
    #     print("create folder")
    #     os.makedirs(train_save_path)

    # if not os.path.exists(val_save_path):
    #     print("create folder")
    #     os.makedirs(val_save_path)

    if not os.path.exists(test_save_path):
        print("create folder")
        os.makedirs(test_save_path)

    # print("ptrdict train!!!")
    # for idx in tqdm(range(train_len)):
    #     prd_one_sample(model, train_loader, device, idx, args,train_save_path)
    
    # print("ptrdict val!!!")
    # for idx in tqdm(range(val_len)):
    #     prd_one_sample(model, val_loader, device, idx, args,val_save_path)
        
    print("ptrdict test!!!")
    for idx in tqdm(range(test_len)):
        prd_one_sample(model, test_loader, device, idx, args,test_save_path)
        
    # evaluation
    evaluate(model, test_loader, device, d_min=1e-3, d_max=args.d_max, eval_low_height = False)

    print('\n Low height')
    evaluate(model, test_loader, device, d_min=1e-3, d_max=args.d_max, eval_low_height = True)
    

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
    
