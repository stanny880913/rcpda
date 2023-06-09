'''
   Generate radar inputs.
'''

import os
from os.path import join
import numpy as np
import argparse
from timeit import default_timer as timer
import re

import torch
from nuscenes.nuscenes import NuScenes

import _init_paths
from fuse_radar_own import merge_selected_radar, cal_depthMap_flow, radarFlow2uv

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group(0)) if match else -1

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_data', default='/media/stannyho/ssd/rc-pda/data_own',type=str)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--start_idx', type=int)
    parser.add_argument('--end_idx', type=int)
    
    args = parser.parse_args()  
    
    if args.dir_data == None:
        this_dir = os.path.dirname(__file__)
        args.dir_data = join(this_dir, '..', 'data')
    # dir_nuscenes = join(args.dir_data, 'nuscenes')
    start_idx = args.start_idx
    end_idx = args.end_idx
        
    downsample_scale = 4
    y_cutoff = 33

    # nusc = NuScenes(args.version, dataroot = dir_nuscenes, verbose=False)
    dir_data_out = join(args.dir_data, 'prepared_data')   
        
    sample_indices = torch.load(join(args.dir_data,'own_data_split.tar'))['own_sample_files']
    sample_indices.sort(key=extract_number)

    N_total = len(sample_indices)
    print('Total sample number:', N_total)
        
    if start_idx == None:
        start_idx = 0
        
    if end_idx == None or end_idx > N_total - 1:
        end_idx = N_total -1
    
    frm_range = [0,4]
    
    ct = 0

    for index,sample_idx in enumerate(sample_indices[start_idx: end_idx+1]):
    
        start = timer()

        # matrix = np.load(join(dir_data_out, '%05d_matrix.npz' % sample_idx))
        # K = matrix['K']
        
        K = [[700.63928,    0,  612.81753],
             [0,  706.29767,  363.39884],
             [0,    0,    1]]

        K = np.array(K)

        x1, y1, depth1, all_times1, x2, y2, depth2, all_times2, rcs, v_comp= merge_selected_radar(sample_idx, frm_range)
                
        depth_map1, flow, time_map1, rcs_map1, v_comp_map1 = cal_depthMap_flow(x1, y1, depth1, all_times1, x2, y2, depth2, all_times2, rcs, v_comp, downsample_scale=4, y_cutoff=33)        
        uv2 = radarFlow2uv(flow, K, depth_map1, downsample_scale=4, y_cutoff=33)
        
        radar_data = np.concatenate((depth_map1[..., None], uv2), axis=2) 
        
        np.save(join(dir_data_out, '%05d_radar.npy' % index), radar_data) 
        
        ct += 1
        print('compute depth %d/%d' % ( ct, end_idx - start_idx + 1 ) )
        
        end = timer()
        t = end-start     
        print('Time used: %.1f s' % t)
        
        