import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import GMMDataset
from networks import GMM, load_checkpoint, save_checkpoint
from visualize import save_images, save_visual
from utils import mkdir

def get_opt():
    parser = argparse.ArgumentParser(description='Run GMM model')
    parser.add_argument('--checkpoint', '-c', type=str, default='../result/GMM/epoch_99.pth', help='checkpoint to load')
    parser.add_argument('--data_root', '-d', type=str, default='data', help='path to data root directory')
    parser.add_argument('--out_dir', '-o', type=str, default='../result', help='path to result directory')
    parser.add_argument('--name', '-n', type=str, default='GMM', help='model name')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--gpu_id', '-g', type=str, default='0', help='GPU ID')
    parser.add_argument('--log_freq', type=int, default=100, help='log frequency')
    parser.add_argument('--fine_width', type=int, default=192)
    parser.add_argument('--fine_height', type=int, default=256)
    parser.add_argument('--radius', type=int, default=5)
    parser.add_argument('--grid_size', type=int, default=5, help='hyperparameter for the network')
    opt = parser.parse_args()
    return opt

def run(opt, model, data_loader, mode):
	if torch.cuda.is_available():
		device = torch.device('cuda:'+str(opt.gpu_id))  
	else:
		device = torch.device('cpu')
	model = model.to(device)

	warp_cloth_dir = os.path.join(opt.data_root, mode, 'warp-cloth')
	warp_cloth_mask_dir = os.path.join(opt.data_root, mode, 'warp-cloth-mask')
	visual_dir = os.path.join(opt.out_dir, opt.name, mode)
	dirs = [warp_cloth_dir, warp_cloth_mask_dir, visual_dir]
	for d in dirs:
		mkdir(d)
	
	data_iter = tqdm(data_loader, total=len(data_loader), bar_format='{l_bar}{r_bar}')
	for _data in data_iter:
		data = {}
		for key, value in _data.items():
			if not 'name' in key:
				data[key] = value.to(device) # Load data on GPU
			else:
				data[key] = value
		cloth = data['cloth']
		person = data['person']
		body_mask = data['body_mask']
		cloth_name = data['cloth_name']

		grid, _ = model(data['feature'], cloth)
		warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
		warped_cloth_mask = F.grid_sample(data['cloth_mask'], grid, padding_mode='zeros')
		save_images(warped_cloth, cloth_name, warp_cloth_dir) 
		save_images(warped_cloth_mask*2-1, cloth_name, warp_cloth_mask_dir) 
		if mode=='train': # No visuals
			continue

		warped_grid = F.grid_sample(data['grid'], grid, padding_mode='zeros')
		warped_person = body_mask*person + (1-body_mask)*warped_cloth
		gt = body_mask*person + (1-body_mask)*data['cloth_parse']
		visuals = [ [data['head'], data['shape'], data['pose']], 
				[cloth, warped_cloth, warped_grid], 
				[warped_person, gt, person]]
		save_visual(visuals, cloth_name, visual_dir)
        
def main():
	opt = get_opt()
	print(opt)

	model = GMM(opt)
	load_checkpoint(model, opt.checkpoint)
	model.cuda()
	model.eval()

	modes = ['train', 'val', 'test']
	for mode in modes:
		print('Run on {} data'.format(mode.upper()))
		dataset = GMMDataset(opt, mode=mode, data_list=mode+'_pairs.txt')
		dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.n_worker, shuffle=False)   
		with torch.no_grad():
			run(opt, model, dataloader, mode)
	print('Successfully completed')

if __name__=='__main__':
    main()
