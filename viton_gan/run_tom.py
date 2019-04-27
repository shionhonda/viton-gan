import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import TOMDataset
from networks import UnetGenerator, load_checkpoint, save_checkpoint
from visualize import save_images, save_visual
from utils import mkdir

def get_opt():
    parser = argparse.ArgumentParser(description='Run TOM model')
    parser.add_argument('--checkpoint', '-c', type=str, default='../result/TOM/gen_epoch_50.pth', help='checkpoint to load')
    parser.add_argument('--data_root', '-d', type=str, default='data', help='path to data root directory')
    parser.add_argument('--out_dir', '-o', type=str, default='../result', help='path to result directory')
    parser.add_argument('--name', '-n', type=str, default='TOM', help='model name')
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

	tryon_dir = os.path.join(opt.data_root, mode, 'tryon-person')
	mkdir(tryon_dir)
	visual_dir = os.path.join(opt.out_dir, opt.name, mode)
	mkdir(visual_dir)

	data_iter = tqdm(data_loader, total=len(data_loader), bar_format='{l_bar}{r_bar}')
	for _data in data_iter:
		data = {}
		for key, value in _data.items():
			if not 'name' in key:
				data[key] = value.to(device) # Load data on GPU
			else:
				data[key] = value
		cloth = data['cloth']
		cloth_mask = data['cloth_mask']
		person = data['person']
		cloth_name = data['cloth_name']

		outputs = model(torch.cat([data['feature'], cloth],1)) # (batch, channel, height, width)
		rendered_person, composition_mask = torch.split(outputs, 3,1)
		rendered_person = torch.tanh(rendered_person)
		composition_mask = torch.sigmoid(composition_mask)
		tryon_person = cloth*composition_mask + rendered_person*(1-composition_mask)
		visuals = [[data['head'], data['shape'], data['pose']], 
				[cloth, cloth_mask*2-1, composition_mask*2-1], 
				[rendered_person, tryon_person, person]]
		save_images(tryon_person, cloth_name, tryon_dir) 
		save_visual(visuals, cloth_name, visual_dir)
        
def main():
	opt = get_opt()
	print(opt)

	model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
	load_checkpoint(model, opt.checkpoint)
	model.cuda()
	model.eval()

	mode = 'test'
	print('Run on {} data'.format(mode.upper()))
	dataset = TOMDataset(opt, mode, data_list=mode+'_pairs.txt', train=False)
	dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.n_worker, shuffle=False)   
	with torch.no_grad():
		run(opt, model, dataloader, mode)
	print('Successfully completed')

if __name__=='__main__':
    main()
