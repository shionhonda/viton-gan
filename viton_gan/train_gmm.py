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
from visualize import board_add_images

class GMMTrainer:
    def __init__(self, model, dataloader_train, dataloader_test, gpu_id, log_freq, save_dir):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:'+str(gpu_id))  
        else:
             self.device = torch.device('cpu')
        self.model = model.to(self.device)

        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.optim = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.criterionL1 = nn.L1Loss()
        self.log_freq = log_freq
        print('Total Parameters:', sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        "Iterate 1 epoch over train data and return loss
        "
        return self.iteration(epoch, self.dataloader_train)

    def test(self, epoch):
        "Iterate 1 epoch over test data and return loss
        "
        return self.iteration(epoch, self.dataloader_test, train=False)

    def iteration(self, epoch, data_loader, train=True):
        data_iter = tqdm(enumerate(data_loader), desc='epoch: %d' % (epoch), total=len(data_loader), bar_format='{l_bar}{r_bar}')

        total_loss = 0.0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()} # Load data on GPU
            cloth = data['cloth']
            person = data['person']
            body_mask = data['body_mask']

            grid, _ = self.model(data['feature'], cloth)
            warped_cloth = F.grid_sample(cloth, grid, padding_mode='border')
            warped_grid = F.grid_sample(data['grid'], grid, padding_mode='zeros')
            warped_person = body_mask*person + (1-body_mask)*warped_cloth
            gt = body_mask*gt + (1-body_mask)*cloth
            visuals = [ [data['head'], data['shape'], data['pose']], 
                    [cloth, warped_cloth, warped_grid], 
                    [warped_person, gt, person]]

            loss = self.criterionL1(warped_person, gt)  + 0.5*self.criterionL1(warped_cloth, data['cloth_parse']) 
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            total_loss += loss.item()
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": total_loss/(i+1),
                "loss": loss.item()
            }
            if train and i%self.log_freq==0:
                data_iter.write(str(post_fix))
                board_add_images(visuals, epoch, i, save_dir)
        
        return total_loss/len(data_iter)
        

def get_opt():
    parser = argparse.ArgumentParser(description='Train GMM model')
    parser.add_argument('--n_epoch', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--data_root', '-d', type=str, default='data', help='path to data root directory')
    parser.add_argument('--out_dir', '-o', type=str, default='../result', help='path to result directory')
    parser.add_argument('--name', '-n', type=str, default='GMM', help='model name')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--gpu_id', '-g', type=str, default='0', help='GPU ID')
    parser.add_argument('--log_freq', type=int, default=200, help='log frequency')
    parser.add_argument('--grid_size', type=int, default=5, help='hyperparameter for the network')
    opt = parser.parse_args()
    return opt

def main():
    opt = get_opt()
    print(opt)

    prnit('Loading dataset')
    dataset_train = GMMDataset(opt.data_root, mode='train', data_list='train_pairs.txt')
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, num_workers=opt.n_worker)
    dataset_test = GMMDataset(opt.data_root, mode='test', data_list='test_pairs.txt')
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, num_workers=opt.n_worker)

    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)
    save_dir = os.path.join(opt.out_dir, opt.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_dir = os.path.join(opt.out_dir, 'log')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_name = os.path.join(log_dir, opt.name+'.csv')
    with open(log_name, 'w') as f:
        f.write('epoch,train_loss,test_loss\n')

    print('Building GMM model')
    model = GMM(opt)
    model.cuda()
    trainer = GMMTrainer(model, dataloader_train, dataloader_test, opt.gpu_id, save_dir)

    print('Start training GMM')
    for epoch in tqdm(range(args.n_epoch)):
        print('Epoch: {}'.format(epoch))
        loss = trainer.train(epoch)
        print('Train loss: {:.3f}'.format(loss))
        with open(log_name, 'a') as f:
            f.write('{},{:.3f}'.format(epoch, loss))
        save_checkpoint(model, os.path.join(save_dir, 'epoch_{:02}.pth'.format(epoch)))
        
        loss = trainer.test(epoch)
        print('Test loss: {:.3f}'.format(loss))
        with open(log_name, 'a') as f:
            f.write('{:.3f}\n'.format(loss))
    print('Finish training GMM')

if __name__=='__main__':
    main()