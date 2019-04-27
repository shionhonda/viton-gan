import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import TOMDataset
from networks import UnetGenerator, VGGLoss, load_checkpoint, save_checkpoint, NLayerDiscriminator
from visualize import board_add_images
from utils import mkdir

class TOMTrainer:
    def __init__(self, gen, dis, dataloader_train, dataloader_val, gpu_id, log_freq, save_dir, n_step):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:'+str(gpu_id))  
        else:
             self.device = torch.device('cpu')
        self.gen = gen.to(self.device)
        self.dis = dis.to(self.device)

        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optim_g = torch.optim.Adam(gen.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.optim_d = torch.optim.Adam(dis.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.criterionL1 = nn.L1Loss()
        self.criterionVGG = VGGLoss()
        self.criterionAdv = torch.nn.BCELoss()
        self.log_freq = log_freq
        self.save_dir = save_dir
        self.n_step = n_step
        self.step = 0
        print('Generator Parameters:', sum([p.nelement() for p in self.gen.parameters()]))
        print('Discriminator Parameters:', sum([p.nelement() for p in self.dis.parameters()]))

    def train(self, epoch):
        """Iterate 1 epoch over train data and return loss
        """
        return self.iteration(epoch, self.dataloader_train)

    def val(self, epoch):
        """Iterate 1 epoch over validation data and return loss
        """
        return self.iteration(epoch, self.dataloader_val, train=False)

    def iteration(self, epoch, data_loader, train=True):
        data_iter = tqdm(enumerate(data_loader), desc='epoch: %d' % (epoch), total=len(data_loader), bar_format='{l_bar}{r_bar}')

        total_loss = 0.0
        FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        for i, _data in data_iter:
            data = {}
            for key, value in _data.items():
                if not 'name' in key:
                    data[key] = value.to(self.device) # Load data on GPU
            cloth = data['cloth']
            cloth_mask = data['cloth_mask']
            person = data['person']
            batch_size = person.shape[0]

            outputs = self.gen(torch.cat([data['feature'], cloth],1)) # (batch, channel, height, width)
            rendered_person, composition_mask = torch.split(outputs, 3,1)
            rendered_person = torch.tanh(rendered_person)
            composition_mask = torch.sigmoid(composition_mask)
            tryon_person = cloth*composition_mask + rendered_person*(1-composition_mask)
            visuals = [[data['head'], data['shape'], data['pose']], 
                    [cloth, cloth_mask*2-1, composition_mask*2-1], 
                    [rendered_person, tryon_person, person]]         

            # Adversarial ground truths
            real = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False) # Batch size
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
            # Loss measures generator's ability to fool the discriminator
            l_l1 = self.criterionL1(tryon_person, person)
            l_mask = self.criterionL1(composition_mask, cloth_mask)
            l_vgg = self.criterionVGG(tryon_person, person)
            dis_fake = self.dis(torch.cat([data['feature'], cloth, tryon_person],1)) # Dis forward
            l_adv = self.criterionAdv(dis_fake, real)
            loss_g = l_l1 + l_vgg + l_mask + l_adv/batch_size
            # Loss for discriminator
            loss_d = ( self.criterionAdv(self.dis(torch.cat([data['feature'], cloth, person],1)), real) +\
                        self.criterionAdv(self.dis(torch.cat([data['feature'], cloth, tryon_person],1).detach()), fake) )\
                        / 2 

            if train:
                self.optim_g.zero_grad()
                loss_g.backward()
                self.optim_g.step()
                self.optim_d.zero_grad()
                loss_d.backward()
                self.optim_d.step()
                self.step += 1

            total_loss = total_loss + loss_g.item() + loss_d.item()
            post_fix = {
                'epoch': epoch,
                'iter': i,
                'avg_loss': total_loss/(i+1),
                'loss_recon': l_l1.item() + l_vgg.item() + l_mask.item(),
                'loss_g': l_adv.item(),
                'loss_d': loss_d.item()
            }
            if train and i%self.log_freq==0:
                data_iter.write(str(post_fix))
                board_add_images(visuals, epoch, i, os.path.join(self.save_dir,'train'))
        
        return total_loss/len(data_iter)
        

def get_opt():
    parser = argparse.ArgumentParser(description='Train TOM model')
    parser.add_argument('--n_epoch', '-e', type=int, default=100, help='number of epochs')
    parser.add_argument('--data_root', '-d', type=str, default='data', help='path to data root directory')
    parser.add_argument('--out_dir', '-o', type=str, default='../result', help='path to result directory')
    parser.add_argument('--name', '-n', type=str, default='TOM', help='model name')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--n_worker', '-w', type=int, default=16, help='number of workers')
    parser.add_argument('--gpu_id', '-g', type=str, default='0', help='GPU ID')
    parser.add_argument('--log_freq', type=int, default=100, help='log frequency')
    parser.add_argument('--radius', type=int, default=5)
    # Not used
    parser.add_argument('--fine_width', type=int, default=192)
    parser.add_argument('--fine_height', type=int, default=256)
    parser.add_argument('--grid_size', type=int, default=5, help='hyperparameter for the network')
    opt = parser.parse_args()
    return opt

def main():
    opt = get_opt()
    print(opt)

    print('Loading dataset')
    dataset_train = TOMDataset(opt, mode='train', data_list='train_pairs.txt')
    dataloader_train = DataLoader(dataset_train, batch_size=opt.batch_size, num_workers=opt.n_worker, shuffle=True)
    dataset_val = TOMDataset(opt, mode='val', data_list='val_pairs.txt', train=False)
    dataloader_val = DataLoader(dataset_val, batch_size=opt.batch_size, num_workers=opt.n_worker, shuffle=True)

    save_dir = os.path.join(opt.out_dir, opt.name)
    log_dir = os.path.join(opt.out_dir, 'log')
    dirs = [opt.out_dir, save_dir, os.path.join(save_dir,'train'), log_dir]
    for d in dirs:
        mkdir(d)
    log_name = os.path.join(log_dir, opt.name+'.csv')
    with open(log_name, 'w') as f:
        f.write('epoch,train_loss,val_loss\n')

    print('Building TOM model')
    gen = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
    dis = NLayerDiscriminator(28, ndf=64, n_layers=6, norm_layer=nn.InstanceNorm2d, use_sigmoid=True)
    gen.cuda()
    dis.cuda()
    n_step = int(opt.n_epoch*len(dataset_train) / opt.batch_size)
    trainer = TOMTrainer(gen, dis, dataloader_train, dataloader_val, opt.gpu_id, opt.log_freq, save_dir, n_step)

    print('Start training TOM')
    for epoch in tqdm(range(opt.n_epoch)):
        print('Epoch: {}'.format(epoch))
        loss = trainer.train(epoch)
        print('Train loss: {:.3f}'.format(loss))
        with open(log_name, 'a') as f:
            f.write('{},{:.3f},'.format(epoch, loss))
        save_checkpoint(gen, os.path.join(save_dir, 'gen_epoch_{:02}.pth'.format(epoch)))
        save_checkpoint(dis, os.path.join(save_dir, 'dis_epoch_{:02}.pth'.format(epoch)))
        
        loss = trainer.val(epoch)
        print('Validation loss: {:.3f}'.format(loss))
        with open(log_name, 'a') as f:
            f.write('{:.3f}\n'.format(loss))
    print('Finish training TOM')

if __name__=='__main__':
    main()
