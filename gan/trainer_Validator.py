import torch
import os
import tqdm
import numpy as np
import glob
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from .losses import wasserstein_distance, gradient_penalty
from einops import rearrange
from collections import defaultdict

class ValidatorTrainer:
    
    def __init__(self, encoder, gen, disc, savefolder, device='cuda'):
        
        encoder.apply(weights_init)
        
        self.generator = gen
        self.discriminator = disc
        self.encoder = encoder
        self.device = device
        
        
        
        if savefolder[-1] != '/':
            savefolder += '/'

        self.savefolder = savefolder
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)
        
        self.start = 1
        
        
        self.loss_fn = torch.nn.MSELoss()
        self.alpha = 0.8
        self.param_alpha = 10
        
        
    def batch(self, x, y, train=False):
        
        if not isinstance(x, torch.Tensor):
            input_tensor = torch.as_tensor(x, dtype=torch.float).to(self.device, non_blocking=True)
            input_condition = torch.as_tensor(y, dtype=torch.float).to(self.device)
        else:
            input_tensor = x.to(self.device, non_blocking=True)
            input_condition = y.to(self.device, non_blocking=True)
        
        latent_embedding_encoder, cond_encoder_parameters = self.encoder(input_tensor)
        batch_size = latent_embedding_encoder.shape[0]
        
        
        gen_img = self.generator(latent_embedding_encoder, cond_encoder_parameters)
        disc_out_real, disc_features_real = self.discriminator(input_tensor, input_condition)
        
        disc_out_gen, disc_features_gen = self.discriminator(gen_img, cond_encoder_parameters)

        img_loss = torch.sum(torch.mean(torch.pow(input_tensor - gen_img, 2), dim=0))
        feat_loss = torch.sum(torch.mean(torch.pow(disc_features_real - disc_features_gen, 2), dim=0))
        
        
        input_cond_reshaped = input_condition.view(batch_size,4,6).to(self.device)
        encoder_cond_reshaped = cond_encoder_parameters.view(batch_size,4,6).to(self.device)
        
        size_loss = torch.sum(torch.mean(torch.pow(input_cond_reshaped - encoder_cond_reshaped, 2), dim=0))
        other_param_loss = torch.sum(torch.mean(torch.pow(input_cond_reshaped[:,:,1:] - encoder_cond_reshaped[:,:,1:], 2), dim=0))
        parameter_pred_loss = 1 * size_loss + other_param_loss
        
        total_loss = (1 - self.alpha) * img_loss + (self.alpha) * feat_loss + self.param_alpha * parameter_pred_loss
        
        
        if train:
            self.encoder.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
        keys = ['img_loss', 'feat_loss', 'param_loss', 'total_loss']
        mean_losses = [img_loss.item(), feat_loss.item(), parameter_pred_loss.item(), total_loss.item()]
        loss_dict = {key: val for key, val in zip(keys, mean_losses)}
        
        return loss_dict
    
    def train(self, train_data, val_data, epochs, lr=1e-4, save_freq=10):
        self.optimizer = optim.NAdam(self.encoder.parameters(), lr=lr, betas=(0.5,0.9))
        
        losses_ep = []
        for epoch in range(self.start, epochs + 1):
            
            print(f"Epoch {epoch}")
            print("-------------------------------------------------------")
            
            pbar = tqdm.tqdm(train_data, desc='Training', dynamic_ncols=True)

            if hasattr(train_data, 'shuffle'):
                train_data.shuffle()


            self.generator.eval()
            self.discriminator.eval()
            
            self.encoder.train()

            losses = defaultdict(list)
            for i, (input_img, input_cond) in enumerate(pbar):
                batch_loss = self.batch(input_img, input_cond, train=True)

                loss_mean = {}
                for key, value in batch_loss.items():
                    losses[key].append(value)
                    loss_mean[key] = np.mean(losses[key], axis=0)

                loss_str = " ".join([f"{key}: {value:.2e}" for key, value in loss_mean.items()])
                pbar.set_postfix_str(loss_str)
            
            losses_ep.append([loss_mean['img_loss'], loss_mean['feat_loss'], loss_mean['param_loss'], loss_mean['total_loss']])
            
            
            self.encoder.eval()
            if hasattr(val_data, 'shuffle'):
                val_data.shuffle()

            pbar = tqdm.tqdm(val_data, desc='Validation: ')
            losses = defaultdict(list)
            for i, (input_img, input_cond) in enumerate(pbar):
                batch_loss = self.batch(input_img, input_cond, train=False)
                loss_mean = {}
                for key, value in batch_loss.items():
                    losses[key].append(value)
                    loss_mean[key] = np.mean(losses[key], axis=0)

                loss_str = " ".join([f"{key}: {value:.2e}" for key, value in loss_mean.items()])

                pbar.set_postfix_str(loss_str)

            if epoch % save_freq == 0:
                self.save(epoch)
                self.save_metrics(losses_ep)
        
        return
    
    def get_activation(self, layer_name):
        def hook(model, input, output):
            self.feature_vector_storage[layer_name] = output.detach()
        return hook
    
    def save(self, epoch):
        save_filename = f'{self.savefolder}/validator_ep_{epoch:03d}.pth'
        torch.save(self.encoder.state_dict(), save_filename)
        print(f'Saving to {save_filename}')
    def save_metrics(self, loss_values):
        np.save(f'{self.savefolder}/loss_values.npy', np.array(loss_values))
        
    

        
        
def weights_init(net):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.xavier_uniform_(m.weight.data)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('InstanceNorm') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data, 1.0)
            torch.nn.init.constant_(m.bias.data, 0.0)