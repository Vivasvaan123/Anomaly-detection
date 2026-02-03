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

import os
import numpy as np
import torch
import tqdm
from collections import defaultdict
from torch import optim
from torch.cuda.amp import autocast, GradScaler


class EncoderTrainer:

    def __init__(self, encoder, gen, disc, savefolder, device='cuda'):
        encoder.apply(weights_init)

        self.encoder = encoder
        self.generator = gen
        self.discriminator = disc
        self.device = device

        if savefolder[-1] != '/':
            savefolder += '/'
        self.savefolder = savefolder
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)

        self.start = 1
        self.alpha = 0.8

        # AMP scaler
        self.scaler = GradScaler()

    # ------------------------------------------------------------
    # AMP‑ENABLED BATCH FUNCTION
    # ------------------------------------------------------------
#     def batch(self, x, train=False):

#         # Move input to device
#         if not isinstance(x, torch.Tensor):
#             input_tensor = torch.as_tensor(x, dtype=torch.float).to(self.device, non_blocking=True)
#         else:
#             input_tensor = x.to(self.device, non_blocking=True)

#         # -------------------------
#         # Forward pass under autocast
#         # -------------------------
#         with autocast():
#             z = self.encoder(input_tensor)
#             gen_img = self.generator(z)

#             disc_out_real, disc_features_real = self.discriminator(input_tensor)
#             disc_out_gen, disc_features_gen = self.discriminator(gen_img)

#             img_loss = torch.sum(torch.mean((input_tensor - gen_img) ** 2, dim=0))
#             feat_loss = torch.sum(torch.mean((disc_features_real - disc_features_gen) ** 2, dim=0))

#             total_loss = (1 - self.alpha) * img_loss + self.alpha * feat_loss

#         # -------------------------
#         # Backward pass with AMP
#         # -------------------------
#         if train:
#             self.encoder.zero_grad(set_to_none=True)
#             self.scaler.scale(total_loss).backward()
#             self.scaler.step(self.optimizer)
#             self.scaler.update()

#         # -------------------------
#         # Return loss dict
#         # -------------------------
#         keys = ['img_loss', 'feat_loss', 'total_loss']
#         vals = [img_loss.item(), feat_loss.item(), total_loss.item()]
#         return {k: v for k, v in zip(keys, vals)}

    def batch(self, x, train=False):
        if not isinstance(x, torch.Tensor):
            input_tensor = torch.as_tensor(x, dtype=torch.float).to(self.device, non_blocking=True)
        else:
            input_tensor = x.to(self.device, non_blocking=True)

        batch_size = input_tensor.size(0)
        n_z = self.encoder.n_z

        # -------------------------
        # 1. Encode real images
        # -------------------------
        z_enc = self.encoder(input_tensor)          # gradients ON (only here)
        gen_img = self.generator(z_enc)             # gradients OFF (we will freeze G)

        # -------------------------
        # 2. Discriminator features (NO GRAD)
        # -------------------------
        with torch.no_grad():
            _, disc_features_real = self.discriminator(input_tensor)
            _, disc_features_gen  = self.discriminator(gen_img)

        img_loss  = torch.mean((input_tensor - gen_img)**2)
        feat_loss = torch.mean((disc_features_real - disc_features_gen)**2)

        # -------------------------
        # 3. Latent regression (NO GRAD through G)
        # -------------------------
        with torch.no_grad():
            z_rand = torch.randn(batch_size, n_z, device=self.device)
            x_fake = self.generator(z_rand)

        z_hat = self.encoder(x_fake)                # gradients ON
        latent_loss = torch.mean((z_hat - z_rand)**2)

        # -------------------------
        # 4. Total loss
        # -------------------------
        beta = 1.0
        total_loss = (1 - self.alpha) * img_loss + self.alpha * feat_loss + beta * latent_loss

        if train:
            self.encoder.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        keys = ['img_loss', 'feat_loss', 'latent_loss', 'total_loss']
        mean_losses = [img_loss.item(), feat_loss.item(), latent_loss.item(), total_loss.item()]
        return {k: v for k, v in zip(keys, mean_losses)}


    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------
    def train(self, train_data, val_data, epochs, lr=1e-4, save_freq=10):

        self.optimizer = optim.NAdam(self.encoder.parameters(), lr=lr, betas=(0.5, 0.9))

        losses_ep = []

        for epoch in range(self.start, epochs + 1):
            print(f"Epoch {epoch}")
            print("-------------------------------------------------------")

            # Set modes
            self.generator.eval()
            self.discriminator.eval()
            self.encoder.train()

            # -------------------------
            # TRAINING
            # -------------------------
            pbar = tqdm.tqdm(train_data, desc='Training', dynamic_ncols=True)
            losses = defaultdict(list)

            for i, input_img in enumerate(pbar):
                batch_loss = self.batch(input_img, train=True)

                loss_mean = {}
                for key, value in batch_loss.items():
                    losses[key].append(value)
                    loss_mean[key] = np.mean(losses[key])

                pbar.set_postfix_str(
                    " ".join([f"{k}: {v:.2e}" for k, v in loss_mean.items()])
                )

            losses_ep.append([
                loss_mean['img_loss'],
                loss_mean['feat_loss'],
                loss_mean['total_loss']
            ])

            # -------------------------
            # VALIDATION
            # -------------------------
            self.encoder.eval()
            pbar = tqdm.tqdm(val_data, desc='Validation')
            losses = defaultdict(list)

            for i, input_img in enumerate(pbar):
                batch_loss = self.batch(input_img, train=False)

                loss_mean = {}
                for key, value in batch_loss.items():
                    losses[key].append(value)
                    loss_mean[key] = np.mean(losses[key])

                pbar.set_postfix_str(
                    " ".join([f"{k}: {v:.2e}" for k, v in loss_mean.items()])
                )

            # -------------------------
            # SAVE CHECKPOINTS
            # -------------------------
            if epoch % save_freq == 0:
                self.save(epoch)
                self.save_metrics(losses_ep)

        return

    # ------------------------------------------------------------
    # CHECKPOINTING
    # ------------------------------------------------------------
    def save(self, epoch):
        save_filename = f'{self.savefolder}/encoder_ep_{epoch:03d}.pth'
        torch.save(self.encoder.state_dict(), save_filename)
        print(f'Saving to {save_filename}')

    def save_metrics(self, loss_values):
        np.save(f'{self.savefolder}/loss_values.npy', np.array(loss_values))


# class EncoderTrainer:
    
#     def __init__(self, encoder, gen, disc, savefolder, device='cuda'):
        
#         encoder.apply(weights_init)
        
#         self.generator = gen
#         self.discriminator = disc
#         self.encoder = encoder
#         self.device = device
        
        
        
#         if savefolder[-1] != '/':
#             savefolder += '/'

#         self.savefolder = savefolder
#         if not os.path.exists(savefolder):
#             os.mkdir(savefolder)
        
#         self.start = 1
        
        
#         self.loss_fn = torch.nn.MSELoss()
#         self.alpha = 0.8
        
        
#     def batch(self, x, train=False):
        
#         if not isinstance(x, torch.Tensor):
#             input_tensor = torch.as_tensor(x, dtype=torch.float).to(self.device, non_blocking=True)
#         else:
#             input_tensor = x.to(self.device, non_blocking=True)
        
#         latent_embedding_encoder = self.encoder(input_tensor)
#         self.feature_vector_storage = {}
        
        
#         gen_img = self.generator(latent_embedding_encoder)
#         disc_out_real, disc_features_real = self.discriminator(input_tensor)
#         disc_out_gen, disc_features_gen = self.discriminator(gen_img)
# #         hook1 = self.discriminator.rearr_final.register_forward_hook(self.get_activation('rearr_final'))
# #         disc_real_run = self.discriminator(input_tensor)
# #         disc_features_real = self.feature_vector_storage['rearr_final']

# #         disc_gen_run = self.discriminator(gen_img)
# #         disc_features_gen = self.feature_vector_storage['rearr_final']
# #         hook1.remove()
        
# #         img_loss = torch.sum(torch.mean(self.loss_fn(input_tensor, gen_img), dim=0))
# #         feat_loss = torch.sum(torch.mean(self.loss_fn(disc_features_real, disc_features_gen), dim=0))
#         img_loss = torch.sum(torch.mean(torch.pow(input_tensor - gen_img, 2), dim=0))
#         feat_loss = torch.sum(torch.mean(torch.pow(disc_features_real - disc_features_gen, 2), dim=0))
        
#         total_loss = (1 - self.alpha) * img_loss + (self.alpha) * feat_loss
        
# #         total_loss = torch.autograd.Variable(self.alpha * feat_loss + (1-self.alpha) * img_loss, requires_grad=True)
        
#         if train:
#             self.encoder.zero_grad()
#             total_loss.backward()
#             self.optimizer.step()
            
#         keys = ['img_loss', 'feat_loss', 'total_loss']
#         mean_losses = [img_loss.item(), feat_loss.item(), total_loss.item()]
#         loss_dict = {key: val for key, val in zip(keys, mean_losses)}
        
#         return loss_dict
    
#     def train(self, train_data, val_data, epochs, lr=1e-4, save_freq=10):
#         self.optimizer = optim.NAdam(self.encoder.parameters(), lr=lr, betas=(0.5,0.9))
        
#         losses_ep = []
#         for epoch in range(self.start, epochs + 1):
            
#             print(f"Epoch {epoch}")
#             print("-------------------------------------------------------")
            
#             pbar = tqdm.tqdm(train_data, desc='Training', dynamic_ncols=True)

#             if hasattr(train_data, 'shuffle'):
#                 train_data.shuffle()


#             self.generator.eval()
#             self.discriminator.eval()
            
#             self.encoder.train()

#             losses = defaultdict(list)
#             for i, input_img in enumerate(pbar):
#                 batch_loss = self.batch(input_img, train=True)

#                 loss_mean = {}
#                 for key, value in batch_loss.items():
#                     losses[key].append(value)
#                     loss_mean[key] = np.mean(losses[key], axis=0)

#                 loss_str = " ".join([f"{key}: {value:.2e}" for key, value in loss_mean.items()])
#                 pbar.set_postfix_str(loss_str)
            
#             losses_ep.append([loss_mean['img_loss'], loss_mean['feat_loss'], loss_mean['total_loss']])
            
            
#             self.encoder.eval()
#             if hasattr(val_data, 'shuffle'):
#                 val_data.shuffle()

#             pbar = tqdm.tqdm(val_data, desc='Validation: ')
#             losses = defaultdict(list)
#             for i, input_img in enumerate(pbar):
#                 batch_loss = self.batch(input_img, train=False)
#                 loss_mean = {}
#                 for key, value in batch_loss.items():
#                     losses[key].append(value)
#                     loss_mean[key] = np.mean(losses[key], axis=0)

#                 loss_str = " ".join([f"{key}: {value:.2e}" for key, value in loss_mean.items()])

#                 pbar.set_postfix_str(loss_str)

#             if epoch % save_freq == 0:
#                 self.save(epoch)
#                 self.save_metrics(losses_ep)
        
#         return
    
#     def get_activation(self, layer_name):
#         def hook(model, input, output):
#             self.feature_vector_storage[layer_name] = output.detach()
#         return hook
    
#     def save(self, epoch):
#         save_filename = f'{self.savefolder}/encoder_ep_{epoch:03d}.pth'
#         torch.save(self.encoder.state_dict(), save_filename)
#         print(f'Saving to {save_filename}')
#     def save_metrics(self, loss_values):
#         np.save(f'{self.savefolder}/loss_values.npy', np.array(loss_values))
        
    

        
        
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