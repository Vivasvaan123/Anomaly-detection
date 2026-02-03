import numpy as np
import glob
import torch
from torch.utils.data import DataLoader, random_split
from gan_utils.io import VeritasDataGenNorm, StereoVeristasDataGenNorm
import matplotlib.pyplot as plt
from torchinfo import summary
from gan.trainer_encoder import EncoderTrainer
import tqdm
from gan.model import Generator, Discriminator, Encoder, DiscriminatorFeatures
# from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from copy import deepcopy
from gan_utils.functions import transform_state_dict


device = 'cuda' if torch.cuda.is_available() else 'cpu'

veritas_data = StereoVeristasDataGenNorm(input_file = "/home/fortson/manth145/data/Veritas/71802_dl1_full.h5", size_threshold=1000)

train_val_split = 0.9
batch_size = 32
nworkers = 18

dataloader = DataLoader(veritas_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nworkers, persistent_workers=True)

torch_rand_gen = torch.Generator().manual_seed(9999)
train_datagen, val_datagen = random_split(veritas_data, [train_val_split, 1 - train_val_split], generator=torch_rand_gen)


train_data = DataLoader(train_datagen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nworkers, persistent_workers=True)
val_data = DataLoader(val_datagen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nworkers, persistent_workers=True)



gen_model = Generator(n_z=256, input_filt=256, final_size=96, out_channels=4, pool=False, norm=False).to(device)
discfeat_model = DiscriminatorFeatures(in_channels=4, input_size=96, n_layers=4, pool=False, norm=False).to(device)
enc_model = Encoder(in_channels=4, n_z=256, n_layers=5, input_size=96, norm=False, pool=False).to(device)

gen_model.load_state_dict(torch.load('./checkpoints_norm_stereo_NAdam_1e4/generator_ep_500.pth'))

new_keys = ['Down_0.0.weight', 'Down_0.0.bias', 'Down_1.0.weight', 
            'Down_1.0.bias', 'Down_2.0.weight', 'Down_2.0.bias', 
            'Down_3.0.weight', 'Down_3.0.bias', 'act_final.weight', 'act_final.bias']

discfeat_model.load_state_dict(transform_state_dict(torch.load('./checkpoints_norm_stereo_NAdam_1e4/discriminator_ep_500.pth'), new_keys))

enc_trainer = EncoderTrainer(enc_model, gen_model, discfeat_model, savefolder='./checkpoints_encoder_norm_NAdam_1e4', device=device)

enc_trainer.train(train_data, val_data, epochs=500, save_freq = 10, lr=1e-4)