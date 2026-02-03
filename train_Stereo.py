import numpy as np
import glob
import torch
from torch.utils.data import DataLoader, random_split
from gan_utils.io import VeritasDataGen, VeritasDataGenNorm, StereoVeristasDataGenNorm
import matplotlib.pyplot as plt
from torchinfo import summary
from gan.trainer import Trainer
import tqdm
from gan.model import Generator, Discriminator


device = 'cuda' if torch.cuda.is_available() else 'cpu'

veritas_data = StereoVeristasDataGenNorm(input_file = "/home/fortson/manth145/data/Veritas/71802_dl1_full.h5", size_threshold=1000, mode='union')

train_val_split = 0.9
batch_size = 32
nworkers = 18

dataloader = DataLoader(veritas_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nworkers, persistent_workers=True)

torch_rand_gen = torch.Generator().manual_seed(9999)
train_datagen, val_datagen = random_split(veritas_data, [train_val_split, 1 - train_val_split], generator=torch_rand_gen)


train_data = DataLoader(train_datagen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nworkers, persistent_workers=True)
val_data = DataLoader(val_datagen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nworkers, persistent_workers=True)


gen_model = Generator(n_z=256, input_filt=256, final_size=96, out_channels=4, pool=False, norm=False).to(device)
disc_model = Discriminator(in_channels=4, input_size=96, n_layers=4, norm=False, pool=False).to(device)

summary(gen_model, input_size=(1,256))

summary(disc_model, input_size=(1,4,96,96))

gan_trainer = Trainer(generator=gen_model, discriminator=disc_model, savefolder='./checkpoints_norm_stereo_broad_NAdam_1e4', device=device)

gan_trainer.train(train_data, val_data, epochs=500, save_freq = 10, dsc_learning_rate=1.e-4, gen_learning_rate=1.e-4, lr_decay=0.95, decay_freq=10)

# lr_decay=0.95, decay_freq=10
# , decay_freq = 5