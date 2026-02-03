import numpy as np
import glob
import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset
from gan_utils.io import CondStereoVeristasDataGenNormUniformSampling
import matplotlib.pyplot as plt
from torchinfo import summary
from gan.trainer_Validator import ValidatorTrainer
import tqdm
from gan.model import ConditionalGenerator, ConditionalDiscriminator, Validator


device = 'cuda' if torch.cuda.is_available() else 'cpu'


gen_model = ConditionalGenerator(n_z=256, input_filt=256, final_size=96, out_channels=4, n_embeddings=24, embedding_features=64).to(device)
disc_model = ConditionalDiscriminator(in_channels=4, n_embeddings=24, embedding_features=64, norm=False, pool=False).to(device)
validator_model = Validator(in_channels=4, n_parameters=24, n_z=256).to(device)


summary(validator_model, input_size=(1,4,96,96))


gen_model.load_state_dict(torch.load('./checkpoints_CondwGAN_Stereo_v5/generator_ep_500.pth'))
disc_model.load_state_dict(torch.load('./checkpoints_CondwGAN_Stereo_v5/discriminator_ep_500.pth'))


veritas_data_1 = CondStereoVeristasDataGenNormUniformSampling(input_file = "/home/fortson/shared/veritas/datafiles/71802_dl1_full.h5", 
                                         size_threshold=100, mode='intersect')

veritas_data_2 = CondStereoVeristasDataGenNormUniformSampling(input_file = "/home/fortson/shared/veritas/datafiles/70441_dl1_full.h5", 
                                         size_threshold=100, mode='intersect')

veritas_data_3 = CondStereoVeristasDataGenNormUniformSampling(input_file = "/home/fortson/shared/veritas/datafiles/70488_dl1_full.h5", 
                                         size_threshold=100, mode='intersect')

veritas_data_4 = CondStereoVeristasDataGenNormUniformSampling(input_file = "/home/fortson/shared/veritas/datafiles/70533_dl1_full.h5", 
                                         size_threshold=100, mode='intersect')

veritas_data_5 = CondStereoVeristasDataGenNormUniformSampling(input_file = "/home/fortson/shared/veritas/datafiles/74916_dl1_full.h5", 
                                         size_threshold=100, mode='intersect')

veritas_data_6 = CondStereoVeristasDataGenNormUniformSampling(input_file = "/home/fortson/shared/veritas/datafiles/75752_dl1_full.h5", 
                                         size_threshold=100, mode='intersect')

train_val_split = 0.9
batch_size = 32
nworkers = 18

# dataloader = DataLoader(veritas_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nworkers, persistent_workers=True)
concatenated_dataset = ConcatDataset([veritas_data_1, veritas_data_2, veritas_data_3, veritas_data_4, veritas_data_5, veritas_data_6])

torch_rand_gen = torch.Generator().manual_seed(9999)
train_datagen, val_datagen = random_split(concatenated_dataset, [train_val_split, 1 - train_val_split], generator=torch_rand_gen)


train_data = DataLoader(train_datagen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nworkers, persistent_workers=True)
val_data = DataLoader(val_datagen, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=nworkers, persistent_workers=True)


valid_trainer = ValidatorTrainer(validator_model, gen_model, disc_model, savefolder='./checkpoints_CondwGAN_Validator_v7', device=device)

valid_trainer.train(train_data, val_data, epochs=500, save_freq = 10, lr=1e-4)