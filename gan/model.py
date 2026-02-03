from torch import nn
from einops.layers.torch import Rearrange
from collections import OrderedDict
import torch

class UpConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation, norm=True, pool=True, **kwargs):
        if activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            activation = nn.LeakyReLU(0.2, True)

        layers = [nn.ConvTranspose2d(in_channels, out_channels, **kwargs)]
        if norm:
            layers += [nn.InstanceNorm2d(out_channels)]
        if pool:
            layers += [nn.MaxPool2d((5, 5), stride=(1, 1), padding=2)]
        layers += [activation]

        super().__init__(*layers)


class DownConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation, iteration, norm=False, **kwargs):
        if activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            activation = nn.LeakyReLU(0.2)

        layers = {f'conv2d_{iteration}': nn.Conv2d(in_channels, out_channels, **kwargs)}
        if norm:
            layers[f'norm_{iteration}'] = nn.InstanceNorm2d(out_channels)

        layers[f'act_{iteration}'] = activation

        super().__init__(OrderedDict(nn.ModuleDict(layers)))


class Generator(nn.Sequential):
    def __init__(self, n_z, input_filt=512, norm=False, pool=False, n_layers=5, out_channels=3, final_size=512):
        # final_size is now 512 to match padded images
        self.n_z = n_z

        layers = []
        prev_filt = input_filt
        for _ in range(n_layers):
            layers.append(
                UpConvLayer(
                    prev_filt,
                    int(prev_filt / 2),
                    activation='leakyrelu',
                    norm=norm,
                    pool=pool,
                    kernel_size=(6, 6),
                    stride=(2, 2),
                    padding=2
                )
            )
            prev_filt = int(prev_filt / 2)

        # initial_size = final_size / 2**n_layers
        initial_size = final_size / 2 ** n_layers
        if initial_size % 1 != 0:
            raise ValueError(
                f"Cannot create a model to produce a {final_size} x {final_size} image with {n_layers} layers"
            )

        initial_size = int(initial_size)

        super().__init__(
            nn.Linear(n_z, initial_size * initial_size * input_filt),
            nn.LeakyReLU(0.2, True),
            Rearrange('b (h w z) -> b z h w', h=initial_size, w=initial_size, z=input_filt),
            *layers,
            nn.Conv2d(prev_filt, out_channels, (5, 5), stride=(1, 1), padding=2),
            nn.Sigmoid()
        )


class Discriminator(nn.Sequential):
    def __init__(self, in_channels, n_layers=5, input_size=512, norm=False, pool=False):
        # input_size is now 512 (padded from 500)
        prev_filt = 8
        layers = []
        for i in range(n_layers):
            layers.append(
                DownConvLayer(
                    prev_filt if i > 0 else in_channels,
                    prev_filt * 2,
                    activation='leakyrelu',
                    kernel_size=(6, 6),
                    stride=(2, 2),
                    padding=2,
                    norm=norm,
                    iteration=i
                )
            )
            prev_filt = prev_filt * 2
            input_size = input_size / 2  # 512 -> 256 -> 128 -> 64 -> 32 -> 16

        # final spatial size = 16 x 16
        super().__init__(
            *layers,
            Rearrange('b z h w -> b (z h w)'),
            nn.Linear(int(input_size) * int(input_size) * prev_filt, 1)
#             nn.Linear(25088, 1)

        )
        


class DiscriminatorFeatures(nn.Module):
    def __init__(self, in_channels, n_layers=4, input_size=512, norm=False, pool=False):
        super().__init__()

        # Here we only use 4 downsamples, so final size = 512 / 2^4 = 32
        # But your original code assumed 6x6; we’ll adapt it to 16x16 or 32x32.
        # To keep it consistent with 5-layer discriminator, let's also go to 16x16.

        self.Down_0 = DownConvLayer(
            in_channels, 16, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=0
        )
        self.Down_1 = DownConvLayer(
            16, 32, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=1
        )
        self.Down_2 = DownConvLayer(
            32, 64, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=2
        )
        self.Down_3 = DownConvLayer(
            64, 128, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=3
        )
        # After 4 downsamples: 512 -> 256 -> 128 -> 64 -> 32
        # If you want 16x16, add one more Down layer:
        self.Down_4 = DownConvLayer(
            128, 256, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=4
        )
        # Now: 32 -> 16

        self.rearr_final = Rearrange('b z h w -> b (z h w)')
        # final spatial size = 16 x 16, channels = 256
        self.act_final = nn.Linear(16 * 16 * 256, 1)

    def forward(self, x):
        x = self.Down_0(x)
        x = self.Down_1(x)
        x = self.Down_2(x)
        x = self.Down_3(x)
        x = self.Down_4(x)
        feats = self.rearr_final(x)
        output = self.act_final(feats)
        return output, feats
    
    
class Encoder(nn.Sequential):
    def __init__(self, in_channels, n_z=128, n_layers=5, input_size=512, norm=False, pool=False):
        super().__init__()

        self.n_z = n_z   # <-- ADD THIS LINE

        prev_filt = 8
        modules = {}
        for i in range(n_layers):
            modules[f'Down_{i}'] = DownConvLayer(
                prev_filt if i > 0 else in_channels,
                prev_filt * 2,
                activation='leakyrelu',
                kernel_size=(6, 6),
                stride=(2, 2),
                padding=2,
                norm=False,
                iteration=i
            )
            prev_filt = prev_filt * 2
            input_size = input_size / 2

        modules['rearr_final'] = Rearrange('b z h w -> b (z h w)')
        modules['act_final'] = nn.Linear(int(input_size) * int(input_size) * prev_filt, n_z)

        module_dict = nn.ModuleDict(modules)
        super().__init__(OrderedDict(module_dict))



# class Encoder(nn.Sequential):
#     def __init__(self, in_channels, n_z=128, n_layers=5, input_size=512, norm=False, pool=False):
#         prev_filt = 8
#         modules = {}
#         for i in range(n_layers):
#             modules[f'Down_{i}'] = DownConvLayer(
#                 prev_filt if i > 0 else in_channels,
#                 prev_filt * 2,
#                 activation='leakyrelu',
#                 kernel_size=(6, 6),
#                 stride=(2, 2),
#                 padding=2,
#                 norm=False,
#                 iteration=i
#             )
#             prev_filt = prev_filt * 2
#             input_size = input_size / 2  # 512 -> 256 -> 128 -> 64 -> 32 -> 16

#         modules['rearr_final'] = Rearrange('b z h w -> b (z h w)')
#         modules['act_final'] = nn.Linear(int(input_size) * int(input_size) * prev_filt, n_z)
#         module_dict = nn.ModuleDict(modules)

#         super().__init__(OrderedDict(module_dict))


class ConcatLayer(nn.Module):
    def __init__(self, n_e, n_e_feats, n_z, n_z_features):
        super().__init__()
        self.embed_layer = nn.Embedding(n_e, n_e_feats)
        self.latent_layer = nn.Linear(n_z, n_z_features)

    def forward(self, z_vec, cond):
        X = self.latent_layer(z_vec)
        Y = self.embed_layer(cond)
        return torch.cat([X, Y], dim=1)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, in_channels=1, n_embeddings=3, embedding_features=32, norm=False, pool=False):
        super().__init__()
        self.embedding_layer = nn.Linear(n_embeddings, embedding_features)

        self.Down_0 = DownConvLayer(
            in_channels, 16, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=0
        )
        self.Down_1 = DownConvLayer(
            16, 32, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=1
        )
        self.Down_2 = DownConvLayer(
            32, 64, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=2
        )
        self.Down_3 = DownConvLayer(
            64, 128, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=3
        )
        self.Down_4 = DownConvLayer(
            128, 256, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=4
        )
        # 512 -> 256 -> 128 -> 64 -> 32 -> 16

        self.rearr_final = Rearrange('b z h w -> b (z h w)')
        # final features: 16 x 16 x 256
        self.act_final = nn.Linear((16 * 16 * 256) + embedding_features, 1)

    def forward(self, x, label):
        x = self.Down_0(x)
        x = self.Down_1(x)
        x = self.Down_2(x)
        x = self.Down_3(x)
        x = self.Down_4(x)
        feats = self.rearr_final(x)
        label_em = self.embedding_layer(label)
        merged_feats = torch.cat([feats, label_em], 1)
        output = self.act_final(merged_feats)
        return output, feats


class ConditionalGenerator(nn.Module):
    def __init__(self, n_z, n_embeddings=3, input_filt=512, final_size=512,
                 embedding_features=32, out_channels=1, norm=False, pool=False):
        super().__init__()
        self.n_z = n_z

        # We want to start from 16x16 with 256 channels:
        # 16 * 16 * 256 = 65536
        # We'll reserve embedding_features for the condition.
        self.zlayer = nn.Linear(n_z, (16 * 16 * 256) - embedding_features)
        self.embedding_layer = nn.Linear(n_embeddings, embedding_features)
        self.act_z = nn.LeakyReLU(0.2, True)
        self.rearrange = Rearrange('b (h w z) -> b z h w', h=16, w=16, z=256)

        self.Up_1 = UpConvLayer(
            256, 128, activation='leakyrelu', norm=norm, pool=pool,
            kernel_size=(6, 6), stride=(2, 2), padding=2
        )
        self.Up_2 = UpConvLayer(
            128, 64, activation='leakyrelu', norm=norm, pool=pool,
            kernel_size=(6, 6), stride=(2, 2), padding=2
        )
        self.Up_3 = UpConvLayer(
            64, 32, activation='leakyrelu', norm=norm, pool=pool,
            kernel_size=(6, 6), stride=(2, 2), padding=2
        )
        self.Up_4 = UpConvLayer(
            32, 16, activation='leakyrelu', norm=norm, pool=pool,
            kernel_size=(6, 6), stride=(2, 2), padding=2
        )
        self.Up_5 = UpConvLayer(
            16, 8, activation='leakyrelu', norm=norm, pool=pool,
            kernel_size=(6, 6), stride=(2, 2), padding=2
        )
        # 16 -> 32 -> 64 -> 128 -> 256 -> 512

        self.Conv1 = nn.Conv2d(8, out_channels, (5, 5), stride=(1, 1), padding=2)
        self.act_final = nn.Sigmoid()

    def forward(self, x, label):
        latent_embedding = self.zlayer(x)
        cond_embedding = self.embedding_layer(label)
        merged = torch.cat([latent_embedding, cond_embedding], 1)
        out = self.act_z(merged)
        out = self.rearrange(out)
        out = self.Up_1(out)
        out = self.Up_2(out)
        out = self.Up_3(out)
        out = self.Up_4(out)
        out = self.Up_5(out)
        out = self.Conv1(out)
        out = self.act_final(out)
        return out


class Validator(nn.Module):
    def __init__(self, in_channels, n_parameters=1, n_z=256, n_layers=5, input_size=512, norm=False, pool=False):
        super().__init__()

        self.Down_0 = DownConvLayer(
            in_channels, 16, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=0
        )
        self.Down_1 = DownConvLayer(
            16, 32, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=1
        )
        self.Down_2 = DownConvLayer(
            32, 64, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=2
        )
        self.Down_3 = DownConvLayer(
            64, 128, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=3
        )
        self.Down_4 = DownConvLayer(
            128, 256, activation='leakyrelu',
            kernel_size=(6, 6), stride=(2, 2), padding=2,
            norm=norm, iteration=4
        )
        # 512 -> 256 -> 128 -> 64 -> 32 -> 16

        self.rearr_final = Rearrange('b z h w -> b (z h w)')
        # final spatial size = 16 x 16, channels = 256
        self.act_final = nn.Linear(16 * 16 * 256, n_parameters)
        self.latent = nn.Linear(16 * 16 * 256, n_z)

    def forward(self, x):
        x = self.Down_0(x)
        x = self.Down_1(x)
        x = self.Down_2(x)
        x = self.Down_3(x)
        x = self.Down_4(x)
        feats = self.rearr_final(x)
        output = self.act_final(feats)
        latent_z = self.latent(feats)
        return latent_z, output










#Old Code Below:

# from torch import nn
# from einops.layers.torch import Rearrange
# from collections import OrderedDict
# import torch

# class UpConvLayer(nn.Sequential):
#     def __init__(self, in_channels, out_channels, activation, norm=True, pool=True, **kwargs):
#         if activation == 'sigmoid':
#             activation = nn.Sigmoid()
#         elif activation == 'leakyrelu':
#             activation = nn.LeakyReLU(0.2, True)

#         layers = [nn.ConvTranspose2d(in_channels, out_channels, **kwargs)]
#         if norm:
#             layers += [nn.InstanceNorm2d(out_channels)]
#         if pool:
#             layers+=[nn.MaxPool2d((5,5), stride=(1,1), padding=2)]
#         layers += [activation]

#         super().__init__(*layers)


# class DownConvLayer(nn.Sequential):
#     def __init__(self, in_channels, out_channels, activation, iteration, norm=False, **kwargs):
#         if activation == 'sigmoid':
#             activation = nn.Sigmoid()
#         elif activation == 'leakyrelu':
#             activation = nn.LeakyReLU(0.2)

#         layers = {f'conv2d_{iteration}': nn.Conv2d(in_channels, out_channels, **kwargs)}
#         if norm:
#             layers[f'norm_{iteration}'] = nn.InstanceNorm2d(out_channels)
        
#         layers[f'act_{iteration}'] = activation

#         super().__init__(OrderedDict(nn.ModuleDict(layers)))

# class Generator(nn.Sequential):
#     def __init__(self, n_z, input_filt=512, norm=False, pool=False, n_layers=5, out_channels=3, final_size=256):
#         self.n_z = n_z

#         layers = []

#         prev_filt = input_filt
#         for _ in range(n_layers):
#             layers.append(UpConvLayer(prev_filt, int(prev_filt / 2), activation='leakyrelu', norm=norm, pool=pool,
#                                       kernel_size=(6, 6), stride=(2, 2), padding=2))
#             prev_filt = int(prev_filt / 2)

#         initial_size = final_size / 2 ** n_layers
#         if initial_size % 1 != 0:
#             raise ValueError(f"Cannot create a model to produce a {final_size} x {final_size} image with {n_layers} layers")

#         initial_size = int(initial_size)

#         super().__init__(
#             nn.Linear(n_z, initial_size * initial_size * input_filt),
#             nn.LeakyReLU(0.2, True),
#             Rearrange('b (h w z) -> b z h w', h=initial_size, w=initial_size, z=input_filt),
#             *layers,
#             nn.Conv2d(prev_filt, out_channels, (5, 5), stride=(1, 1), padding=2),
#             nn.Sigmoid()
#         )



# class Discriminator(nn.Sequential):
#     def __init__(self, in_channels, n_layers=5, input_size=256, norm=False, pool=False):
#         prev_filt = 8
#         layers = []
#         for i in range(n_layers):
#             layers.append(DownConvLayer(prev_filt if i > 0 else in_channels, prev_filt * 2, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=i))
#             prev_filt = prev_filt * 2
#             input_size = input_size / 2

#         super().__init__(
#             *layers,
#             Rearrange('b z h w -> b (z h w)'),
#             nn.Linear(int(input_size) * int(input_size) * prev_filt, 1)
#         )


# # # class Discriminator(nn.Sequential):
# # #     def __init__(self, in_channels, n_layers=5, input_size=256):
# # #         prev_filt = 8
# # #         modules = {}
# # #         for i in range(n_layers):
# # #             modules[f'Down_{i}'] = DownConvLayer(prev_filt if i > 0 else in_channels, prev_filt * 2, activation='leakyrelu',
# # #                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=False, iteration=i)
# # #             prev_filt = prev_filt * 2
# # #             input_size = input_size / 2
# # #         modules[f'rearr_final'] = Rearrange('b z h w -> b (z h w)')
# # #         modules[f'act_final'] = nn.Linear(int(input_size) * int(input_size) * prev_filt, 1)
# # #         module_dict = nn.ModuleDict(modules)

# # #         super().__init__( OrderedDict(module_dict)
            
# # #         )




# class DiscriminatorFeatures(nn.Module):
#     def __init__(self, in_channels, n_layers=5, input_size=256, norm=False, pool=False):
#         super().__init__()
        
#         self.Down_0 = DownConvLayer(in_channels, 16, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=0)
#         self.Down_1 = DownConvLayer(16, 32, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=1)
        
#         self.Down_2 = DownConvLayer(32, 64, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=2)
        
#         self.Down_3 = DownConvLayer(64, 128, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=3)
        
#         self.rearr_final = Rearrange('b z h w -> b (z h w)')
#         self.act_final = nn.Linear(6 * 6 * 128, 1)
        
        
#     def forward(self, x):
#         x = self.Down_0(x)
#         x = self.Down_1(x)
#         x = self.Down_2(x)
#         x = self.Down_3(x)
#         feats = self.rearr_final(x)
#         output = self.act_final(feats)
#         return output, feats
        
        

# class Encoder(nn.Sequential):
#     def __init__(self, in_channels , n_z = 128, n_layers = 5, input_size = 256, norm=False, pool=False):
#         prev_filt = 8
#         modules = {}
#         for i in range(n_layers):
#             modules[f'Down_{i}'] = DownConvLayer(prev_filt if i > 0 else in_channels, prev_filt * 2, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=False, pool=False, iteration=i)
#             prev_filt = prev_filt * 2
#             input_size = input_size / 2
#         modules[f'rearr_final'] = Rearrange('b z h w -> b (z h w)')
#         modules[f'act_final'] = nn.Linear(int(input_size) * int(input_size) * prev_filt, n_z)
#         module_dict = nn.ModuleDict(modules)

#         super().__init__( OrderedDict(module_dict)
            
#         )
        
# class ConcatLayer(nn.Module):
#     def __init__(self, n_e, n_e_feats, n_z, n_z_features):
#         super().__init__()
#         self.embed_layer = nn.Embedding(n_e, n_e_feats)
#         self.latent_layer = nn.Linear(n_z, n_z_features)

#     def forward(self, z_vec, cond):
#         X = self.latent_layer(z_vec)
#         Y = self.embed_layer(cond)
#         return torch.cat([X,Y], dim=1)

# class ConditionalDiscriminator(nn.Module):
#     def __init__(self, in_channels=1, n_embeddings=3, embedding_features=32, norm=False, pool=False):
#         super().__init__()
#         self.embedding_layer = nn.Linear(n_embeddings, embedding_features)
#         self.Down_0 = DownConvLayer(in_channels, 16, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=0)
#         self.Down_1 = DownConvLayer(16, 32, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=1)
        
#         self.Down_2 = DownConvLayer(32, 64, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=2)
        
#         self.Down_3 = DownConvLayer(64, 128, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=3)
        
#         self.rearr_final = Rearrange('b z h w -> b (z h w)')
#         self.act_final = nn.Linear((6 * 6 * 128)+embedding_features, 1)
        
#     def forward(self, x, label):
#         x = self.Down_0(x)
#         x = self.Down_1(x)
#         x = self.Down_2(x)
#         x = self.Down_3(x)
#         feats = self.rearr_final(x)
#         label_em = self.embedding_layer(label)
#         merged_feats = torch.cat([feats, label_em], 1)
#         output = self.act_final(merged_feats)
#         return output, feats
        
# class ConditionalGenerator(nn.Module):
#     def __init__(self, n_z, n_embeddings=3, input_filt=512, final_size=256,embedding_features=32, out_channels=1, norm=False, pool=False):
#         super().__init__()
#         self.n_z = n_z
#         self.zlayer = nn.Linear(n_z, (3 * 3 * 256) - embedding_features)
#         self.embedding_layer = nn.Linear(n_embeddings, embedding_features)
#         self.act_z = nn.LeakyReLU(0.2, True)
#         self.rearrange = Rearrange('b (h w z) -> b z h w', h=3, w=3, z=256)
        
#         self.Up_1 = UpConvLayer(256, 128, activation='leakyrelu', norm=norm, pool=pool,
#                                       kernel_size=(6, 6), stride=(2, 2), padding=2)
#         self.Up_2 = UpConvLayer(128, 64, activation='leakyrelu', norm=norm, pool=pool,
#                                       kernel_size=(6, 6), stride=(2, 2), padding=2)
#         self.Up_3 = UpConvLayer(64, 32, activation='leakyrelu', norm=norm, pool=pool,
#                                       kernel_size=(6, 6), stride=(2, 2), padding=2)
#         self.Up_4 = UpConvLayer(32, 16, activation='leakyrelu', norm=norm, pool=pool,
#                                       kernel_size=(6, 6), stride=(2, 2), padding=2)
#         self.Up_5 = UpConvLayer(16, 8, activation='leakyrelu', norm=norm, pool=pool,
#                                       kernel_size=(6, 6), stride=(2, 2), padding=2)
        
#         self.Conv1 = nn.Conv2d(8, out_channels, (5, 5), stride=(1, 1), padding=2)
#         self.act_final = nn.Sigmoid()
    
#     def forward(self, x, label):
#         latent_embedding = self.zlayer(x)
#         cond_embedding = self.embedding_layer(label)
#         merged = torch.cat([latent_embedding, cond_embedding], 1)
#         out = self.act_z(merged)
#         out = self.rearrange(out)
#         out = self.Up_1(out)
#         out = self.Up_2(out)
#         out = self.Up_3(out)
#         out = self.Up_4(out)
#         out = self.Up_5(out)
#         out = self.Conv1(out)
#         out = self.act_final(out)
#         return out
    
    
# class Validator(nn.Module):
#     def __init__(self, in_channels, n_parameters=1, n_z=256, n_layers=5, input_size=256, norm=False, pool=False):
#         super().__init__()
        
#         self.Down_0 = DownConvLayer(in_channels, 16, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=0)
#         self.Down_1 = DownConvLayer(16, 32, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=1)
        
#         self.Down_2 = DownConvLayer(32, 64, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=2)
        
#         self.Down_3 = DownConvLayer(64, 128, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=norm, pool=pool, iteration=3)
        
#         self.rearr_final = Rearrange('b z h w -> b (z h w)')
#         self.act_final = nn.Linear(6 * 6 * 128, n_parameters)
#         self.latent = nn.Linear(6 * 6 * 128, n_z)
        
        
#     def forward(self, x):
#         x = self.Down_0(x)
#         x = self.Down_1(x)
#         x = self.Down_2(x)
#         x = self.Down_3(x)
#         feats = self.rearr_final(x)
#         output = self.act_final(feats)
#         latent_z = self.latent(feats)
#         return latent_z, output