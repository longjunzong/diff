import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init

class NoiseAdapter(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size == 3:
            self.feat = nn.Sequential(
                # Bottleneck(channels, channels, reduction=8),
                GcBlock(channels, channels, reduction=8),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            self.feat = nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
            )
        self.pred = nn.Linear(channels, 2)

    def forward(self, x):
        x = self.feat(x).flatten(1)
        x = self.pred(x).softmax(1)[:, 0]
        return x
    

class DiffusionModel(nn.Module):
    def __init__(self, channels_in, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.time_embedding = nn.Embedding(1280, channels_in)

        if kernel_size == 3:
            self.pred = nn.Sequential(
                # Bottleneck(channels_in, channels_in),
                # Bottleneck(channels_in, channels_in),
                GcBlock(channels_in, channels_in),
                GcBlock(channels_in, channels_in),
                nn.Conv2d(channels_in, channels_in, 1),
                nn.BatchNorm2d(channels_in)
            )
        else:
            self.pred = nn.Sequential(
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1),
                nn.BatchNorm2d(channels_in),
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1)
            )

    def forward(self, noisy_image, t):
        if t.dtype != torch.long:
            t = t.type(torch.long)
        feat = noisy_image
        feat = feat + self.time_embedding(t)[..., None, None]
        ret = self.pred(feat)
        return ret


class AutoEncoder(nn.Module):
    def __init__(self, channels, latent_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, latent_channels, 1, padding=0),
            nn.BatchNorm2d(latent_channels)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, channels, 1, padding=0),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        return hidden, out

    def forward_encoder(self, x):
        return self.encoder(x)
    

class DDIMPipeline:
    '''
    Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    '''

    def __init__(self, model, scheduler, noise_adapter=None, solver='ddim'):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.noise_adapter = noise_adapter
        self._iter = 0
        self.solver = solver

    def __call__(
            self,
            batch_size,
            device,
            dtype,
            shape,
            feat,
            generator = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            proj = None
    ):

        # Sample gaussian noise to begin loop
        image_shape = (batch_size, *shape)

        if self.noise_adapter is not None:
            noise = torch.randn(image_shape, device=device, dtype=dtype)
            timesteps = self.noise_adapter(feat)
            image = self.scheduler.add_noise_diff2(feat, noise, timesteps)
        else:
            noise = torch.randn(image_shape, device=device, dtype=dtype)
            timesteps = torch.ones(batch_size)*0.25
            timesteps = timesteps.to(feat)
            image = self.scheduler.add_noise_diff2(feat, noise, timesteps)
            image = feat

        # set step values
        self.scheduler.set_timesteps(num_inference_steps*2)

        for t in self.scheduler.timesteps[len(self.scheduler.timesteps)//2:]:
            noise_pred = self.model(image, t.to(device))

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                noise_pred, t, image, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample'] 
                
        self._iter += 1        
        return image


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        return out + x


class GcBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.LayerNorm([in_channels // 2, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.reset_parameters()

    def spatial_pool(self, x):
        b, c, w, h = x.size()
        input_x = x
        input_x = input_x.view(b, c, h*w)
        input_x = input_x.unsqueeze(1)
        context_mask = self.conv_mask(x)
        context_mask = context_mask.view(b, 1, h*w)
        context_mask = F.softmax(context_mask, dim=2)
        context_mask = context_mask.unsqueeze(-1)
        context = torch.matmul(input_x, context_mask)
        context = context.view(b, c, 1, 1)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        channel_add = self.channel_add_conv(context)
        x = x + channel_add
        out = self.block(x)
        return out + x

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    def reset_parameters(self):
        kaiming_init(self.conv_mask, mode='fan_in')
        self.conv_mask.inited = True
        self.last_zero_init(self.channel_add_conv)