import torch.nn as nn
import torch
import segmentation_models_pytorch as smp

class ModelComplex(nn.Module):
    def __init__(self,radar_cnn,nasadem_cnn,jrc_cnn,unet):
        super(ModelComplex, self).__init__()
        # iterate the list and create a ModuleList of single Conv1d blocks
        self.radar_cnn = radar_cnn
        self.nasadem_cnn = nasadem_cnn
        self.jrc_cnn = jrc_cnn
        self.unet = unet

    def forward(self, inputs):
        # now you can build a single output from the list of convs
        radar_img = self.radar_cnn(inputs[:,:2,...])
        nasadem_img = self.nasadem_cnn(inputs[:,2:3,...])
        jrc_img = self.jrc_cnn(inputs[:,3:,...])
        # you can return a list, or even build a single tensor, like so
        img = torch.cat([radar_img,nasadem_img,jrc_img], dim=1)
        out = self.unet(img)
        return out


class MultiScaleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_sizes:list = (3, 5, 7)):
        super(MultiScaleConv2d, self).__init__()
        # iterate the list and create a ModuleList of single Conv1d blocks
        self.models = nn.ModuleList()
        if isinstance(kernel_sizes,int):
            self.models.append(nn.Conv2d(in_channels, out_channels, kernel_sizes, padding=int((kernel_sizes-1)/2)))    
        else:
            for k in kernel_sizes:
                self.models.append(nn.Conv2d(in_channels, out_channels, k, padding=int((k-1)/2)))

    def forward(self, inputs):
        # now you can build a single output from the list of convs
        out = [module(inputs) for module in self.models]
        # you can return a list, or even build a single tensor, like so
        return torch.cat(out, dim=1)
