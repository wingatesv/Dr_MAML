# from https://github.com/khtao/StainNet.git

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

STAINNET_WEIGHTS = '/content/New_MAML/data/StainNet-Public-centerUni_layer3_ch32.pth'

class StainNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_layer=3, n_channel=32, kernel_size=1):
        super(StainNet, self).__init__()
        model_list = []
        model_list.append(nn.Conv2d(input_nc, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
        model_list.append(nn.ReLU(True))
        for n in range(n_layer - 2):
            model_list.append(
                nn.Conv2d(n_channel, n_channel, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))
            model_list.append(nn.ReLU(True))
        model_list.append(nn.Conv2d(n_channel, output_nc, kernel_size=kernel_size, bias=True, padding=kernel_size // 2))

        self.rgb_trans = nn.Sequential(*model_list)

    def forward(self, x):
        return self.rgb_trans(x)

class StainNetTransform(object):
    def __init__(self):
        self.stainnet_model = StainNet().cuda()
        self.stainnet_model.load_state_dict(torch.load(STAINNET_WEIGHTS))

    def __call__(self, image):
        # Convert the image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Preprocess the image for StainNet
        image = self.norm(image)
        image = self.stainnet_model(image)
        image = self.un_norm(image)

        # Convert the processed image to PIL Image
        image = Image.fromarray(image)

        return image

    def norm(self, image):
        image = np.array(image).astype(np.float32)
        image = image.transpose((2, 0, 1))
        image = ((image / 255) - 0.5) / 0.5
        image = image[np.newaxis, ...]
        image = torch.tensor(image).cuda()
        return image

    def un_norm(self, image):
        assert image.dim() == 4 and image.shape[0] == 1, \
            "Expected input tensor to have shape (1, C, H, W), but got {}".format(image.shape)
        image = image.detach().cpu().numpy()[0]
        image = ((image * 0.5 + 0.5) * 255).astype(np.uint8).transpose((1, 2, 0))
        return image
