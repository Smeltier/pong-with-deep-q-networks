import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        c, h, w = input_shape

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_output_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = x.float()
        conv_out = self.conv(x)
        conv_out = conv_out.reshape(conv_out.size(0), -1)
        return self.fc(conv_out)

    def _get_conv_out(self, shape):
        with torch.no_grad():
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.shape[1:]))
