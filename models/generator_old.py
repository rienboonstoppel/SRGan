import torch
import torch.nn as nn

class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=1 * filters)
        self.b3 = block(in_features=2 * filters)
        self.b4 = block(in_features=3 * filters)
        self.b5 = block(in_features=4 * filters)#, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        out1 = self.b1(inputs)
        out2 = self.b2(out1)
        cat1 = torch.cat([out2, out1], 1)
        out3 = self.b3(cat1)
        cat2 = torch.cat([out3, out2, out1], 1)
        out4 = self.b4(cat2)
        cat3 = torch.cat([out4, out3, out2, out1], 1)
        out5 = self.b5(cat3)

        return out5 + x

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.drb = DenseResidualBlock(filters)

    def forward(self, x):
        out1 = self.drb(x)
        out2 = self.drb(out1)
        out3 = self.drb(out2)
        cat = torch.cat([out1, out2, out3], 1)
        return cat


class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=1):
        super(GeneratorRRDB, self).__init__()

        # First conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Final output block
        self.conv2 = nn.Sequential(
            nn.Conv2d(num_res_blocks * 3 * filters, channels, kernel_size=3, stride=1, padding=1),
            # nn.Linear(filters, filters),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.res_blocks(out1)
        out3 = self.conv2(out2)
        out = torch.add(x, out3)
        return out