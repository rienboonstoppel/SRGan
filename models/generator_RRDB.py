import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualDenseBlock, self).__init__()

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, kernel_size=(3,3), padding='same')]
            if non_linearity:
                layers += [nn.ReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=1 * filters)
        self.b3 = block(in_features=2 * filters)
        self.b4 = block(in_features=3 * filters)
        self.b5 = block(in_features=4 * filters)

    def forward(self, x):
        c1 = self.b1(x)
        c2 = self.b2(c1)
        merge1 = torch.cat([c2, c1], 1)
        c3 = self.b3(merge1)
        merge2 = torch.cat([c3, c2, c1], 1)
        c4 = self.b4(merge2)
        merge3 = torch.cat([c4, c3, c2, c1], 1)
        gate = self.b5(merge3)
        output = torch.add(x, gate)
        return output

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.RDB = ResidualDenseBlock(filters)

    def forward(self, x):
        out1 = self.RDB(x)
        out2 = self.RDB(out1)
        out3 = self.RDB(out2)
        merge = torch.cat([out1, out2, out3], 1)
        return merge


class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=1):
        super(GeneratorRRDB, self).__init__()

        # First conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, filters, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(filters, filters, kernel_size=(3,3), padding='same'),
            nn.ReLU(),
        )
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Final output conv
        self.conv2 = nn.Conv2d(num_res_blocks * 3 * filters, channels, kernel_size=(3,3), padding='same')

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.res_blocks(out1)
        out3 = self.conv2(out2)
        out = torch.add(x, out3)
        return out