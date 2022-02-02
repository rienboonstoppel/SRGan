import torch
from torch import nn

# edge loss function 1 (mean of sobel in patch)
def edge_loss1(out, target):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same', padding_mode='replicate', bias=False)
    convy = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same', padding_mode='replicate', bias=False)

    weights_x = torch.tensor(
        [[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]],
        device=device,
        dtype=torch.float32
    )

    weights_y = torch.tensor(
        [[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]],
        device=device,
        dtype=torch.float32
    )

    convx.weight = nn.Parameter(weights_x)
    convy.weight = nn.Parameter(weights_y)

    g1_x = convx(out)
    g2_x = convx(target)
    g1_y = convy(out)
    g2_y = convy(target)
    
    g_1 = torch.sqrt(g1_x * g1_x + g1_y * g1_y + 1e-8)  # edge map prediction
    g_2 = torch.sqrt(g2_x * g2_x + g2_y * g2_y + 1e-8)  # edge map target
    
    diff = g_1 - g_2
    diff2 = diff * diff
    
    eloss = torch.mean(diff2)  # MSE of Sobel edge difference

    return eloss


# edge loss function 2 (mean of sobel * abs pixel difference in patch)
def edge_loss2(out, target):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same', padding_mode='replicate', bias=False)
    convy = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same', padding_mode='replicate', bias=False)
    weights_x = torch.tensor(
        [[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]],
        device=device,
        dtype=torch.float32
    )

    weights_y = torch.tensor(
        [[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]],
        device=device,
        dtype=torch.float32
    )

    convx.weight = nn.Parameter(weights_x)
    convy.weight = nn.Parameter(weights_y)

    g_x = convx(target)  # gradient in x
    g_y = convy(target)  # gradient in y
    
    g_tot = torch.sqrt(g_x * g_x + g_y * g_y + 1e-8)  # Sobel edge map
    
    pixel_diff = torch.abs(out - target)    # absolute pixel difference
    tloss = torch.mean(g_tot * pixel_diff)  # mean of (pixel difference * edge_strength)

    return tloss


# edge loss function 2 (max difference * mean sobel in patch)
def edge_loss3(out, target):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    convx = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same', padding_mode='replicate', bias=False)
    convy = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding='same', padding_mode='replicate', bias=False)
    weights_x = torch.tensor(
        [[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]],
        device=device,
        dtype=torch.float32
    )

    weights_y = torch.tensor(
        [[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]],
        device=device,
        dtype=torch.float32
    )

    convx.weight = nn.Parameter(weights_x)
    convy.weight = nn.Parameter(weights_y)

    g_x = convx(target)  # gradient in x
    g_y = convy(target)  # gradient in y
    
    g_tot = torch.sqrt(g_x * g_x + g_y * g_y + 1e-8)  # Sobel edge map
    
    pixel_diff = torch.abs(out - target)     # absolute pixel difference
    max_diff = float(torch.max(pixel_diff))  # max error
    tloss = max_diff * torch.mean(g_tot)     # max error * edge_strength

    return tloss