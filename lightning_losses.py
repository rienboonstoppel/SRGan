from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg

class GANLoss(nn.Module):
    """
    PyTorch module for GAN loss.
    """
    def __init__(self,
                 gan_mode,
                 real_label=1.0,
                 fake_label=0.0):

        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))

        self.gan_mode = gan_mode

        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgan':
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, sample, target_is_real):
        if target_is_real:
            target_tensor = self.real_label.expand_as(sample)
            target_tensor = target_tensor.type_as(sample)
        else:
            target_tensor = self.fake_label.expand_as(sample)
            target_tensor = target_tensor.type_as(sample)
        return target_tensor

    def forward(self, sample, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(sample, target_is_real)
            loss = self.loss(sample, target_tensor)
        elif self.gan_mode == 'wgan':
            if target_is_real:
                loss = - sample.mean()
            else:
                loss = sample.mean()
        return loss

class GradientPenalty(nn.Module):
    """
    PyTorch module for Gradient Penalty
    """
    def __init__(self,
                 critic,
                 fake_label=0.0,
                 ):
        super(GradientPenalty, self).__init__()

        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.critic = critic

    def forward(self, samples_real, samples_fake):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand((samples_real.size(0), 1, 1, 1)).type_as(samples_real)

        # mix/interpolate images
        interpolates = (alpha * samples_real + (1 - alpha) * samples_fake).requires_grad_(True)
        interpolates = interpolates.type_as(samples_real)

        # Calculate the critic's scores on the mixed images
        interpolates_scores = self.critic(interpolates)

        target_tensor = self.fake_label.expand_as(interpolates_scores)
        target_tensor = target_tensor.type_as(interpolates_scores)

        # Take the gradient of the scores with respect to the images
        gradients = torch.autograd.grad(
            inputs=interpolates,
            outputs=interpolates_scores,
            grad_outputs=target_tensor,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Flatten the gradients so that each row captures one image
        gradients = gradients.view(gradients.size(0), -1).type_as(samples_real)

        # Calculate the magnitude of every row
        gradients_norm = gradients.norm(2, dim=1)

        # Penalize the mean squared distance of the gradient norms from 1
        gradient_penalty = torch.mean((gradients_norm - 1) ** 2)

        return gradient_penalty


class VGGLoss(nn.Module):
    """
    PyTorch module for VGG loss.
    Parameter
    ---------
    net_type : str
        type of vgg network, i.e. `vgg16` or `vgg19`.
    layer : str
        layer where the mean squared error is calculated.
    rescale : float
        rescale factor for VGG Loss
    """
    def __init__(self, net_type='vgg19', layer='relu2_2', rescale=0.006):
        super(VGGLoss, self).__init__()

        if net_type == 'vgg16':
            assert layer in ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
            self.__vgg_net = VGG16()
            self.__layer = layer
        elif net_type == 'vgg19':
            assert layer in ['relu1_2', 'relu2_2', 'relu3_4',
                             'relu4_4', 'relu5_4']
            self.__vgg_net = VGG19()
            self.__layer = layer

        self.register_buffer(
            name='vgg_mean',
            tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]],
                                requires_grad=False)
        )
        self.register_buffer(
            name='vgg_std',
            tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]],
                                requires_grad=False)
        )
        self.register_buffer(   # to balance VGG loss with other losses.
            name='rescale',
            tensor=torch.tensor(rescale, requires_grad=False)
        )

    def __normalize(self, img):
        img = img.sub(self.vgg_mean.detach())
        img = img.div(self.vgg_std.detach())
        return img

    def forward(self, x, y):
        """
        Paramenters
        ---
        x, y : torch.Tensor
            input or output tensor. they must be normalized to [0, 1].
        Returns
        ---
        out : torch.Tensor
            mean squared error between the inputs.
        """
        norm_x = self.__normalize(x)
        norm_y = self.__normalize(y)
        feat_x = getattr(self.__vgg_net(norm_x), self.__layer)
        feat_y = getattr(self.__vgg_net(norm_y), self.__layer)
        out = F.mse_loss(feat_x, feat_y) * self.rescale
        return out


class VGG16(nn.Module):
    """
    Blockwise pickable VGG16.
    This code is inspired by https://gist.github.com/crcrpar/a5d46738ffff08fc12138a5f270db426
    """
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)

        return out


class VGG19(nn.Module):
    """
    Blockwise pickable VGG19.
    This code is inspired by https://gist.github.com/crcrpar/a5d46738ffff08fc12138a5f270db426
    """
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2',
                           'relu3_4', 'relu4_4', 'relu5_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2,
                          h_relu3_4, h_relu4_4, h_relu5_4)

        return out


class TVLoss(nn.Module):
    """
    Total Variation Loss.
    This code is copied from https://github.com/leftthomas/SRGAN/blob/master/loss.py
    """
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]