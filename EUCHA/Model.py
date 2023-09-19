import torch.nn as nn
import torch


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Base_Model(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, use_dropout=False, padding_type='reflect', n_blocks=6,):
        super(Base_Model,self).__init__()
        print(input_nc,output_nc)
        
        self.conv1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0))


        self.down1 = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(ngf*2),
                                   nn.ReLU(True))

        self.down2 = nn.Sequential(nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(ngf * 4),
                                   nn.ReLU(True))
        self.down3 = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(ngf * 8),
                                   nn.ReLU(True))

        self.cfnet1 = CFNet(ngf)
        self.cfnet2 = CFNet(ngf * 2)
        self.cfnet3 = CFNet(ngf * 4)

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU(True)
        model_res = []
        for i in range(n_blocks):
            model_res += [ResnetBlock(ngf * 8, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        self.model_res = nn.Sequential(*model_res)

       
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 8),
            nn.ReLU(True))

        self.up2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
                                 nn.InstanceNorm2d(ngf*4),
                                 nn.ReLU(True))

        self.up3 = nn.Sequential(
                                  nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
                                  nn.Tanh())
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                 nn.Tanh())

        self.feture1 = FeatureInteraction(ngf * 8, ngf*16,m=-0.50)
        self.feture2 = FeatureInteraction(ngf * 4, ngf * 8,m=-0.50)
        self.feture3 = FeatureInteraction(ngf * 4, ngf*8,m=-0.50)
        self.feture4 = FeatureInteraction(ngf * 2 , ngf*4,m=-0.50)
        self.feture5 = FeatureInteraction(ngf * 2 , ngf*4,m=-0.50)
        self.feture6 = FeatureInteraction(ngf, ngf*2,m=-0.50)

        self.pa1 = PALayer(ngf*8)
        self.pa2 = PALayer(ngf*4)
        self.pa3 = PALayer(ngf*2)

        self.ca1 = CALayer(ngf*8)
        self.ca2 = CALayer(ngf*4)
        self.ca3 = CALayer(ngf*2)

    def forward(self, input):
        x = self.conv1(input)
        x_down1 = self.down1(x)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x_cf1, p41, p51 = self.cfnet1(x)
        x_cf2, p42, p52 = self.cfnet2(x_down1)
        x_cf3, p43, p53 = self.cfnet3(x_down2)

        x6 = self.model_res(x_down3)  
        x6 = self.ca1(x6)
        x6 = self.pa1(x6)

        x6 = self.up1(x6)
        p5 = self.feture1(p52,p53)
        p5 = self.feture2(p51,p5)

        x6 = x6 + p5
        x6 = self.ca2(x6)
        x6 = self.pa2(x6)

        x6 = self.up2(x6)
        p4 = self.feture3(p42,p43)
        p4 = self.feture4(p41,p4)
        x6 = x6 + p4
        x6 = self.ca3(x6)
        x6 = self.pa3(x6)

        x6 = self.up3(x6)
        p3 = self.feture5(x_cf2,x_cf3)
        p3 = self.feture6(x_cf1,p3)
        x6 = x6 + p3
        x = self.conv2(x6)
        return x


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class DehazeBlock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()

        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)   
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        res = self.act1(self.conv1(x))
        res = self.conv2(res)

        return res

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class PALayer1(nn.Module):
    def __init__(self, channel):
        super(PALayer1, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return y

class CALayer1(nn.Module):
    def __init__(self, channel):
        super(CALayer1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return y


class FeatureInteraction(nn.Module):
    def __init__(self,low_c,high_c,m = -0.50):
        super(FeatureInteraction, self).__init__()
        modules_ca = []
        modules_pa = []
        self.low_conv = nn.Sequential(nn.Conv2d(low_c,low_c,kernel_size=3, stride=1, padding=1))
        self.high_conv = nn.Sequential(nn.ConvTranspose2d(high_c,low_c, kernel_size=3,stride=2,padding=1,output_padding=1))
        modules_ca.append(CALayer1(channel=low_c))
        modules_pa.append(PALayer1(channel=low_c))
        self.ca = nn.Sequential(*modules_ca)
        self.pa = nn.Sequential(*modules_pa)

        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()


    def forward(self,low_level,high_level):
        low = self.low_conv(low_level)
        high = self.high_conv(high_level)
        low_pa = self.pa(low)
        high_ca = self.ca(high)
        a = low * high_ca
        b = high * low_pa

        mix_factor = self.mix_block(self.w)
        out = a * mix_factor.expand_as(a) + b * (1 - mix_factor.expand_as(b))
        #c = a+b
        return out


class CFNet(nn.Module):
    def __init__(self,inplane):
        super(CFNet, self).__init__()
        self.Block1 = DehazeBlock(default_conv,inplane ,kernel_size=3) 
        self.Block2 = DehazeBlock(default_conv,inplane*2,kernel_size=3)  
        self.Block3 = DehazeBlock(default_conv,inplane*4,kernel_size=3) 
        self.down1 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(inplane, inplane*2, kernel_size=7, stride=2,padding=0),
                                   nn.InstanceNorm2d(inplane*2),
                                   nn.ReLU(True))
        self.down2 = nn.Sequential(nn.Conv2d(inplane*2, inplane * 4, kernel_size=3, stride=2, padding=1),
                                   nn.InstanceNorm2d(inplane*4),
                                   nn.ReLU(True))
        self.up = nn.Sequential(
            nn.ConvTranspose2d(inplane * 4, inplane*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(inplane * 2),
            nn.ReLU(True))

        self.init = FeatureInteraction(inplane*2 , inplane * 4,m=-0.50)
        self.layer1 = FeatureInteraction(inplane ,inplane * 2,m=-0.50)


    def forward(self , x):   
            x1 = self.Block1(x)
            x2 = self.down1(x1)
            x3 = self.Block2(x2)
            x4 = self.down2(x3)
            x5 = self.Block3(x4)
            p5 = x5
            p4 = x3 + self.up(x5)
            x_init = self.init(x3, x5)
            x_layer1 = self.layer1( x1,x_init)
            return x_layer1 ,p4 ,p5
            

class Discriminator(nn.Module):
    def __init__(self, bn=False, ngf=64):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ngf, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 2),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, padding=0),
            nn.InstanceNorm2d(ngf * 4) if not bn else nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(ngf * 4) if not bn else nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=3, padding=0),
            nn.BatchNorm2d(ngf * 8) if bn else nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(ngf * 8) if bn else nn.InstanceNorm2d(ngf * 8),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ngf * 8, ngf * 16, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ngf * 16, 1, kernel_size=1)

        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))