import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
from torchvision import models

class I2I(BaseModel):
    def name(self):
        return 'I2I'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.netG= networks.define_attunet(opt.input_nc, opt.output_nc, opt.init_type, self.gpu_ids)
        
        self.vgg=VGG16().cuda()
        
        self.hist_loss_fn = HistogramLoss()

        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = self.input_A
        self.fake_B, self.attention_map = self.netG(self.real_A)
        self.real_B = self.input_B

    
    def test(self):
        self.real_A = self.input_A
        self.fake_B = self.netG(self.real_A)
        self.real_B = self.input_B

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv

        self.loss_D.backward()

        
    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_adv

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        
        self.VGG_real=self.vgg(self.real_B.expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        self.VGG_fake=self.vgg(self.fake_B.expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        self.VGG_loss=self.criterionL1(self.VGG_fake,self.VGG_real)* self.opt.lambda_vgg
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.VGG_loss
        
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('G_VGG', self.VGG_loss.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
    
    def get_current_visuals_3(self):
        attention_map1 = util.tensor2im_3(self.attention_map)
        return OrderedDict([('attention_map1', attention_map1)])
    
    
    def get_current_visuals_2(self):
        real_A = util.tensor2im_2(self.real_A.data)
        fake_B = util.tensor2im_2(self.fake_B.data)
        real_B = util.tensor2im_2(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, X):
        h_relu1 = self.stage1(X)
        h_relu2 = self.stage2(h_relu1)       
        return h_relu2
    
    
class HistogramLoss(torch.nn.Module):
    def __init__(self, bins=256):
        super(HistogramLoss, self).__init__()
        self.bins = bins

    def forward(self, fake, real):

        hist_fake = torch.histc(fake, bins=self.bins, min=-1, max=1)
        hist_real = torch.histc(real, bins=self.bins, min=-1, max=1)

        hist_fake = hist_fake / hist_fake.sum()
        hist_real = hist_real / hist_real.sum()

        loss = F.mse_loss(hist_fake, hist_real)
        return loss


