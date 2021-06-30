# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:51:48 2021

@author: Lenovo_ztk
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import glob,os,time, random
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''方便的参数设置'''
savepath = "./result"   #风格迁移后的图片保存的文件夹路径
testpath = './data/test/*'  #不在训练集里的图片，直接用模型做风格迁移
model_loadpath = './model.pth'  #模型的保存路径

#  输出图像大小
imsize = 128    #这一版代码中不能改输出图片大小，因为模型训练图片大小就是128*128


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels,channels,3,1,1,bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels,channels,3,1,1,bias=False),
            nn.InstanceNorm2d(channels),
            )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.layer(x)+x)

class TransNet(nn.Module):

    def __init__(self):
        super(TransNet, self).__init__()
        self.layer = nn.Sequential(  #3*
            nn.Conv2d(3, 32, 9, 1, 4, bias=False),   #32*
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64,3,2,1, bias=False),     #64*
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),    #128*
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            nn.Upsample(scale_factor=2, mode='nearest'),#128*
            nn.Conv2d(128,64,3,1,1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),#64*
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,3,9,1,4),  #3*
            nn.Sigmoid()    #将像素值映射到[0,1]
        )

    def forward(self, x):
        return self.layer(x)
    

#生成文件夹，保存风格迁移过程量
def mkdir(path):    
    if not os.path.exists(path):
        os.makedirs(path)
mkdir(savepath)

loader = transforms.Compose([
    transforms.Resize([imsize,imsize]),
    transforms.ToTensor()])
unloader = transforms.ToPILImage()

def load_image(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = image.convert('RGB')
    image = loader(image).unsqueeze(0) # 增加一维，大小为1，即从a*b变成1*a*b
    return image.to(device, torch.float)

def Load_model(net, model_loadpath):
    if device == torch.device('cpu'):
        net.load_state_dict(torch.load(model_loadpath , map_location='cpu'))
    else :
        net.load_state_dict(torch.load(model_loadpath))
    return net



# 以下全是变量定义，直接去看main就行
# 1 模型
t_net = TransNet().to(device)
t_net = Load_model(t_net, model_loadpath)


# 3 图片命名
testimg_name_dic = glob.glob(testpath) #为了方便 保存的图片 命名
saveimg_name_dic = [e.split('\\')[-1] for e in testimg_name_dic] #例4picasso.jpg

    
if __name__ == '__main__':    
    for i,imgdir in enumerate (testimg_name_dic): 
        input = load_image(imgdir)
        
        output = t_net(input)
        output = unloader( output.squeeze(0) )
        
        output.save(savepath + '/' + saveimg_name_dic[i])
         
   
