# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 14:02:51 2021

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
savepath = "./train_resultvgg16"   #风格迁移后的图片保存的文件夹路径
stylepath = "./data/style/8star.jpg"
trainpath = 'content_image.npy'
testpath = './data/test/3dancing.jpg'  #不在训练集里的图片，直接用模型做风格迁移
trainimg_path = './data/train/*.jpg' #用于读取图片文件名，保存图片时要用到，见trainimg_name_dic


EPOCH = 5     #训练总轮数(500) 括号内为实验最优结果时采用的参数，下同
trainsize = 4  #训练集的大小(883),取值[0,883];整个npy里有883张图片，全部用来训练可能太慢了
batch_size = 2  #(4)
savepic_strick, savemodel_strick, test_strick = 1, 1, 1 #保存训练集第一张图片迁移效果,保存模型, 测试的epoch频率
print_strick_iter = 100 #每个epoch中打印误差的iter频率
weight_c, weight_s = 5,2e5 #内容，风格损失权重(2,2e5)，默认为1，1e6
learning_rate = 1e-3    #(1e-3) Adam默认学习率为1e-3
savepath += "_%d_%0.0e_%0.0e" % (weight_c, weight_s, learning_rate) #保存模型的文件夹名带有参数标签，方便调参

#  所需的输出图像大小
# imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
imsize = 128    #这一版代码中不能改输出图片大小，因为做的npy里图片大小就是128*128

ispretrain = False #当接力训练时，设置为True，并且要在下两行写明input_img的读取路径和已训练轮数
model_loadpath = r'./result_temp\epoch1/model.pth'  #模型的保存路径
optim_loadpath = r'./result_temp\epoch1/optimizer.pth'
epoch_pretrain = 0  #这个变量保证保存各种对象（图片，模型等)的命名合理



def adjust_learning_rate(optimizer, epoch):
    strick = 1e9 #学习率衰减的epoch频率，如果不希望衰减，设置为1e9就行了（我就不信能训练1e9个epoch）
    ''' 每隔若干epoch，衰减学习率为原来的1/10'''
    lr = learning_rate * (0.1**(epoch//strick))
    if lr != learning_rate:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

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
    
    
def get_gram_matrix(f_map):
    n, c, h, w = f_map.shape
    f_map = f_map.reshape(n, c, h * w)
    # matmul是同torch.bmm矩阵的批乘法,(p,m,n)*(p,n,a) = (p,m,a)
    gram_matrix = torch.matmul(f_map, f_map.transpose(1, 2))
    return gram_matrix.div(n * c * h * w)


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1).to(device) 
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        a = models.vgg16(pretrained=True).features.to(device).eval()
        self.layer1 = a[:4]     #relu1_2
        self.layer2 = a[4:9]    #relu2_2
        self.layer3 = a[9:16]   #relu3_3
        self.layer4 = a[16:23]  #relu4_3
	
    """输出四层的特征图"""
    def forward(self, input_):
        normalized = (input_ - cnn_normalization_mean) / cnn_normalization_std
        out1 = self.layer1(normalized)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        return out1, out2, out3, out4


#生成文件夹，保存风格迁移过程量
def mkdir(path):    
    if not os.path.exists(path):
        os.makedirs(path)
mkdir(savepath)
mkdir(savepath + '/test')

loader = transforms.Compose([
    transforms.Resize([imsize,imsize]),
    transforms.ToTensor()])
unloader = transforms.ToPILImage()

def load_image(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0) # 增加一维，大小为1，即从a*b变成1*a*b
    return image.to(device, torch.float)

def Load_model(net, optimizer, model_loadpath, optim_loadpath):
    net.load_state_dict(torch.load(model_loadpath))
    optimizer.load_state_dict(torch.load(optim_loadpath))
    return net, optimizer



# 以下全是变量定义，直接去看train,test,main就行
# 1 模型
vgg16 = VGG16().to(device)
t_net = TransNet().to(device)
optimizer = torch.optim.Adam(t_net.parameters(), lr = learning_rate)
loss_func = nn.MSELoss().to(device)
if ispretrain ==True:
    t_net, optimizer = Load_model(t_net, optimizer, model_loadpath, optim_loadpath)

# 2 数据集
data_loader = np.load(trainpath)
data_loader = torch.tensor(data_loader).to(torch.float)/255 #因为当时做npy的时候，像素保存的范围是0-255
data_loader = data_loader.permute(0,3,1,2) #把通道数放到第二维

style_image = load_image(stylepath)
test_image = load_image(testpath)

# 3 图片命名
trainimg_name_dic = glob.glob(trainimg_path) #为了方便 保存的图片 命名
trainimg_name_dic = [e.split('\\')[-1][:-4] for e in trainimg_name_dic] #例4picasso.jpg->4picasso
testimg_name_dic = testpath.split('/')[-1][:-4]


"""4计算风格,并计算gram矩阵"""
s1_1, s2_1, s3_1, s4_1 = vgg16(style_image)
#expand是为了让batch中的每一张输入图片都有一个vgg处理后的风格层用来计算损失（s1[0,:,:]和s1[1,0,0]等是一样的）
s1 = get_gram_matrix(s1_1).detach().expand(batch_size,s1_1.shape[1],s1_1.shape[1])
s2 = get_gram_matrix(s2_1).detach().expand(batch_size,s2_1.shape[1],s2_1.shape[1])
s3 = get_gram_matrix(s3_1).detach().expand(batch_size,s3_1.shape[1],s3_1.shape[1])
s4 = get_gram_matrix(s4_1).detach().expand(batch_size,s4_1.shape[1],s4_1.shape[1])

s1_1, s2_1, s3_1, s4_1 = get_gram_matrix(s1_1).detach(),get_gram_matrix(s2_1).detach(),\
    get_gram_matrix(s3_1).detach(), get_gram_matrix(s4_1).detach()
    

    
real_trainsize = trainsize//batch_size*batch_size #防止下标越界,例883//2*2=882
maxiter = real_trainsize//batch_size
def train(epoch): 
    
    epoch_time_s = time.time()
    epoch_loss=np.zeros(3)  #三个元素分别表示 总,内容，风格损失
    t_net.train() 
    #分批训练
    for i in range(0,real_trainsize,batch_size):
        iter = i/batch_size+1    #表示这是第几代，例如trainsize,batch_size=883,2时,共有441代
        iter_time_s = time.time()
        
        """生成图片，计算损失"""
        content_image = data_loader[i:i+batch_size].to(device)
        image_g = t_net(content_image)
        
        out1, out2, out3, out4 = vgg16(image_g)
        # loss = loss_func(image_g, image_c)
        
        """计算风格损失"""
        loss_s1 = loss_func(get_gram_matrix(out1), s1)
        loss_s2 = loss_func(get_gram_matrix(out2), s2)
        loss_s3 = loss_func(get_gram_matrix(out3), s3)
        loss_s4 = loss_func(get_gram_matrix(out4), s4)
        
        loss_s = loss_s1+loss_s2+loss_s3+loss_s4

        """计算内容损失"""
        c1, c2, c3, c4 = vgg16(content_image)

        # loss_c1 = loss_func(out1, c1.detach())
        loss_c2 = loss_func(out2, c2.detach())  #relu2_2
        # loss_c3 = loss_func(out3, c3.detach())
        # loss_c4 = loss_func(out4, c4.detach())
        

        """总损失"""
        loss = weight_c*loss_c2 + weight_s * loss_s
        
        
        """清空梯度、计算梯度、更新参数"""
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        """保存生成的图片"""
        imgs_g = image_g if iter==1 else torch.cat((imgs_g, image_g), axis=0)
        
        epoch_loss = epoch_loss+[loss.item(),weight_c*loss_c2.item(),weight_s*loss_s.item()]
        iter_time_e = time.time()
        if iter % print_strick_iter == 0:
            print('EPOCH:%d, Iter:%d, total_loss:%0.2f, content_loss:%0.2f, \
                  style_loss:%0.2e, iter_time_use:%0.2fs' %
                  (epoch, iter, loss.item(), weight_c*loss_c2.item(), weight_s*loss_s.item(), \
                   iter_time_e-iter_time_s) )
            
    epoch_time_e = time.time()
    epoch_loss= epoch_loss/maxiter
    print('--------------------EPOCH%d completed--------------------' %(epoch))
    print('total_loss:%0.2f, content_loss:%0.2f, \
          style_loss:%0.2e, epoch_time_use:%0.2fmin\n' %
          (epoch_loss[0], epoch_loss[1], epoch_loss[2], \
           (epoch_time_e-epoch_time_s)/60 ) )
    return imgs_g, epoch_loss   #返回生成的图片和损失

def tst(epoch):
    t_net.eval()  
    with torch.no_grad():   
        epoch_loss=np.zeros(3)  #三个元素分别表示 总,内容，风格损失
        """生成图片，计算损失"""
        content_image = test_image
        image_g = t_net(content_image)
        out1, out2, out3, out4 = vgg16(image_g)
        # loss = loss_func(image_g, image_c)
        
        """计算风格损失"""
        loss_s1 = loss_func(get_gram_matrix(out1), s1_1)
        loss_s2 = loss_func(get_gram_matrix(out2), s2_1)
        loss_s3 = loss_func(get_gram_matrix(out3), s3_1)
        loss_s4 = loss_func(get_gram_matrix(out4), s4_1)
        loss_s = loss_s1+loss_s2+loss_s3+loss_s4
        
        """计算内容损失"""
        c1, c2, c3, c4 = vgg16(content_image)
        
        # loss_c1 = loss_func(out1, c1.detach())
        loss_c2 = loss_func(out2, c2.detach())  #conv2_2
        # loss_c3 = loss_func(out3, c3.detach())
        # loss_c4 = loss_func(out4, c4.detach())
        
        """总损失"""
        loss = weight_c*loss_c2 + weight_s * loss_s
        epoch_loss = epoch_loss+[loss.item(),weight_c*loss_c2.item(),weight_s*loss_s.item()]
        print('--------------------TESTing--------------------')
        print('EPOCH:%d,total_loss:%0.2f, content_loss:%0.2f, \
              style_loss:%0.2e\n' %(epoch, epoch_loss[0], epoch_loss[1], epoch_loss[2]))
        
        img_savepath = savepath + '/test/' + testimg_name_dic + '_epoch'+ str(epoch)+'.jpg'
        img_t = unloader(image_g[0])
        # img_t.show()
        img_t.save(img_savepath)

def Save_model(net, optimizer, model_savepath):  
    torch.save(net.state_dict(), model_savepath + '/model.pth')
    torch.save(optimizer.state_dict(), model_savepath + '/optimizer.pth')
    # 记录损失
    file = open(model_savepath + '/loss.txt', 'w')
    file.write('total_loss:%0.2f, content_loss:%0.2f, style_loss:%0.2e\n' %
               (epoch_loss[0], epoch_loss[1], epoch_loss[2]))
    file.close()
         

    
if __name__ == '__main__':
    total_time_s = time.time()
    
    for epoch in range(1,EPOCH+1): 
        
        #训练
        imgs_g, epoch_loss = train(epoch)  #返回了所有生成的图片
          
        #测试         
        if epoch % test_strick==0:
            tst(epoch)
         
        #保存训练图片
        if epoch % savepic_strick==0:  
#            img_id = real_trainsize-batch_size #最后一个iter的第一张图片在883张图片中的id
            img_id = 0
            img_savepath = savepath + '/' + trainimg_name_dic[img_id] + \
                '_epoch'+ str(epoch+epoch_pretrain)+'.jpg'
            img_t = unloader(imgs_g[img_id])
            # img_t.show()
            img_t.save(img_savepath)
         
        #保存模型
        if epoch % savemodel_strick == 0:  
            model_savepath = savepath + '/epoch' + str(epoch+epoch_pretrain)
            mkdir(model_savepath)
            # '_loss' + str(epoch_loss[0]) + '.pth'
            Save_model(t_net, optimizer, model_savepath)
        
        # adjust_learning_rate(optimizer, epoch)
        
    
    total_time_e = time.time()
    print('total use %0.2fmin' % ((total_time_e-total_time_s)/60))
   
