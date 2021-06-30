# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy, os, time, sys, cv2

''''参数设置'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#1
frame_path = "./frame" #对gif逐帧提取后的图片的保存路径
save_path = "./train_image1" #迭代过程中图片的保存路径
img_save_path = "./result1" #风格迁移后图片的保存路径

#2 输入图片路径
content_path = "./content/2.jpg"
style_path = "./style/001.jpg"

#3 输出视频路径
video_path = "./video"
video_dir = video_path + "/" + "001.mp4"
gif_dir = video_path + "/" + "001.gif"

#4 默认SIZE_flag为False，输出图片大小为输入内容图的大小
# 如果要自己设置输出图片大小，请修改SIZE_flag为True
# 在height、width设置为想要设置的值
SIZE_flag = False
height = 0
width = 0
#5 帧率
fps = 24

#6
EPOCH = 500
# 在这里设置保存图片的轮次
# 例如save_range_s, save_range_e = 0, 300,保存训练次数为1~300的图片
save_range_s, save_range_e = 0, 500

#7 根据输入的内容是否为gif设置输出类型
GIF_flag = content_path.endswith(".gif")
# 图片显示开关
PIL_flag = False

#8 当以已训练了若干轮次的图片作为input_img时，设置为True，并且要在下两行写明input_img的读取路径和已训练轮数
pre_train = False
input_path = None

#9 设置VGG模型
set_layer = False
# =============================================================================
# 可以指定用于计算内容损失和风格损失的vgg卷积层输出的具体层数
# 如果ischange_layer = False，请忽略下面两行。
# 一般认为内容层要指定地往后一些（vgg靠近输出的特征包含更多内容信息),风格层指定的要往前一些（浅一些）
# # 数字越大表示层数越深，表示经过的卷积层越多，表示越接近vgg的输出层
# # 可以选择conv_1~conv16不等
# 下面这套8;2-4-6-8-12是我从知乎的一篇文章里近似出来的。默认为4;1-2-3-4-5,来源于pytorch官方文档的
# 但效果似乎更差些，因为风格损失层靠后了，保留了过多风格图片的内容
# =============================================================================
content_layers_setting = ['conv_8']
style_layers_setting = ['conv_2', 'conv_4', 'conv_6', 'conv_8', 'conv_12']


# 依据路径打开图片，增加虚拟的一维（batchsize），转化为tensor
def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)  # 增加一维，大小为1，即从a*b变成1*a*b
    return image.to(device, torch.float)


# 输入tensor，带标题地以PIL形式输出
def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# 输入tensor，以PIL形式保存图片
def imsave(tensor, savepath):
    img = tensor.cpu().clone()
    img = img.squeeze(0)
    img = unloader(img)
    img.save(savepath)


# 生成文件夹，保存风格迁移过程量
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


''''定义内容损失和风格损失'''


class ContentLoss(nn.Module):
    """一个ContentLoss对象其实对应内容图片的一个卷积层输出"""

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        '''真实内容确实不能更新，需要在计算图中设置为“不需要梯度”'''
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()  # target当然只需要计算一次

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    # 这里为了归一正则，除以了特征层的元素个数，
    # 别的代码有“除以元素个数的平方”的现象，不同手段是可选的，都合理
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()  # 格雷姆矩阵当然只需要计算一次

    def forward(self, input):
        G = gram_matrix(input)  # G的大小为3*3,64*64或128*128等
        self.loss = F.mse_loss(G, self.target)
        return input


''''导入模型'''
cnn = models.vgg19(pretrained=True).features.to(device).eval()
# 这个开始的归一化是因为导入的vgg19训练时将图片做了如下的归一化
# 这里为了要让vgg正常做特征提取，所以也对输入到vgg的图片归一化
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.as_tensor(mean).view(-1, 1, 1)
        self.std = torch.as_tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

''''将内容风格损失层插入到vgg19中'''


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    # 展示vgg19的结构
    # print(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)
    # 展示加入内容和风格损失层后模型的结构
    # print(model)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    # 把最后一个内容或风格损失层后的层删掉(因为用不上)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


# if you want to use white noise instead uncomment the below line:
# input_img = torch.randn(content_img.data.size(), device=device)


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


''''训练'''


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img,
                       content_layers=content_layers_default,
                       style_layers=style_layers_default,
                       num_steps=EPOCH, style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    model, style_losses, content_losses = \
        get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers, style_layers)
    optimizer = get_input_optimizer(input_img)

    run = [0]

    while run[0] < num_steps:
        def closure():
            epoch_start_T = time.time()

            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            epoch_end_T = time.time()
            if run[0] % 20 == 0:
                print("轮次{}:".format(run))
                print('风格图损失: {:4f} | 内容图损失: {:4f} | 本轮训练用时: {:2f}秒'.format(
                    style_score.item(), content_score.item(), (epoch_end_T - epoch_start_T)))
                print()
            # print("epoch{}:".format(run))
            # print('Style Loss : {:4f} | Content Loss: {:4f} | Time of epoch: {:2f}S'.format(
            #     style_score.item(), content_score.item(), (epoch_end_T - epoch_start_T)))
            # print()

            mkdir(save_path)
            if save_range_s <= run[0] <= save_range_e:
                impath = save_path + '/' + str(run[0]).zfill(3) + '.png'
                imsave(input_img, impath)

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


# 函数用于取gif图片中各帧图片
def processImage(infile):
    try:
        im = Image.open(infile)
    except IOError:
        print("不能载入文件：", infile)
        sys.exit(1)
    img_num = 1
    mypalette = im.getpalette()

    try:
        while 1:
            im.putpalette(mypalette)
            new_im = Image.new("RGBA", im.size)
            new_im.paste(im)
            W = new_im.size[0]
            H = new_im.size[1]
            mkdir(frame_path)
            img_path = frame_path + '/' + str(img_num).zfill(3) + '.png'
            new_im.save(img_path)

            img_num += 1
            im.seek(im.tell() + 1)

    except EOFError:
        pass  # end of sequence
    img_num -= 1
    if SIZE_flag is False:
        return img_num, H, W
    else:
        return img_num


'''七，主函数'''
# ——————————————————————————————————————切割gif为图片——————————————————————————————————————
print("——————————————————————————————————————判断输入内容——————————————————————————————————————")
print("输入的内容为：", content_path)
if GIF_flag:
    print("输入的内容为gif类型，切割gif为图片")
    if SIZE_flag is False:
        img_num, height, width = processImage(content_path)
    else:
        img_num = processImage(content_path)
    print("切割的图片数为：", img_num)
else:
    print("输入的内容不为gif类型")
    Img = Image.open(content_path)
    if SIZE_flag is False:
        height, width = Img.size[1], Img.size[0]
    img_num = 1
print("图片宽度:", width, " | 图片高度:", height)
# ——————————————————————————————————————将切割的图片进行风格迁移——————————————————————————————————————
print("———————————————————————————————————将切割的图片进行风格迁移———————————————————————————————————")
loader = transforms.Compose([
    transforms.Resize([height, width]),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image

iter_num = img_num
print('——————————————————————————开始训练——————————————————————————')
for img_number in range(1, iter_num + 1):
    mkdir(img_save_path)
    print("[正在处理第" + str(img_number) + "张图片 | 剩余" + str(iter_num - img_number) + "张图片]")
    img_save_name = img_save_path + '/' + str(img_number).zfill(3) + '.png'
    if GIF_flag:
        content_name = frame_path + '/' + str(img_number).zfill(3) + '.png'
    else:
        content_name = content_path

    style_img = image_loader(style_path)
    content_img = image_loader(content_name)
    content_img = content_img[0, 0:3, :, :]
    content_img = content_img.view(1, 3, height, width)

    assert style_img.size() == content_img.size(), \
        "风格图与内容图格式大小不一致！"
    # 初始化输入图片
    input_img = content_img.clone() if pre_train == False else image_loader(input_path)

    total_start_T = time.time()
    # 设置内容风格损失层的位置
    [content_layers, style_layers] = [content_layers_setting, style_layers_setting] \
        if set_layer == True else [content_layers_default, style_layers_default]

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    imsave(output, img_save_name)

    total_end_T = time.time()
    print('训练总时长: {:2f}分钟'.format((total_end_T - total_start_T) / 60))
    print("————————————————————————————————————————————————————————————————————————————————————")

    if PIL_flag:
        plt.ion()  # 打开plt的交互模式

        plt.figure()
        imshow(style_img, title='Style Image')

        plt.figure()
        imshow(content_img, title='Content Image')

        plt.figure()
        imshow(output, title='Output Image')

        # sphinx_gallery_thumbnail_number = 4
        plt.ioff()  # 关闭plt的交互模式
        plt.show()
print('——————————————————————————训练完成——————————————————————————')
print()
# ——————————————————————————————————————将风格迁移后的图片合成为mp4和gif形式——————————————————————————————————————
if GIF_flag:
    print("——————————————————————————————————将风格迁移后的图片合成为mp4和gif形式——————————————————————————————————")
    images = []
    mkdir(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, (width, height))

    print("已完成图片有:")
    for i in range(1, iter_num + 1):
        im_name = os.path.join(img_save_path, str(i).zfill(3) + '.png')
        # im_name = os.path.join(im_dir, str(i) + '.jpg')
        frame = cv2.imread(im_name)
        videoWriter.write(frame)
        print(im_name)
        images.append(Image.open(im_name))

    videoWriter.release()
    images[0].save(gif_dir, fomat='GIF', append_images=images[1:],
                   save_all=True, duration=1e3 / 25, loop=0)

print('风格迁移完成！')
