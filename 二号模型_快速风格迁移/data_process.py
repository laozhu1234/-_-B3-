# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 18:51:54 2021

@author: Lenovo_ztk
"""


import torch, numpy as np
import glob
import PIL.Image as Image
import torchvision.transforms as transforms

picpath = './data/train/*.jpg'	#读取文件的路径
savepath = 'content_image.npy'
imsize = 128

loader = transforms.Compose([
    transforms.Resize(imsize)
    ])

def data_process(picpath,savepath):
    picdir = glob.glob(picpath)
    
    num = len(picdir)
    
    pic = np.zeros((num,imsize,imsize,3),dtype=np.uint8)
    for i in range(num):
        t_pic = Image.open(picdir[i])
        t_pic = loader(t_pic)
        pic[i] = np.asarray(t_pic)
        
    np.save(savepath, pic)
    
    
if __name__ == '__main__':    
    data_process(picpath, savepath)
    
    '''测试写入正确'''
    # pic = np.load(savepath)
    # print(pic.shape)
    # a = Image.fromarray(pic[0])
    # a.show()
    
