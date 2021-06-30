------------------------------------------------------------------------------------------
*一号模型_基于图像迭代：
	两个文件夹是两个独立的实验，分别运行main.py即可
	可以在main.py开头修改内容图片，风格图片等

------------------------------------------------------------------------------------------
*二号模型_快速单风格迁移
	若要训练
	①将训练图片拷贝到./data/train中，然后运行data_process.py，生成content_image.npy
	②直接运行main.py，参数调节均在文件开头
	测试：
	若要直接调用模型测试，将图片放到./data/test中，运行test.py即可在result文件夹内看到风格迁移结果

注：一号模型不需要训练集，只需要两张图片做输入
我们的二号模型是在
https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/中的
vangogh2photo-trainB中以2014开头的883张图片下训练出来的。


------------------------------------------------------------------------------------------
*参考文献
方法一官方文档：https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#sphx-glr-advanced-neural-style-tutorial-py
方法二博客：https://blog.csdn.net/weixin_48866452/article/details/109309245
知乎方法总结：https://zhuanlan.zhihu.com/p/144296971