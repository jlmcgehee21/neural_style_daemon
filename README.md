This is a fork of the repository [Neural Artistic Style in Python](https://github.com/andersbll/neural_artistic_style), which is an implementation of [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576). A method to transfer the style of one image to the subject of another image.

The goal of this project is to create a Linux Upstart daemon which monitors given directory for sets of images to batch process.

**Currently a giant WIP**

### Requirements
 - [DeepPy](http://github.com/andersbll/deeppy), Deep learning in Python.
 - [CUDArray](http://github.com/andersbll/cudarray) with [cuDNN](https://developer.nvidia.com/cudnn), CUDA-accelerated NumPy.
 - [Pretrained VGG 19 model](http://www.vlfeat.org/matconvnet/pretrained), choose *imagenet-vgg-verydeep-19*.
