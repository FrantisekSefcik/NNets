# Proposal 

## Motivation

In this project we will solve problem of text to image generation by Deep Convolutional Generative Adversarial Network 
(GAN). Today GANs are very interesting field of research, because they belong to the set of generative models. 
It means that they are able to generate new content, what is kind of magic. We know using of GANs from many cases but 
creating new images from text description is interesting ideas in computer science today. 

## Related work

[Generative Adversarial Text to Image Synthesis](https://arxiv.org/pdf/1605.05396.pdf)

[StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks](
http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_StackGAN_Text_to_ICCV_2017_paper.pdf)

 [Multi-Scale Gradient GAN for Stable Image Synthesis](https://arxiv.org/pdf/1903.06048.pdf)

 [Keras-text-to-image](https://github.com/chen0040/keras-text-to-image#keras-text-to-image)
 
 [Conditional GAN](https://golden.com/wiki/Conditional_generative_adversarial_network_(cGAN))
 
 
## Datasets

We identified several datasets with available data for our experiments. Datasets usually contains images with objects 
and theirs labels or text description. We will describe then more precisely below. 

#### COCO dataset [[13](http://cocodataset.org/#home)]

#### Open images [[12](https://storage.googleapis.com/openimages/web/index.html)]

#### Flowers [[14](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)]








[12] https://storage.googleapis.com/openimages/web/index.html

[13] http://cocodataset.org/#home

[14] (http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)


## High-Level Solution Proposal 

As we mentioned above, we will try to generate images from text. As architecture of the model we proposed Conditional 
generative adversarial network (cGAN) is an extension GAN. You can see the architecture on the picture bellow.

![Drag Racing](cGAN.png)  

The architecture is comprised of generator and discriminator model. The generator model is responsible for 
generating new examples that ideally are indistinguishable from real examples in the dataset. 
The discriminator model is responsible for classifying a given image as either real
or fake (generated). Input to generator is vector that will be created from
random noise and Word2Vec encoding of image label or image description. This implementation will be build on top of 
[DCGAN in Tensorflow](https://www.tensorflow.org/tutorials/generative/dcgan). 

We plan build our model in experiments:
1. We will train prototype of model with Mnist dataset. This will be entry point that our model works properly.
2. In the next step we will prepare our data from either COCO or Openimage dataset. We will select images from set of 
several classes to avoid big complexity. Then we will tune and train our model to generate most reasonable images.
3. As a third experiment we would like to use data challenge from Openimage dataset. This challenge offers images with 
relations of two objects in image (man at horse, etc ...).  In case it wouldn't be out of scope we will implement 
generation of images with two objects.    