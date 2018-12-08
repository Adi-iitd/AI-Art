# AI-Art

***

## Motivation

Creativity is something we closely associate with what it means to be human. But with digital technology now enabling machines to recognize, learn from and respond to humans and the world, an inevitable question follows: 

> Can machine be creative? And will artificial intelligence ever be able to make art?

Recent art experiments are the use of "generative adversarial networks" (GANs). GANs are "neural networks" that teach themselves through their own experimentation, rather than being programmed by humans. *It could be argued that the ability of machines to learn what things look like, and then make convincing new examples, marks the advent of "creative" AI.*

I will cover three different methods by which you can create novel arts, solely by code - **Neural Style Transfer, CycleGAN,** and **Pix2pix.**  

***

## Neural Style Transfer

Neural Style Transfer (NST) is one of the most fun techniques in deep learning. As seen below, it merges two images, namely, a "content" image (C) and a "style" image (S), to create a "generated" image (G). The generated image G combines the "content" of the image C with the "style" of image S. 

![neural-style](https://user-images.githubusercontent.com/41862477/49682529-b23e2880-fadb-11e8-8625-82fc2b14c487.png)

Neural Style Transfer (NST) uses a previously trained convolutional network, and builds on top of that. I will use VGG-19 which has already been trained on the very large ImageNet database. It learned to recognize a variety of *low level features* (at the earlier layers) and *high level features* (at the deeper layers). Building the NST algorithm takes three steps:

**Content Cost**:  ```Jcontent (C, G)```
**Style Cost**  :  ```Jstyle (S, G)``` 
**TV Cost**     :  ```Jtv (G) ```

*Putting all together*  :  J(G) = (alpha) * Jcontent (C, G) + (beta) * Jstyle (S, G) + (gamma)* Jtv (G).

> Let's delve deeper to know more profoundly what's going on under the hood of these algorithms.

###  Content Cost

The earlier layers of a ConvNet tend to detect lower-level features such as edges and simple textures, and the later layers tend to detect higher-level features such as more complex textures as well as object classes. Content loss tries to make sure that **generated** image G has similar content as the input image C and for that, we choose some layer's activations to represent the content of an image. Practically, we'll get the most visually pleasing results if we choose a layer in the middle of the network -- neither too shallow nor too deep is. 

Suppose we picked activations of **Conv_2** layer to represent the content cost. Now, set the image C as the input to the pretrained VGG network, and run forward propagation. Let  a(C) be the hidden layer activations which will be a **nH * nW * nC** tensor. Repeat the same process for the generated image. Let  a(G) be the corresponding hidden layer activations. Then the **Content Cost** function is defined as follows:

![3](https://user-images.githubusercontent.com/41862477/49682789-6772df80-fae0-11e8-8f7c-5805421e8121.JPG)






