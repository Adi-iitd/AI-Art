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

Neural Style Transfer (NST) uses a previously trained convolutional network and builds on top of that. I will be using VGG-19 which has already been trained on the very large ImageNet database. It learned to recognize a variety of * low-level features* (at the earlier layers) and * high-level features* (at the deeper layers). Building the NST algorithm takes three steps:

- **Content Cost** : **J***content* (C, G)
- **Style Cost** : **J***style* (S, G) 
- **Total Variation (TV) Cost** : **J***tv* (G)

*Putting all together*  :  **J***tot* (G) = (alpha) * **J***content* (C, G) + (beta) * **J***style* (S, G) + (gamma)* **J***tv* (G)

> Let's delve deeper to know more profoundly what's going on under the hood of these algorithms.

###  Content Cost

The earlier layers of a ConvNet tend to detect lower-level features such as edges and simple textures, and the later layers tend to detect higher-level features such as more complex textures as well as object classes. Content loss tries to make sure that "generated" image G has similar content as the input image C. For that, we need to choose some layer's activation to represent the content of an image. *Practically, we'll get the most visually pleasing results if we choose a layer in the middle of the network - neither too shallow nor too deep.* Suppose we picked activations of **Conv_3_2** layer to represent the content cost. Now, set the image C as the input to the pre-trained VGG network, and run forward propagation. 

Let  a(C) be the hidden layer activations which will be a **nH * nW * nC** tensor. Repeat the same process for the generated image and let  a(G) be the corresponding hidden layer activations. Finally, the **Content Cost** function is defined as follows:

![3](https://user-images.githubusercontent.com/41862477/49682789-6772df80-fae0-11e8-8f7c-5805421e8121.JPG)

nH, nW, and nC are the height, width, and the number of channels of the hidden layer chosen. In order to compute the cost **J***content* (C, G), it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below.

![1](https://user-images.githubusercontent.com/41862477/49682841-10b9d580-fae1-11e8-851f-ec9fbf37dd92.JPG)

### Style Cost

First, we need to know something about the **Gram Matrix**. In linear algebra, the Gram matrix G of a set of vectors  (v1, …, vn) is the matrix of dot products, whose entries are  *G (i, j) = np.dot(vi, vj)*. In other words,  *G (i, j)*  compares how similar vi is to vj. If they are highly similar, the outcome would be a large value, otherwise, it would be low suggesting lower correlation. In NST, we can compute the Gram matrix by multiplying the **unrolled** filter matrix with their transpose as shown below:

![2](https://user-images.githubusercontent.com/41862477/49682895-f8968600-fae1-11e8-8fbd-b754c625542a.JPG)

The result is a matrix of dimension  *(nC, nC)* where nC is the number of filters. The value *G (i, j)* measures how similar the activations of filter i are to the activations of filter j. One important part of the gram matrix is that the diagonal elements such as  *G (i, i)* also measures how active filter i is. For example, suppose filter i is detecting vertical textures in the image, then *G (i, i)*  measures how common vertical textures are in the image as a whole. 

>*By capturing the prevalence of different types of features G (i, i), as well as how much different features occur together  G (i, j), the Gram matrix G measures the **style** of an image.*

After we have the Gram matrix, we want to minimize the distance between the Gram matrix of the "style" image S and that of the "generated" image G. Usually, we take more than one layers in the account to calculate **Style cost** as opposed to Content cost (in which only one layer is sufficient), and the reason for doing so is discussed later on in the post. For a single hidden layer, the corresponding style cost is defined as:

![4](https://user-images.githubusercontent.com/41862477/49683030-54620e80-fae4-11e8-9f79-a500da7f12c3.JPG)

### Total Variation (TV) Cost

It acts like a regularizer which encourages spatial smoothness in the generated image (G). This was not used in the original paper proposed by [Gatys et al.](https://arxiv.org/pdf/1508.06576.pdf) but it can sometimes improve the results. For 2D signal (or image), it is defined as follows: 

![5](https://user-images.githubusercontent.com/41862477/49683156-1b2a9e00-fae6-11e8-8321-34b3c1173175.JPG)

### Experiments

> What will happen if we zero out the coefficients of the Content and TV loss, assuming we are taking only one layer's activation to compute Style cost?

As many of you might have guessed, the optimization algorithm will now only have to minimize the Style cost. So, for a given **Style image**, we would see what kind of brush-strokes will the model try to enforce in the final generated image (G). Remember, we started with only one layer's activation in the Style cost, so running the experiments for different layers would give different kind of brush-strokes that would be there in the final generated image. Suppose the style image is famous **The great wall of Kanagawa** shown below:

![6](https://user-images.githubusercontent.com/41862477/49683530-af97ff00-faec-11e8-9d30-e3bc15e9fa88.jpg)

Here are the brush-strokes that we get after running the experiment taking into account the different layers, one at a time.

![2_2](https://user-images.githubusercontent.com/41862477/49683610-e15d9580-faed-11e8-8d3f-58de7ee88595.png)
![3_1](https://user-images.githubusercontent.com/41862477/49683611-e15d9580-faed-11e8-80d6-3d216487f678.png)
![3_2](https://user-images.githubusercontent.com/41862477/49683613-e15d9580-faed-11e8-836f-b8d3dab32f03.png)
![3_3](https://user-images.githubusercontent.com/41862477/49683614-e1f62c00-faed-11e8-964f-6e0e4085cc3d.png)
![4_1](https://user-images.githubusercontent.com/41862477/49683615-e1f62c00-faed-11e8-9583-a6ca7cfc058b.png)
![4_3](https://user-images.githubusercontent.com/41862477/49683616-e1f62c00-faed-11e8-9cf2-cbc5c3f5e18b.png)
![4_4](https://user-images.githubusercontent.com/41862477/49683617-e1f62c00-faed-11e8-9e09-4147889c3b01.png)
![5_1](https://user-images.githubusercontent.com/41862477/49683618-e28ec280-faed-11e8-92b3-f48787c98f8a.png)
![5_4](https://user-images.githubusercontent.com/41862477/49683619-e28ec280-faed-11e8-8076-85145ff382ea.png)

> *These are brush-strokes that the model learned when layers **Conv_2_2, Conv_3_1, and Conv_3_2, Conv_4_1, Conv_4_3, Conv_4_4, Conv_5_1, and Conv_5_4** (left to right and top to bottom) were used one at a time in the Style cost.*

***You might be wondering why am I showing these images, what one can conclude after looking at these brush-strokes?***

So, the reason behind running this experiment was that - authors of the original paper gave equal weight to the styles learned by different layers while calculating the **Total Style Cost** (weighted summation of style loss corresponding to different layers). Now, that's not intuitive at all after looking at these images, because we can see that styles learned by the shallower layers are more aesthetically pleasing, compared to what deeper layers learned. So, we would like to assign a lower weight to the deeper layers and higher to the shallower ones; Exponentially decreasing the weights as we go deeper and deeper could be one way.

> *Similarly, you can run the experiment to minimize only the content cost, and see which layer performs the best (You should always keep in mind that, you only want to transfer the content of the image not exactly copy paste it in the final generated image). I generally find Conv_3_2 to be the best (earlier layers are very good at reconstructing the ditto original image).*

***

### Results


