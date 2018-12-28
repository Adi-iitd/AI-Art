# AI-Art

***

## Motivation

Creativity is something we closely associate with what it means to be human. But with digital technology now enabling machines to recognize, learn from and respond to humans and the world, an inevitable question follows: 

> Can machine be creative? And will artificial intelligence ever be able to make art?

Recent art experiments are the use of "generative adversarial networks" (GANs). GANs are "neural networks" that teach themselves through their own experimentation, rather than being programmed by humans. *It could be argued that the ability of machines to learn what things look like, and then make convincing new examples, marks the advent of "creative" AI.*

I will cover three different methods by which you can create novel arts, solely by code - **Neural Style Transfer, CycleGAN,** and **Pix2pix.**  

***
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

> *These are brush-strokes that the model learned when layers **Conv_2_2, Conv_3_1, Conv_3_2, Conv_3_3, Conv_4_1, Conv_4_3, Conv_4_4, Conv_5_1, and Conv_5_4** (left to right and top to bottom) were used one at a time in the Style cost.*

***You might be wondering why am I showing these images, what one can conclude after looking at these brush-strokes?***

So, the reason behind running this experiment was that - authors of the original paper gave equal weight to the styles learned by different layers while calculating the **Total Style Cost** (weighted summation of style loss corresponding to different layers). Now, that's not intuitive at all after looking at these images, because we can see that styles learned by the shallower layers are more aesthetically pleasing, compared to what deeper layers learned. So, we would like to assign a lower weight to the deeper layers and higher to the shallower ones; Exponentially decreasing the weights as we go deeper and deeper could be one way.

> *Similarly, you can run the experiment to minimize only the content cost, and see which layer performs the best (You should always keep in mind that, you only want to transfer the content of the image not exactly copy paste it in the final generated image). I generally find Conv_3_2 to be the best (earlier layers are very good at reconstructing the ditto original image).*

***

### Results

![3](https://user-images.githubusercontent.com/41862477/49685490-3b6c5400-fb0a-11e8-876a-526a95591cb5.jpg)
![6](https://user-images.githubusercontent.com/41862477/49685493-3c04ea80-fb0a-11e8-8a2a-822130da61d6.png)
![res_1](https://user-images.githubusercontent.com/41862477/49685494-3c04ea80-fb0a-11e8-8a9c-42e7173fdb1b.png)
![1](https://user-images.githubusercontent.com/41862477/49685487-3ad3bd80-fb0a-11e8-833b-e34dfd340957.jpg)
![6](https://user-images.githubusercontent.com/41862477/49685493-3c04ea80-fb0a-11e8-8a2a-822130da61d6.png)
![res_2](https://user-images.githubusercontent.com/41862477/49685495-3c9d8100-fb0a-11e8-937f-b62c62a6016a.png)
![3](https://user-images.githubusercontent.com/41862477/49685490-3b6c5400-fb0a-11e8-876a-526a95591cb5.jpg)
![4](https://user-images.githubusercontent.com/41862477/49685491-3b6c5400-fb0a-11e8-9161-1c6940d5e6bc.jpg)
![res_3](https://user-images.githubusercontent.com/41862477/49685496-3c9d8100-fb0a-11e8-9240-39be822aee63.png)
![3](https://user-images.githubusercontent.com/41862477/49685490-3b6c5400-fb0a-11e8-876a-526a95591cb5.jpg)
![5](https://user-images.githubusercontent.com/41862477/49685492-3c04ea80-fb0a-11e8-8308-d770f4d0185d.jpg)
![res_4](https://user-images.githubusercontent.com/41862477/49685497-3d361780-fb0a-11e8-8f13-57d2965ccbd0.png)
![1](https://user-images.githubusercontent.com/41862477/49685487-3ad3bd80-fb0a-11e8-833b-e34dfd340957.jpg)
![2](https://user-images.githubusercontent.com/41862477/49685488-3ad3bd80-fb0a-11e8-8c9c-4fecff45f42b.jpg)
![res_5](https://user-images.githubusercontent.com/41862477/49685498-3d361780-fb0a-11e8-8728-a92867787e9a.png)
![3](https://user-images.githubusercontent.com/41862477/49686281-44642200-fb18-11e8-8e75-6cb5ab4d32c8.JPG)
![2](https://user-images.githubusercontent.com/41862477/49686280-44642200-fb18-11e8-914d-3127fd6dcd2e.JPG)
![50_3_2](https://user-images.githubusercontent.com/41862477/49686282-44642200-fb18-11e8-80b3-6b2216370595.png)

***

***

## Pix2pix

![1](https://user-images.githubusercontent.com/41862477/49689620-be60cf00-fb49-11e8-97b4-6cf53801ad3d.JPG)

The authors investigated Conditional adversarial networks as a general-purpose solution to **Image-to-Image Translation** problems in this [paper](https://arxiv.org/pdf/1611.07004.pdf). These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. In analogy to automatic language translation, we define automatic image-to-image translation as the task of translating one possible representation of a scene into another, given sufficient training data.

In Generative Adversarial Networks settings, we could specify only a high-level goal, like “make the output indistinguishable from reality”, and then it automatically learn a loss function appropriate for satisfying this goal. Like other GANs, Conditional GANs also have one discriminator (or critic depending on the loss function we are using) and one generator, and it tries to learn a conditional generative model which makes it suitable for Image-to-Image translation tasks, where we condition on an input image and generate a corresponding output image. 

> If mathematically expressed, CGANs learn a mapping from observed image X and random noise vector z, to y, *G : {x, z} → y*. The generator G is trained to produce outputs that cannot be distinguished from **real** images by an adversarially trained discriminator, D, which in turn is itself optimized to do as well as possible at identifying the generator’s **fakes**.

![2](https://user-images.githubusercontent.com/41862477/49689774-1698d080-fb4c-11e8-92af-dc3d48e66ec2.JPG)

> *The figure shown above illustrates the working of GAN in Conditional setting.*

### Loss Function

The objective of a conditional GAN can be expressed as:

> **```Lc GAN (G, D) = Ex,y (log D(x, y)) + Ex,z (log(1 − D(x, G(x, z)))```** where G tries to minimize this objective against an adversarial D that tries to maximize it, i.e. **```G∗ = arg min(G)max(D) Lc GAN (G, D)```**. It is beneficial to mix the GAN objective with a more traditional loss, such as L1 distance to make sure that, the ground truth and the output are close to each other in L1 sense **```L(G) = Ex,y,z ( ||y − G(x, z)|| )```**.

Without z, the net could still learn a mapping from x to y, but would produce deterministic outputs, and therefore fail to match any distribution other than a **delta function**. Instead, the authors of Pix2pix provided noise only in the form of **dropout**, applied on several layers of the generator at **both training and test time**.

The Min-Max objective mentioned above was used in the original paper, when GAN was first proposed by **Ian Goodfellow** in 2014, but unfortunately, it doesn't perform well due to vanishing gradients problems. Since then, there has been a lot of development, and many researchers have proposed different kinds of loss formulation (LS-GAN, WGAN, WGAN-GP) to overcome these issues. Authors of this paper used **Least-square** objective function while running their optimization process. 

### Network Architecture

#### Generator: 

In Image-to-image translation problems, we map a high resolution input grid to a high resolution output grid. Both are renderings of the same underlying structure with the only difference in the surface appearance. The authors designed the generator architecture around these considerations. They used an encoder-decoder network in which the input is passed through a series of layers that progressively downsample, until a bottleneck layer, at which point the process is reversed. 

> To preserve the low-level details, skip connections are used. Specifically, skip connections are added between each layer i and layer n − i, where n is the total number of layers. Each skip connection simply concatenates all channels at layer i with those at layer n − i.

```Architecture: 
Encoder:  E(64, 1) - E(64, 1) - E(64, 2) - E(128, 2) - E(256, 2) - E(512, 2) - E(512, 2) - E(512, 2) - E(512, 2)
Decoder:  D(512, 2) - D(512, 2) - D(512, 2) - D(256, 2) - D(128, 2) - D(64, 2) - D(64, 2) - D(64, 1) - D(3, 1)
```

#### Discriminator:

The GAN discriminator models high-frequency structure term, relying on an L1 term to force low-frequency correctness. In order to model high-frequencies, it is sufficient to restrict the attention to the structure in local image patches. Therefore, discriminator architecture was termed PatchGAN – that only penalizes structure at the scale of patches. This discriminator tries to classify if each N × N patch in an image is real or fake. We run this discriminator convolutionally across the image, and average all responses to provide the ultimate output of D. Patch GANs discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter. The recpetive field of the discriminator used was 70 * 70 (and was performing best compared to smaller and larger receptive fields).

```The 70 × 70 discriminator architecture is: C64 - C128 - C256 - C512```

#### Optimization

- Alternate between one gradient descent step on D, and one step on G. 
- The objective function was divided by 2 while optimizing D, which slows down the rate at which D learns relative to G. 
- Use **Adam solver**, with a learning rate of 2e-4, and momentum parameters β1 = 0.5, β2 = 0.999.
- Use **Dropout** both at the training and test time.
- Use **instance normalization** (normalization using the statistics of the test batch) instead of batch normalization.
- Can work even with the much smaller datasets.
- Both L1 and cGAN loss are important to reduce the artifacts in the final output.

***
***

## CycleGAN

![1](https://user-images.githubusercontent.com/41862477/50483300-416e9a00-0a11-11e9-8b77-e91d30a409bb.jpg)

Image-to-Image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. So, the authors in [this](https://arxiv.org/pdf/1703.10593.pdf) paper presented an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. 

*The goal is to learn a mapping **G : X → Y** such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, they coupled it with an inverse mapping **F : Y → X** and introduced a cycle consistency loss to enforce **F(G(X)) ≈ X** (and vice-versa).*

### Motivation:

Obtaining paired training data can be difficult and expensive. For example, only a couple of datasets exist for tasks like semantic segmentation, and they are relatively small. Obtaining input-output pairs for graphics tasks like **artistic stylization** can be even more difficult since the desired output is highly complex, typically requiring artistic authoring. For many tasks, like **object transfiguration** (e.g., zebra <-> horse), the desired output is not even well-defined. Therefore, the authors tried to present an algorithm that can learn to translate between domains without paired input-output examples. The primary assumption is that there exists some underlying relationship between the domains. Although there is a lack of supervision in the form of paired examples, supervision at the level of sets can still be exploited: *one set of images in domain X and a different set in domain Y.*

The optimal G thereby translates the domain X to a domain Y* distributed identically to Y. However, such a translation does not guarantee that an individual input x and output y are paired up in a meaningful way – there are infinitely many mappings G that will induce the same distribution over y*. Key points:

- Difficult to optimize adversarial objective in isolation - standard procedures often lead to the well-known problem of mode collapse
- Exploited the property that translation should be **Cycle consistent**. Mathematically, translator G : X → Y and another translator F : Y → X, should be inverses of each other (and both mappings should be bijections). 
- Enforcing the structural assumption by training both the mapping G and F simultaneously, and adding a cycle consistency loss that encourages ***F(G(x)) ≈ x and G(F(y)) ≈ y***.

![2](https://user-images.githubusercontent.com/41862477/50504160-b1bd0000-0a91-11e9-9909-29b2121449b8.jpg)

> *As illustrated in figure, their model includes two mappings **G : X → Y and F : Y → X**. In addition, they introduced two adversarial discriminators DX and DY , where DX aims to distinguish between images {x} and translated images {F(y)}; in the same way, DY aims to discriminate between {y} and {G(x)}. So, final objective contains two types of terms: adversarial losses for matching the distribution of generated images to the data distribution in the target domain; and cycle consistency losses to prevent the learned mappings G and F from contradicting each other.*

#### Adversarial Loss:

Adversarial loss is applied to both mapping functions -  G : X → Y and its discriminator DY and  F : Y → X and its discriminator DX, where G tries to generate images G(x) that look similar to images from domain Y , while DY aims to distinguish between translated samples G(x) and real samples y (similar condition holds for the other one).

- Generator (G) tries to minimize ``` E[x∼pdata(x)] (D(G(x)) − 1)** 2```
- Discriminator (DY) tries to minimize ``` E[y∼pdata(y)] (D(y) − 1)**2 + E[x∼pdata(x)] D(G(x))**2```
- Generator (F) tries to minimize ``` E[y∼pdata(y)] (D(G(y)) − 1)** 2```
- Discriminator (DX) tries to minimize ``` E[x∼pdata(x)] (D(x) − 1)**2 + E[y∼pdata(y)] D(G(y))**2```

#### Cycle Consistency Loss:

Adversarial training can, in theory, learn mappings G and F that produce outputs identically distributed as target domains Y and X respectively (strictly speaking, this requires G and F to be stochastic functions). However, with large enough capacity, a network can map the same set of input images to any random permutation of images in the target domain, where any of the learned mappings can
induce an output distribution that matches the target distribution. Thus, adversarial losses alone cannot guarantee that the learned function can map an individual input xi to a desired output yi. To further reduce the space of possible mapping functions, learned functions should be cycle-consistent.

``` 
Lcyc (G, F) = E[x∼pdata(x)] || F(G(x)) − x|| + E[y∼pdata(y)] || G(F(y)) − y || 
``` 

#### Full Objective:

The full objective is: 
``` 
L (G, F, DX, DY) = LGAN (G, DY , X, Y) + LGAN (F, DX, Y, X) + λLcyc(G, F)
```
, where lambda controls the relative importance of the two objectives.

#### Insights:

- This model can be viewed as training two **autoencoders**: first **F◦G : X → X** jointly with second **G◦F : Y → Y**. 
- These have special internal structures - map an image to itself via an intermediate representation that is a translation of the image into another domain. 
- Can also be seen as a special case of **adversarial autoencoders**, which use an adversarial loss to train the bottleneck layer of an autoencoder to match an arbitrary target distribution. 
- The target distribution for the X → X autoencoder is the domain Y and for the Y → Y autoencoder is the domain X.

## Implementation:

#### Training Details:

- Two **stride-2** convolutions, several **residual** blocks, and two **fractionally strided** convolutions with stride 1/2.
- 6 blocks for 128 × 128 images and 9 blocks for 256 × 256 and higher resolution training images.
- **Instance** normalization instead of batch normalization.
- **Patch Discriminator** - 70 × 70 PatchGANs, which aim to classify whether 70 × 70 overlapping image patches are real or fake (more parameter efficient compared to full-image discriminator)
- To reduce model oscillation, update the discriminators using a history of generated images rather than the latest ones - always keep an image buffer of 50 previously generated images.
- Set λ to 10 in total loss equation, use the Adam solver with a batch size of 1 
- Learning rate of 0.0002 for the first 100 epochs and then linearly decay the rate to zero over the next 100 epochs.

#### Architecture Details:
```
Generator:
- Network with 6 residual blocks: c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3
- Network with 9 residual blocks: c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3

Discriminator:
- C64-C128-C256-C512
```
> *c7s1-k denote a 7×7 Convolution-InstanceNormReLU Layer with k filters and stride 1. dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2. Reflection padding was used to reduce artifacts. Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer. uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride 1/2. Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. After the last layer, a convolution is applied to produce a 1-dimensional output. **Do not** use InstanceNorm for the first C64 layer. Use leaky ReLUs with a slope of 0.2*

#### Application - Photo generation from paintings: 

For painting → photo, they found that it was helpful to introduce an additional loss to encourage the mapping to preserve color composition between the input and output. In particular, they regularized the generator to be near an identity mapping when real samples
of the target domain are provided as the input to the generator i.e., 
```
Lidentity (G, F) = E[y∼pdata(y)] || G(y) − y || + E[x∼pdata(x)] || F(x) − x ||
```

### Results:

#### Photo -> Cezzane Painitings:

![2](https://user-images.githubusercontent.com/41862477/50507791-fd2cd980-0aa4-11e9-9456-6bbf25f54a49.jpg)
![3](https://user-images.githubusercontent.com/41862477/50507793-fdc57000-0aa4-11e9-8981-365b56c92b73.jpg)
![5](https://user-images.githubusercontent.com/41862477/50507794-fdc57000-0aa4-11e9-8b06-2e260f23ac8e.jpg)
![8](https://user-images.githubusercontent.com/41862477/50507795-fdc57000-0aa4-11e9-8da8-d92cb4775264.jpg)

#### Cezzane Painitings -> Photo:

![1](https://user-images.githubusercontent.com/41862477/50507839-2a798780-0aa5-11e9-99e6-7f7770ae8bdb.jpg)
![4](https://user-images.githubusercontent.com/41862477/50507840-2a798780-0aa5-11e9-9022-96e648314a86.jpg)
![6](https://user-images.githubusercontent.com/41862477/50507841-2b121e00-0aa5-11e9-9172-64c3750df36e.jpg)
![7](https://user-images.githubusercontent.com/41862477/50507842-2b121e00-0aa5-11e9-85f8-1eb7276cfd8f.jpg)





***

***Thanks for going through this post! Any feedbacks are duly appreciated.***
