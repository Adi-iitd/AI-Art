# AI Art

***

## Motivation

<p align = "justify"> Creativity is something we closely associate with what it means to be human. But with digital technology now enabling machines to recognize, learn from, and respond to humans and the world, an inevitable question follows: </p>

> <i> Can a machine be creative, and will artificial intelligence ever be able to make art? </i>

<p align = "justify"> Recent art experiments use Deep Learning to teach themselves through their own experimentations, rather than being programmed by humans. <i> It could be argued that the ability of machines to learn what things look like, and then make convincing new examples marks the advent of creative AI. </i> </p>

<p align = "justify"> I will cover four different Deep Learning models in this tutorial to create our own novel arts, solely by code - <b> Style Transfer, Pix2Pix, CycleGAN, and Deep Dream. </b> </p>

***

## Style Transfer

<p align = "justify"> Style Transfer is one of the most fun techniques in Deep learning. It combines the two images, namely, a <b> Content </b> image (C) and a <b> Style </b> image (S), to create an <b> Output </b> image (G). The Output image G combines the Content of image C with the Style of image S. </p>

![neural-style](https://user-images.githubusercontent.com/41862477/49682529-b23e2880-fadb-11e8-8625-82fc2b14c487.png)

<p align = "justify"> Style Transfer uses a pre-trained Convolutional Neural Network <b> VGG-19, </b> (because of it's simple and hierarchical design) which already can recognize a variety of <i> low-level features </i> (at the earlier layers) and <i> high-level features </i> (at the deeper layers). Style Transfer incorporates <i> three </i> different kinds of losses: </p>

- **Content Cost** : **J**<sub>Content</sub> (C, G)
- **Style Cost** : **J**<sub>Style</sub> (S, G)
- **Total Variation (TV) Cost** : **J**<sub>TV</sub> (G)

*Putting all together*: **J**<sub>Total</sub> (G) = α * **J**<sub>Content</sub> (C, G) + β * **J**<sub>Style</sub> (S, G) + γ * **J**<sub>TV</sub> (G)

> Let's delve deeper to know more profoundly what's going on under the hood!

###  Content Cost

<p align = "justify"> Generally each layer in the network defines a non-linear filter bank whose complexity increases with the position of the layer in the network. First few layers of the ConvNet tend to detect low-level features such as edges and simple textures, and the last few layers tend to detect high-level features such as more complex textures as well as object classes. <b>Content loss</b> tries to make sure that the Output image <b>G</b> has similar content as the Input image <b>C</b>, for which, we need to minimize the (<b>MSE</b>) loss between the feature maps of the respective images. <i> Practically, we get the most visually pleasing results if we choose a layer in the middle of the network - neither too shallow nor too deep. The higher layers in the network capture the high-level content in terms of objects and their arrangement in the input image but do not constrain the exact pixel values of the reconstruction very much. In contrast, reconstructions from the lower layers simply reproduce the exact pixel values of the original image (Fig 1, content reconstructions a–e). 

![Con_recons_1_2](https://user-images.githubusercontent.com/41862477/82235677-a8ffef00-9950-11ea-8e38-513055c487cf.jpg)
![Con_recons_2_2](https://user-images.githubusercontent.com/41862477/82235682-aac9b280-9950-11ea-8885-4b8775638bbe.jpg)
![Con_recons_3_2](https://user-images.githubusercontent.com/41862477/82235683-abfadf80-9950-11ea-95b8-d9b8836ffa58.jpg)
![Con_recons_4_2](https://user-images.githubusercontent.com/41862477/82235686-ac937600-9950-11ea-9fe3-14dd979106cc.jpg)
![Con_recons_5_2](https://user-images.githubusercontent.com/41862477/82235688-ad2c0c80-9950-11ea-8a3b-d592d2bfee82.jpg)  
 
> <p align = "justify"> <i> These are reconstructions that the model generated when layers <b> Conv_1_2, Conv_2_2, Conv_3_2, Conv_4_2, and Conv_5_2 </b> (left to right and top to bottom) were used one at a time in the Content cost. </i> </p> 
  
<p align = "justify"> Let  a(C) be the hidden layer activations which will be a <b> nH * nW * nC </b> tensor. Repeat the same process for the generated image and let  a(G) be the corresponding hidden layer activations. Finally, the <b> Content Cost </b> function is defined as follows: </p>

![3](https://user-images.githubusercontent.com/41862477/49682789-6772df80-fae0-11e8-8f7c-5805421e8121.JPG)

<p align = "justify"> nH, nW, and nC are the height, width, and the number of channels of the hidden layer chosen. In order to compute the cost **J***content* (C, G), it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below. </p>

![1](https://user-images.githubusercontent.com/41862477/49682841-10b9d580-fae1-11e8-851f-ec9fbf37dd92.JPG)

### Style Cost

<p align = "justify"> First, we need to know something about the <b> Gram Matrix </b>. In linear algebra, the Gram matrix G of a set of vectors  (v1, …, vn) is the matrix of dot products, whose entries are <i> G (i, j) = np.dot(vi, vj) </i>. In other words, <i> G (i, j) </i> compares how similar vi is to vj. If they are highly similar, the outcome would be a large value, otherwise, it would be low suggesting lower correlation. In NST, we can compute the Gram matrix by multiplying the <b> unrolled </b> filter matrix with their transpose as shown below: </p>

![2](https://user-images.githubusercontent.com/41862477/49682895-f8968600-fae1-11e8-8fbd-b754c625542a.JPG)

<p align = "justify"> The result is a matrix of dimension <i> (nC, nC) </i> where nC is the number of filters. The value <i> G (i, j) </i> measures how similar the activations of filter i are to the activations of filter j. One important part of the gram matrix is that the diagonal elements such as  G (i, i) also measures how active filter i is. For example, suppose filter i is detecting vertical textures in the image, then G (i, i)  measures how common vertical textures are in the image as a whole. 

><p align = "justify"> <i> By capturing the prevalence of different types of features G (i, i), as well as how much different features occur together   G (i, j), the Gram matrix G measures the <b> style </b> of an image. </i> </p>

<p align = "justify"> After we have the Gram matrix, we want to minimize the distance between the Gram matrix of the "style" image S and that of the "generated" image G. Usually, we take more than one layers in the account to calculate <b> Style cost </b> as opposed to Content cost (in which only one layer is sufficient), and the reason for doing so is discussed later on in the post. For a single hidden layer, the corresponding style cost is defined as: </p>

![4](https://user-images.githubusercontent.com/41862477/49683030-54620e80-fae4-11e8-9f79-a500da7f12c3.JPG)

### Total Variation (TV) Cost

<p align = "justify"> It acts like a regularizer which encourages spatial smoothness in the generated image (G). This was not used in the original paper proposed by [Gatys et al.](https://arxiv.org/pdf/1508.06576.pdf) but it can sometimes improve the results. For 2D signal (or image), it is defined as follows: </p> 

![5](https://user-images.githubusercontent.com/41862477/49683156-1b2a9e00-fae6-11e8-8321-34b3c1173175.JPG)

### Experiments

> What will happen if we zero out the coefficients of the Content and TV loss, assuming we are taking only one layer's activation to compute Style cost?

<p align = "justify"> As many of you might have guessed, the optimization algorithm will now only have to minimize the Style cost. So, for a given <b> Style image </b>, we would see what kind of brush-strokes will the model try to enforce in the final generated image (G). Remember, we started with only one layer's activation in the Style cost, so running the experiments for different layers would give different kind of brush-strokes that would be there in the final generated image. Suppose the style image is famous <b> The great wall of Kanagawa </b> shown below: </p>

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

> <p align = "justify"> <i> These are brush-strokes that the model learned when layers <b> Conv_2_2, Conv_3_1, Conv_3_2, Conv_3_3, Conv_4_1, Conv_4_3, Conv_4_4, Conv_5_1, and Conv_5_4 </b> (left to right and top to bottom) were used one at a time in the Style cost. </i> </p>

***You might be wondering why am I showing these images, what one can conclude after looking at these brush-strokes?***

<p align = "justify"> So, the reason behind running this experiment was that - authors of the original paper gave equal weight to the styles learned by different layers while calculating the <b> Total Style Cost </b> (weighted summation of style loss corresponding to different layers). Now, that's not intuitive at all after looking at these images, because we can see that styles learned by the shallower layers are more aesthetically pleasing, compared to what deeper layers learned. So, we would like to assign a lower weight to the deeper layers and higher to the shallower ones; Exponentially decreasing the weights as we go deeper and deeper could be one way. </p>

> <p align = "justify"> <i> Similarly, you can run the experiment to minimize only the content cost, and see which layer performs the best (You should always keep in mind that, you only want to transfer the content of the image not exactly copy paste it in the final generated image). I generally find Conv_3_2 to be the best (earlier layers are very good at reconstructing the ditto original image). </i> </p>

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

***
***

## Pix2pix

![1](https://user-images.githubusercontent.com/41862477/49689620-be60cf00-fb49-11e8-97b4-6cf53801ad3d.JPG)

<p align = "justify"> The authors investigated Conditional adversarial networks as a general-purpose solution to <b> Image-to-Image Translation </b> problems in this [paper](https://arxiv.org/pdf/1611.07004.pdf). These networks not only learn the mapping from input image to output image, but also learn a loss function to train this mapping. In analogy to automatic language translation, we define automatic image-to-image translation as the task of translating one possible representation of a scene into another, given sufficient training data. </p>

<p align = "justify"> In Generative Adversarial Networks settings, we could specify only a high-level goal, like “make the output indistinguishable from reality”, and then it automatically learn a loss function appropriate for satisfying this goal. Like other GANs, Conditional GANs also have one discriminator (or critic depending on the loss function we are using) and one generator, and it tries to learn a conditional generative model which makes it suitable for Image-to-Image translation tasks, where we condition on an input image and generate a corresponding output image. </p>

> <p align = "justify"> If mathematically expressed, CGANs learn a mapping from observed image X and random noise vector z, to y, <i> G : {x, z} → y </i>. The generator G is trained to produce outputs that cannot be distinguished from <b> real </b> images by an adversarially trained discriminator, D, which in turn is itself optimized to do as well as possible at identifying the generator’s <b> fakes. </b> </p>

![2](https://user-images.githubusercontent.com/41862477/49689774-1698d080-fb4c-11e8-92af-dc3d48e66ec2.JPG)

> *The figure shown above illustrates the working of GAN in Conditional setting.*

### Loss Function

The objective of a conditional GAN can be expressed as:

```
Lc GAN (G, D) = Ex,y (log D(x, y)) + Ex,z (log(1 − D(x, G(x, z)))
```

<p align = "justify"> , where G tries to minimize this objective against an adversarial D that tries to maximize it, i.e. </p> 

```
G∗ = arg min(G)max(D) Lc GAN (G, D)
```

<p align = "justify"> It is beneficial to mix the GAN objective with a more traditional loss, such as L1 distance to make sure that, the ground truth and the output are close to each other in L1 sense </p>

```
L(G) = Ex,y,z ( ||y − G(x, z)|| )
```

<p align = "justify"> Without z, the net could still learn a mapping from x to y, but would produce deterministic outputs, and therefore would fail to match any distribution other than a <b> delta function </b>. Instead, the authors of Pix2pix provided noise only in the form of <b> dropout </b>, applied on several layers of the generator at <b> both training and test time </b>. </p>

<p align = "justify"> The Min-Max objective mentioned above was used in the original paper when GAN was first proposed by <b> Ian Goodfellow </b> in 2014, but unfortunately, it doesn't perform well due to vanishing gradients problems. Since then, there has been a lot of development, and many researchers have proposed different kinds of loss formulation (LS-GAN, WGAN, WGAN-GP) to overcome these issues. Authors of this paper used <b> Least-square </b> objective function while running their optimization process. </p>

### Network Architecture

<p align = "justify"> The GAN discriminator models high-frequency structure term, relying on an L1 term to force low-frequency correctness. In order to model high-frequencies, it is sufficient to restrict the attention to the structure in local image patches. Therefore, discriminator architecture was termed <b> PatchGAN </b> – that only penalizes structure at the scale of patches. This discriminator tries to classify if each N × N patch in an image is real or fake. We run this discriminator convolutionally across the image, and average all responses to provide the ultimate output of D. Patch GANs discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter. The recpetive field of the discriminator used was 70 * 70 (and was performing best compared to smaller and larger receptive fields). </p>

```
The 70 × 70 discriminator architecture is: C64 - C128 - C256 - C512
```

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

<p align = "justify"> Image-to-Image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. So, the authors in [this](https://arxiv.org/pdf/1703.10593.pdf) paper presented an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. </p> 

<p align = "justify"> <i> The goal is to learn a mapping <b> G : X → Y </b> such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, they coupled it with an inverse mapping <b> F : Y → X </b> and introduced a cycle consistency loss to enforce <b> F(G(X)) ≈ X </b> (and vice-versa). </i> </p>

### Motivation:

<p align = "justify"> Obtaining paired training data can be difficult and expensive. For example, only a couple of datasets exist for tasks like semantic segmentation, and they are relatively small. Obtaining input-output pairs for graphics tasks like <b> artistic stylization </b> can be even more difficult since the desired output is highly complex, typically requiring artistic authoring. For many tasks, like <b> object transfiguration </b> (e.g., zebra <-> horse), the desired output is not even well-defined. Therefore, the authors tried to present an algorithm that can learn to translate between domains without paired input-output examples. The primary assumption is that there exists some underlying relationship between the domains. Although there is a lack of supervision in the form of paired examples, supervision at the level of sets can still be exploited: <i> one set of images in domain X and a different set in domain Y. </i> </p>

<p align = "justify"> The optimal G thereby translates the domain X to a domain Y <i> distributed identically to Y. However, such a translation does not guarantee that an individual input x and output y are paired up in a meaningful way – there are infinitely many mappings G that will induce the same distribution over y </i>. Key points: </p>

- <p align = "justify"> Difficult to optimize adversarial objective in isolation - standard procedures often lead to the well-known problem of mode collapse. </p>
- <p align = "justify"> Exploited the property that translation should be <b> Cycle consistent </b>. Mathematically, translator G : X → Y and another translator F : Y → X, should be inverses of each other (and both mappings should be bijections). </p> 
- <p align = "justify"> Enforcing the structural assumption by training both the mapping G and F simultaneously, and adding a cycle consistency loss that encourages <b> <i> F(G(x)) ≈ x and G(F(y)) ≈ y </b> </i> </p>.

![2](https://user-images.githubusercontent.com/41862477/50504160-b1bd0000-0a91-11e9-9909-29b2121449b8.jpg)

> <p align = "justify"> <i> As illustrated in figure, their model includes two mappings <b> G : X → Y and F : Y → X. </b> In addition, they introduced two adversarial discriminators DX and DY , where DX aims to distinguish between images {x} and translated images {F(y)}; in the same way, DY aims to discriminate between {y} and {G(x)}. So, final objective contains two types of terms: adversarial losses for matching the distribution of generated images to the data distribution in the target domain; and cycle consistency losses to prevent the learned mappings G and F from contradicting each other. </i> </p>

#### Adversarial Loss:

<p align = "justify"> Adversarial loss is applied to both mapping functions -  G : X → Y and its discriminator DY and  F : Y → X and its discriminator DX, where G tries to generate images G(x) that look similar to images from domain Y , while DY aims to distinguish between translated samples G(x) and real samples y (similar condition holds for the other one). </p>

- Generator (G) tries to minimize ``` E[x∼pdata(x)] (D(G(x)) − 1)** 2```
- Discriminator (DY) tries to minimize ``` E[y∼pdata(y)] (D(y) − 1)**2 + E[x∼pdata(x)] D(G(x))**2```
- Generator (F) tries to minimize ``` E[y∼pdata(y)] (D(G(y)) − 1)** 2```
- Discriminator (DX) tries to minimize ``` E[x∼pdata(x)] (D(x) − 1)**2 + E[y∼pdata(y)] D(G(y))**2```

#### Cycle Consistency Loss:

<p align = "justify"> Adversarial training can, in theory, learn mappings G and F that produce outputs identically distributed as target domains Y and X respectively (strictly speaking, this requires G and F to be stochastic functions). However, with large enough capacity, a network can map the same set of input images to any random permutation of images in the target domain, where any of the learned mappings can induce an output distribution that matches the target distribution. Thus, adversarial losses alone cannot guarantee that the learned function can map an individual input xi to a desired output yi. To further reduce the space of possible mapping functions, learned functions should be cycle-consistent. </p>

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
- <p align = "justify"> These have special internal structures - map an image to itself via an intermediate representation that is a translation of the image into another domain. </p>
- <p align = "justify"> Can also be seen as a special case of <b> adversarial autoencoders </b>, which use an adversarial loss to train the bottleneck layer of an autoencoder to match an arbitrary target distribution. </p> 
- <p align = "justify"> The target distribution for the X → X autoencoder is the domain Y and for the Y → Y autoencoder is the domain X. </p>

## Implementation:

#### Training Details:

- Two **stride-2** convolutions, several **residual** blocks, and two **fractionally strided** convolutions with stride 1/2.
- 6 blocks for 128 × 128 images and 9 blocks for 256 × 256 and higher resolution training images.
- **Instance** normalization instead of batch normalization.
- <p align = "justify"> <b> Patch Discriminator </b> - 70 × 70 PatchGANs, which aim to classify whether 70 × 70 overlapping image patches are real or fake (more parameter efficient compared to full-image discriminator) </p>
- <p align = "justify"> To reduce model oscillation, update the discriminators using a history of generated images rather than the latest ones - always keep an image buffer of 50 previously generated images. </p>
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

> <p align = "justify"> <i> c7s1-k denote a 7×7 Convolution-InstanceNormReLU Layer with k filters and stride 1. dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2. Reflection padding was used to reduce artifacts. Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer. uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride 1/2. Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. After the last layer, a convolution is applied to produce a 1-dimensional output. **Do not** use InstanceNorm for the first C64 layer. Use leaky ReLUs with a slope of 0.2 </i>

#### Application - Photo generation from paintings: 

<p align = "justify"> For painting → photo, they found that it was helpful to introduce an additional loss to encourage the mapping to preserve color composition between the input and output. In particular, they regularized the generator to be near an identity mapping when real samples of the target domain are provided as the input to the generator i.e., </p>

```
Lidentity (G, F) = E[y∼pdata(y)] || G(y) − y || + E[x∼pdata(x)] || F(x) − x ||
```

### Results:

#### Photo -> Cezzane Paintings:

![3](https://user-images.githubusercontent.com/41862477/50507793-fdc57000-0aa4-11e9-8981-365b56c92b73.jpg)
![8](https://user-images.githubusercontent.com/41862477/50507795-fdc57000-0aa4-11e9-8da8-d92cb4775264.jpg)

#### Cezzane Paintings -> Photo:

![1](https://user-images.githubusercontent.com/41862477/50507839-2a798780-0aa5-11e9-99e6-7f7770ae8bdb.jpg)
![4](https://user-images.githubusercontent.com/41862477/50507840-2a798780-0aa5-11e9-9022-96e648314a86.jpg)

***

#### Monet Paintings -> Photo:

![2](https://user-images.githubusercontent.com/41862477/50508108-6b25d080-0aa6-11e9-854c-abb7b70ac864.jpg)
![3](https://user-images.githubusercontent.com/41862477/50508109-6bbe6700-0aa6-11e9-9065-81a591e9946d.jpg)

#### Photo -> Monet Paintings:

![4](https://user-images.githubusercontent.com/41862477/50508126-7aa51980-0aa6-11e9-8d28-bdbc0f4faf8e.jpg)
![5](https://user-images.githubusercontent.com/41862477/50508127-7aa51980-0aa6-11e9-91b6-1e2094dd9ab9.jpg)

***

#### Van_Gogh Paintings -> Photo:

![6](https://user-images.githubusercontent.com/41862477/50508433-ea67d400-0aa7-11e9-9c48-2f200b5f29c2.jpg)
![8](https://user-images.githubusercontent.com/41862477/50508435-eb006a80-0aa7-11e9-8be6-52907c335052.jpg)

#### Photo -> Van_Gogh Paintings:

![1](https://user-images.githubusercontent.com/41862477/50508407-d1f7b980-0aa7-11e9-80c0-c32d24b35764.jpg)
![4](https://user-images.githubusercontent.com/41862477/50508411-d328e680-0aa7-11e9-8422-53576a49d005.jpg)

***
***

## Deep Dream

![tony_stark](https://user-images.githubusercontent.com/41862477/51070752-72cfa280-166c-11e9-92de-e5805804602e.jpg)
![layer_3](https://user-images.githubusercontent.com/41862477/51070747-72370c00-166c-11e9-9590-29b2afad65d7.jpg)
![layer_4](https://user-images.githubusercontent.com/41862477/51070748-72370c00-166c-11e9-9539-e505346cc2fa.jpg)
![layer_7](https://user-images.githubusercontent.com/41862477/51070749-72370c00-166c-11e9-9d8d-ee42be071f52.jpg)
![layer_9](https://user-images.githubusercontent.com/41862477/51070750-72370c00-166c-11e9-932d-2bb959ab04f1.jpg)
![layer_10](https://user-images.githubusercontent.com/41862477/51070751-72cfa280-166c-11e9-9668-06851dda4e01.jpg)

***
***

#### Many more to come soon!

***Thanks for going through this post! Any feedbacks are duly appreciated.***
