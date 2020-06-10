# AI Art

***

## Motivation

<p align = "justify"> Creativity is something we closely associate with what it means to be human. But with digital technology now enabling machines to recognize, learn from, and respond to humans and the world, an inevitable question follows: </p>

> <i> Can a machine be creative, and will artificial intelligence be able to make art? </i>

<p align = "justify"> It could be argued that the ability of machines to learn what things look like, and then make convincing new examples marks the advent of creative AI. I will cover four different Deep Learning models in this tutorial to create novel arts, solely by code - <b> Style Transfer, Pix2Pix, CycleGAN</b>, and <b>Deep Dream. </b> </p>

***

## [Neural Style Transfer](https://arxiv.org/abs/1508.06576)

<p align = "justify"> Style Transfer is one of the most fun techniques in Deep learning. It combines the two images, namely, a <b> Content </b> image (C) and a <b> Style </b> image (S), to create an <b> Output </b> image (G). The Output image G combines the Content of image C with the Style of image S. </p>

![neural-style](https://user-images.githubusercontent.com/41862477/49682529-b23e2880-fadb-11e8-8625-82fc2b14c487.png)

<p align = "justify"> Style Transfer uses a pre-trained Convolutional Neural Network <b> VGG-19, </b> (because of it's simple and hierarchical design) which already can recognize a variety of <i> low-level features </i> (at the earlier layers) and <i> high-level features </i> (at the deeper layers). Style Transfer incorporates <i> three </i> different kinds of losses: </p>

- **Content Cost**: **J**<sub>Content</sub> (C, G)
- **Style Cost**: **J**<sub>Style</sub> (S, G)
- **Total Variation (TV) Cost**: **J**<sub>TV</sub> (G)

*Putting all together*: **J**<sub>Total</sub> (G) = &alpha; * **J**<sub>Content</sub> (C, G) + &beta; * **J**<sub>Style</sub> (S, G) + &gamma; * **J**<sub>TV</sub> (G). Let's delve deeper to know more profoundly what's going on under the hood!

###  Content Cost

<p align = "justify"> Generally, each layer in the network defines a non-linear filter bank whose complexity increases with the position of the layer in the network. The first few layers of the ConvNet tend to detect low-level features such as edges and simple textures, and the last few layers tend to detect high-level features such as more complex textures as well as features specific to different classes. <b>Content loss</b> tries to make sure that the Output image <b>G</b> has similar content as the Input image <b>C</b>, for which, we minimize the (<b>MSE</b>) loss between the feature maps of the respective images.
 
<i> Practically, we get the most visually pleasing results if we choose a layer in the middle of the network - neither too shallow nor too deep. </i> The higher layers in the network capture the high-level content in terms of objects and their arrangement in the input image, but do not constrain the exact pixel values of the reconstruction very much. In contrast, reconstructions from the lower layers simply reproduce the exact pixel values of the original image. 

<p align = "justify"> Let a(C) be the hidden layer activations which is a <b> N<sub>h</sub> * N<sub>w</sub> * N<sub>c</sub> </b> tensor, and let a(G) be the corresponding hidden layer activations of the Output image. Finally, the <b> Content Cost </b> function is defined as follows: </p>

![3](https://user-images.githubusercontent.com/41862477/49682789-6772df80-fae0-11e8-8f7c-5805421e8121.JPG)

<p align = "justify"> N<sub>h</sub>, N<sub>w</sub>, N<sub>c</sub> are the height, width, and the number of channels of the hidden layer chosen. To compute the cost J<sub>Content</sub> (C, G), it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below. </p>

![1](https://user-images.githubusercontent.com/41862477/49682841-10b9d580-fae1-11e8-851f-ec9fbf37dd92.JPG)

<p align = "justify"> <i> The first image is the original one, while the remaining are the reconstructions that we get when layers <b> Conv_1_2, Conv_2_2, Conv_3_2, Conv_4_2, and Conv_5_2 </b> (left to right and top to bottom) are used in the Content loss. </i> </p> 

![Con_recons_1_2](https://user-images.githubusercontent.com/41862477/82235677-a8ffef00-9950-11ea-8e38-513055c487cf.jpg)
![Con_recons_1_2](https://user-images.githubusercontent.com/41862477/82235677-a8ffef00-9950-11ea-8e38-513055c487cf.jpg)
![Con_recons_2_2](https://user-images.githubusercontent.com/41862477/82235682-aac9b280-9950-11ea-8885-4b8775638bbe.jpg)
![Con_recons_3_2](https://user-images.githubusercontent.com/41862477/82235683-abfadf80-9950-11ea-95b8-d9b8836ffa58.jpg)
![Con_recons_4_2](https://user-images.githubusercontent.com/41862477/82235686-ac937600-9950-11ea-9fe3-14dd979106cc.jpg)
![Con_recons_5_2](https://user-images.githubusercontent.com/41862477/82235688-ad2c0c80-9950-11ea-8a3b-d592d2bfee82.jpg)  


### Style Cost

<p align = "justify"> To understand it better, we first need to know something about the <b> Gram Matrix </b>. In linear algebra, the Gram matrix G of a set of vectors  (v1, …, vn) is the matrix of dot products, whose entries are <i> G<sub>(i, j)</sub> = np.dot(v<sub>i</sub>, v<sub>j</sub>) </i>. In other words, <i> G (i, j) </i> compares how similar v<sub>i</sub> is to v<sub>j</sub>. If they are highly similar, the outcome would be a large value, otherwise, it would be low suggesting lower correlation. In Style Transfer, we can compute the Gram matrix by multiplying the <b> unrolled </b> filter matrix with its transpose as shown below: </p>

![2](https://user-images.githubusercontent.com/41862477/49682895-f8968600-fae1-11e8-8fbd-b754c625542a.JPG)

<p align = "justify"> The result is a matrix of dimension <i> (n<sub>C</sub>, n<sub>C</sub>) </i> where n<sub>C</sub> is the number of filters. The value <i> G (i, j) </i> measures how similar the activations of filter i are to the activations of filter j. One important part of the gram matrix is that the diagonal elements such as G (i, i) measures how active filter i is. For example, suppose filter i is detecting vertical textures in the image, then G (i, i)  measures how common vertical textures are in the image as a whole. <i> By capturing the prevalence of different types of features G (i, i), as well as how much different features occur together G (i, j), the Gram matrix G measures the <b> Style </b> of an image. </i>

<p align = "justify"> After we have the Gram matrix, we want to minimize the distance between the Gram matrix of the Style image S and that of the Output image G. Usually, we take more than one layers in account to calculate <b> Style cost </b> as opposed to Content cost (in which only one layer is sufficient), and the reason for doing so is discussed later on in the post. For a single hidden layer, the corresponding style cost is defined as: </p>

![4](https://user-images.githubusercontent.com/41862477/49683030-54620e80-fae4-11e8-9f79-a500da7f12c3.JPG)

### Total Variation (TV) Cost

<p align = "justify"> It acts like a regularizer which encourages spatial smoothness in the generated image (G). This was not used in the original paper proposed by Gatys et al., but it sometimes improve the results. For 2D signal (or image), it is defined as follows: </p> 

![5](https://user-images.githubusercontent.com/41862477/49683156-1b2a9e00-fae6-11e8-8321-34b3c1173175.JPG)

### Experiments

> What will happen if we zero out the coefficients of the Content and TV loss, and take activation from only one layer  to compute the Style cost?

<p align = "justify"> As many of you might have guessed, the optimization algorithm will now only minimize the Style cost. So, for a given <b> Style image </b>, we would see what kind of brush-strokes will the model try to enforce in the final generated image (G). Remember, we started with only one layer's activation in the Style cost, so running the experiments for different layers would give different kind of brush-strokes that would be there in the final generated image. Suppose the style image is famous <b> The great wall of Kanagawa </b> shown below: </p>

![6](https://user-images.githubusercontent.com/41862477/49683530-af97ff00-faec-11e8-9d30-e3bc15e9fa88.jpg)

Here are the brush-strokes that we get after running the experiment taking different layers, one at a time!

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

<p align = "justify"> The reason behind running this experiment was that the authors of the original paper gave equal weight to the styles learned by different layers while calculating the <b> Total Style Cost </b>. Now, that's not intuitive at all after looking at these images, because we can see that styles learned by the shallower layers are more aesthetically pleasing, compared to what deeper layers learned. So, we would like to assign a lower weight to the deeper layers and higher to the shallower ones; exponentially decreasing the weights as we go deeper and deeper could be one way. </p>

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

## [Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)](https://arxiv.org/pdf/1611.07004.pdf)

<img width="1149" alt="Screenshot 2020-05-18 at 10 50 55 PM" src="https://user-images.githubusercontent.com/41862477/82241656-2aa84a80-995a-11ea-9968-686294f97414.png">

<p align = "justify"> The authors of this paper investigated Conditional adversarial networks as a general-purpose solution to <b> Image-to-Image Translation </b> problems. These networks not only learn the mapping from the input image to output image but also learn a loss function to train this mapping. If we take a naive approach and ask CNN to minimize just the Euclidean distance between predicted and ground truth pixels, it will tend to produce blurry results. This is because Euclidean distance is minimized by averaging all plausible outputs, which causes blurring. </p>

<p align = "justify"> In Generative Adversarial Networks settings, we could specify only a high-level goal, like “make the output indistinguishable from reality”, and then it automatically learns a loss function appropriate for satisfying this goal. Like other GANs, Conditional GAN has a discriminator (or critic depending on the loss function we are using) and a generator, and the overall goal is to learn a mapping, where we condition on an input image and generate a corresponding output image. In analogy to automatic language translation, automatic image-to-image translation is defined as the task of translating one possible representation of a scene into another, given sufficient training data. </p>

<p align = "justify> Most formulations treat the output space as “unstructured” in the sense that each output pixel is considered conditionally independent from all others given the input image. Conditional GANs instead learn a structured loss. Structured losses penalize the joint configuration of the output. </p>

> <p align = "justify"> Mathematically, CGANs learn a mapping from observed image X and random noise vector z, to y, <i> G: {x,z} → y. </i> The generator G is trained to produce output that cannot be distinguished from the <b> real </b> images by an adversarially trained discriminator, D, which in turn is optimized to perform best at identifying the <b> fake </b> images generated by the generator. The figure shown below illustrates the working of GAN in the Conditional setting. </p>

<img width="1079" alt="Screenshot 2020-05-18 at 11 17 51 PM" src="https://user-images.githubusercontent.com/41862477/82243881-f3d43380-995d-11ea-8877-5ccdf4828680.png">


### Loss Function

The objective of a conditional GAN can be expressed as:

<code>
L<sub>cGAN</sub> (G,D) = <b>E</b><sub>x,y</sub> [log D(x, y)] + <b>E</b><sub>x,z</sub> [log (1 − D(x, G(x, z))], 
</code> 

<p align = "justify"> where G tries to minimize this objective against an adversarial D that tries to maximize it. It is beneficial to mix the GAN objective with a more traditional loss, such as L1 distance to make sure that, the ground truth and the output are close to each other in L1 sense.
<code>
L<sub>L1</sub> (G) = E<sub>x,y,z</sub> [ ||y − G(x, z)||<sub>1</sub> ].

</code></p>

<p align = "justify"> Without z, the net could still learn a mapping from x to y, but would produce deterministic output, and therefore would fail to match any distribution other than a <b> delta function. </b> So, the authors provided noise in the form of <b> dropout, </b> applied on several layers of the generator at both the <b>training</b> and <b>test</b> time. Despite the dropout noise, there is only minor stochasticity in the output. The complete objective is now, </b>
<code>
G<sup>∗</sup> = <b>arg</b> min<sub>G</sub> max<sub>D</sub> L<sub>cGAN</sub> (G,D) + &lambda;L<sub>L1</sub> (G)
</code></p>

<p align = "justify"> The Min-Max objective mentioned above was proposed by <b> Ian Goodfellow </b> in 2014 in his original paper, but unfortunately, it doesn't perform well because of vanishing gradients problems. Since then, there has been a lot of development, and many researchers have proposed different kinds of loss formulations (LS-GAN, WGAN, WGAN-GP) to alleviate vanishing gradients. I used <b> Least-square </b> objective function while optimizing the architecture. </p>

### Network Architecture

#### Generator:

<p align = "justify"> The input and output differ only in surface appearance and are renderings of the same underlying structure. Therefore, structure in the input is roughly aligned with the structure in the output. The generator architecture is designed around these considerations only. For many image translation problems, there is a great deal of low-level information shared between the input and output, and it would be desirable to shuttle this information directly across the net. To give the generator a means to circumvent the bottleneck for information like this, skip connections are added following the general shape of a <b>U-Net.</b> Specifically, skip connections are added between each layer i and layer n − i, where n is the total number of layers. Each skip connection simply concatenates all channels at layer i with those at layer n − i. The U-Net encoder-decoder architecture consists of::: <b>Encoder:</b> <code> C64-C128-C256-C512-C512-C512-C512-C512</code>, and <b>U-Net Decoder:</b> <code> CD1024-CD1024-CD1024-CD1024-CD512-CD256-CD128,</code> where Ck denote a <i>Convolution-BatchNorm-ReLU</i> layer with k filters, and CDk denotes a <i>Convolution-BatchNorm-Dropout-ReLU</i> layer with a dropout rate of 50%. </p>

#### Discriminator:

<p align = "justify"> The GAN discriminator models high-frequency structure term, and relies on the L1 term to force low-frequency correctness. To model high-frequencies, it is sufficient to restrict the attention to the structure in local image patches. Therefore, discriminator architecture was termed <b> PatchGAN </b> – that only penalizes structure at the scale of patches. This discriminator tries to classify if each N × N patch in an image is real or fake. We run this discriminator convolutionally across the image, and average all responses to provide the ultimate output of D. Patch GANs discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter. The receptive field of the discriminator used was 70 * 70 and was performing best compared to other smaller and larger receptive fields. <code> The 70 × 70 discriminator architecture is: C64 - C128 - C256 - C512 </code> </p>

### Training details

- **Dropout** is used both at the training and test time.
- **Instance normalization** is used instead of batch normalization.
- All convolution kernels are of size 4 × 4 and are applied with stride 2.
- Both L1 and CGAN loss are important to reduce the artifacts in the final output.
- Normalization is not applied to the first layer in the encoder and discriminator. 
- **Adam solver** is used with a learning rate of 2e-4, and momentum parameters β1 = 0.5, β2 = 0.999.
- All ReLUs in the encoder and discriminator are leaky, with slope 0.2, while ReLUs in the decoder are not leaky.
- Objective function was divided by 2 while optimizing D, which slows down the rate at which D learns relative to G. 

### Results

<p align = "justify"> Generated image, True label, and Imput image from left to right. </p>

#### Cityscapes:

![1](https://user-images.githubusercontent.com/41862477/82577615-dcd55180-9ba8-11ea-9c8d-dbe958002e06.png)
![3](https://user-images.githubusercontent.com/41862477/82577633-e2329c00-9ba8-11ea-889c-fc6d0ee29da2.png)
![4](https://user-images.githubusercontent.com/41862477/82577637-e2cb3280-9ba8-11ea-81cf-53ff1a56039c.png)

#### Facades:
![2](https://user-images.githubusercontent.com/41862477/82696095-fc8d7800-9c83-11ea-80c4-0d0ed3b745d3.png)
![2](https://user-images.githubusercontent.com/41862477/82696106-01522c00-9c84-11ea-8aa2-c55788309f51.png)
![3](https://user-images.githubusercontent.com/41862477/82696108-02835900-9c84-11ea-9ee0-6b8c98f0cd72.png)

***

## [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)

<img width="995" align="center" alt="Screenshot 2020-05-23 at 11 40 22 AM" src="https://user-images.githubusercontent.com/41862477/82723149-4c9f2580-9cea-11ea-98cf-bf80e2428a4b.png">

<p align = "justify"> Image-to-Image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. So, the authors in the paper presented an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. </p> 

<p align = "justify"> <i> The goal is to learn a mapping <b> G : X → Y </b> such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. </i> Because this mapping is highly under-constrained, they coupled it with an inverse mapping <b> F : Y → X </b> and introduced a cycle consistency loss to enforce <b> F(G(X)) ≈ X </b> (and vice-versa). </p>

### Motivation:

<p align = "justify"> Obtaining paired training data can be difficult and expensive. For example, only a couple of datasets exist for tasks like semantic segmentation, and they are relatively small. Obtaining input-output pairs for graphics tasks like <b> artistic stylization </b> can be even more difficult since the desired output is highly complex, typically requiring artistic authoring. For many tasks, like <b> object transfiguration </b> (e.g., zebra <-> horse), the desired output is not even well-defined. Therefore, the authors tried to present an algorithm that can learn to translate between domains without paired input-output examples. The primary assumption is that there exists some underlying relationship between the domains. Although there is a lack of supervision in the form of paired examples, supervision at the level of sets can still be exploited: <i> one set of images in domain X and a different set in domain Y. </i> </p>

<p align = "justify"> The optimal G thereby translates the domain X to a domain Y <i> distributed identically to Y. However, such a translation does not guarantee that an individual input x and output y are paired up in a meaningful way – there are infinitely many mappings G that will induce the same distribution over y </i>. Key points: </p>

- <p align = "justify"> Difficult to optimize adversarial objective in isolation - standard procedures often lead to the well-known problem of mode collapse. </p>
- <p align = "justify"> Exploited the property that translation should be <b> Cycle consistent </b>. Mathematically, translator G : X → Y and another translator F : Y → X, should be inverses of each other (and both mappings should be bijections). </p> 
- <p align = "justify"> Enforcing the structural assumption by training both the mapping G and F simultaneously, and adding a cycle consistency loss that encourages <b> F(G(x)) ≈ x and G(F(y)) ≈ y. </b> </p>

![2](https://user-images.githubusercontent.com/41862477/50504160-b1bd0000-0a91-11e9-9909-29b2121449b8.jpg)

> <p align = "justify"> <i> As illustrated in figure, their model includes two mappings <b> G : X → Y and F : Y → X. </b> In addition, they introduced two adversarial discriminators DX and DY , where DX aims to distinguish between images {x} and translated images {F(y)}; in the same way, DY aims to discriminate between {y} and {G(x)}. So, final objective contains two types of terms: adversarial losses for matching the distribution of generated images to the data distribution in the target domain; and cycle consistency losses to prevent the learned mappings G and F from contradicting each other. </i> </p>

#### Adversarial Loss:

<p align = "justify"> Adversarial loss is applied to both mapping functions -  G : X → Y and its discriminator DY and  F : Y → X and its discriminator DX, where G tries to generate images G(x) that look similar to images from domain Y , while DY aims to distinguish between translated samples G(x) and real samples y (similar condition holds for the other one). </p>

- Generator (G) tries to minimize: <code> E<sub>[x∼p<sub>data</sub>(x)]</sub> (D(G(x)) − 1)<sup>2</sup> </code>
- Discriminator (DY) tries to minimize: <code> E<sub>[y∼p<sub>data</sub>(y)]</sub> (D(y) − 1)<sup>2</sup> + E<sub>[x∼p<sub>data</sub>(x)]</sub> D(G(x))<sup>2</sup> </code>
- Generator (F) tries to minimize <code> E<sub>[y∼p<sub>data</sub>(y)]</sub> (D(G(y)) − 1)<sup>2</sup> </code>
- Discriminator (DX) tries to minimize: <code> E<sub>[x∼p<sub>data</sub>(x)]</sub> (D(x) − 1)<sup>2</sup> + E<sub>[y∼p<sub>data</sub>(y)]</sub> D(G(y))<sup>2</sup> </code>

#### Cycle Consistency Loss:

<p align = "justify"> Adversarial training can, in theory, learn mappings G and F that produce outputs identically distributed as target domains Y and X respectively (strictly speaking, this requires G and F to be stochastic functions). However, with large enough capacity, a network can map the same set of input images to any random permutation of images in the target domain, where any of the learned mappings can induce an output distribution that matches the target distribution. Thus, adversarial losses alone cannot guarantee that the learned function can map an individual input xi to a desired output yi. To further reduce the space of possible mapping functions, learned functions should be cycle-consistent. <code> L<sub>cyc</sub> (G, F) = E<sub>[x∼p<sub>data</sub>(x)]</sub> || F(G(x)) − x|| + E<sub>[y∼p<sub>data</sub>(y)]</sub> || G(F(y)) − y || </code> </p>


#### Full Objective:

<p align = "justify"> The full objective is: <code> L (G, F, D<sub>X</sub>, D<sub>Y</sub>) = L<sub>GAN</sub> (G, D<sub>Y</sub> , X, Y) + L<sub>GAN</sub> (F, D<sub>X</sub>, Y, X) + λL<sub>cyc</sub>(G, F) </code> , where lambda controls the relative importance of the two objectives. </p>

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

<p align = "justify"> <i> c7s1-k denote a 7×7 Convolution - InstanceNorm - ReLU Layer with k filters and stride 1. dk denotes a 3 × 3 Convolution - InstanceNorm - ReLU layer with k filters and stride 2. Reflection padding is used to reduce artifacts. Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer. uk denotes a 3 × 3 fractional-strided-ConvolutionInstanceNorm-ReLU layer with k filters and stride 1/2. Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2. After the last layer, a convolution is applied to produce a 1-dimensional output. No InstanceNorm after the first C64 layer. Leaky ReLUs are used with a slope of 0.2 </i> </p>

#### Application - Photo generation from paintings: 

<p align = "justify"> For painting → photo, they found that it was helpful to introduce an additional loss to encourage the mapping to preserve color composition between the input and output. In particular, they regularized the generator to be near an identity mapping when real samples of the target domain are provided as the input to the generator i.e., <code> L<sub>identity</sub> (G, F) = E<sub>[y∼p<sub>data</sub>(y)]</sub> || G(y) − y || + E<sub>[x∼p<sub>data</sub>(x)]</sub> || F(x) − x || </code> </p>


### Results:

#### Photo -> Cezzane Paintings:

<p align = "center"> 
<img width="950" alt="Paint_4" src="https://user-images.githubusercontent.com/41862477/84287053-0e648b80-ab5d-11ea-96b0-2be7ea6fc6fd.png">
<img width="950" alt="Paint_5" src="https://user-images.githubusercontent.com/41862477/84287145-2b00c380-ab5d-11ea-9c66-3b7407c20d8b.png">
<img width="950" alt="Paint_6" src="https://user-images.githubusercontent.com/41862477/84287067-115f7c00-ab5d-11ea-9629-eb1952ff154c.png">
</p>

#### Cezzane Paintings -> Photo:

<p align = "center"> 
<img width="950" alt="Photo_1" src="https://user-images.githubusercontent.com/41862477/84286786-c6ddff80-ab5c-11ea-9701-283ba27ddc3b.png">
<img width="950" alt="Photo_2" src="https://user-images.githubusercontent.com/41862477/84286855-dbba9300-ab5c-11ea-836c-a8397668067d.png">
<img width="950" alt="Photo_3" src="https://user-images.githubusercontent.com/41862477/84286912-e83eeb80-ab5c-11ea-8347-59abf683107d.png">
</p>

<br />

***

## Deep Dream

![tony_stark](https://user-images.githubusercontent.com/41862477/51070752-72cfa280-166c-11e9-92de-e5805804602e.jpg)
![layer_3](https://user-images.githubusercontent.com/41862477/51070747-72370c00-166c-11e9-9590-29b2afad65d7.jpg)
![layer_4](https://user-images.githubusercontent.com/41862477/51070748-72370c00-166c-11e9-9539-e505346cc2fa.jpg)
![layer_7](https://user-images.githubusercontent.com/41862477/51070749-72370c00-166c-11e9-9d8d-ee42be071f52.jpg)
![layer_9](https://user-images.githubusercontent.com/41862477/51070750-72370c00-166c-11e9-932d-2bb959ab04f1.jpg)
![layer_10](https://user-images.githubusercontent.com/41862477/51070751-72cfa280-166c-11e9-9668-06851dda4e01.jpg)

***
