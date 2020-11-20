# AI Art

***
### Edit 2020/11/20:

> Support of [PyTorch Lightning](https://www.pytorchlightning.ai/) added to CycleGAN and Pix2Pix.
> Why PyTorch Lightning?

- Easier to reproduce
- Mixed Precision (16 bit and 32 bit) training support
- More readable by decoupling the research code from the engineering
- Less error prone by automating most of the training loop and tricky engineering
- Keep all the flexibility (this is all pure PyTorch), but removes a ton of boilerplate
- Scalable to any hardware without changing your model (CPU, Single/Multi GPU, TPU)


***

## Motivation

<p align = "justify"> Creativity is something we closely associate with what it means to be human. But with digital technology now enabling machines to recognize, learn from, and respond to humans, an inevitable question follows: <i> Can machines be creative? </i> </p>

<p align = "justify"> It could be argued that the ability of machines to learn what things look like, and then make convincing new examples marks the advent of creative AI. This tutorial will cover four different Deep Learning models to create novel arts, solely by code - <b> Style Transfer, Pix2Pix, CycleGAN, </b> and <b>Deep Dream. </b> </p>

***

## [Neural Style Transfer](https://arxiv.org/abs/1508.06576)

<p align = "justify"> Style Transfer is one of the most fun techniques in Deep learning. It combines the two images, namely, a <b> Content </b> image (C) and a <b> Style </b> image (S), to create an <b> Output </b> image (G). The Output image has the content of image C painted in the style of image S. </p>

<img src = https://user-images.githubusercontent.com/41862477/49682529-b23e2880-fadb-11e8-8625-82fc2b14c487.png width = 1000>

<p align = "justify"> <i>Style Transfer uses a pre-trained Convolutional Neural Network to get the content and style representations of the image, but why do these intermediate outputs within the pre-trained image classification network allow us to define style and content representations?</i> </p>
 
<p align = "justify"> These pre-trained models trained on image classification tasks can understand the image very well. This requires taking the raw image as input pixels and building an internal representation that converts the raw image pixels into a complex understanding of the features present within the image. The activation maps of first few layers represent low-level features like edges and textures; as we go deeper and deeper through the network, the activation maps represent higher-level features - objects like wheels, or eyes, or faces. Style Transfer incorporates <b> three </b> different kinds of losses: </p>

- **Content Cost**: **J**<sub>Content</sub> (C, G)
- **Style Cost**: **J**<sub>Style</sub> (S, G)
- **Total Variation (TV) Cost**: **J**<sub>TV</sub> (G)

<p align = "justify"> <i>Putting all together:</i> <b>J</b><sub>Total</sub> (G) = &alpha; x <b>J</b><sub>Content</sub> (C, G) + &beta; x <b>J</b><sub>Style</sub> (S, G) + &gamma; x <b>J</b><sub>TV</sub> (G). Let's delve deeper to know more profoundly what's going on under the hood! </p>

###  Content Cost

<p align = "justify"> Usually, each layer in the network defines a non-linear filter bank whose complexity increases with the position of the layer in the network. <b>Content loss</b> tries to make sure that the Output image <b>G</b> has similar content as the Input image <b>C</b>, by minimizing the L2 distance between their activation maps.
 
<p align = "justify"> <i> Practically, we get the most visually pleasing results if we choose a layer in the middle of the network - neither too shallow nor too deep. </i> The higher layers in the network capture the high-level content in terms of objects and their arrangement in the input image but do not constrain the exact pixel values of the reconstruction very much. In contrast, reconstructions from the lower layers simply reproduce the exact pixel values of the original image. </p>

<p align = "justify"> Let a(C) be the hidden layer activations which is a N<sub>h</sub> x N<sub>w</sub> x N<sub>c</sub> dimensional tensor, and let a(G) be the corresponding hidden layer activations of the Output image. Finally, the <b> Content Cost </b> function is defined as follows: </p>

<img src = https://user-images.githubusercontent.com/41862477/49682789-6772df80-fae0-11e8-8f7c-5805421e8121.JPG width = 500>

<p align = "justify"> N<sub>h</sub>, N<sub>w</sub>, N<sub>c</sub> are the height, width, and the number of channels of the hidden layer chosen. To compute the cost J<sub>Content</sub> (C, G), it might also be convenient to unroll these 3D volumes into a 2D matrix, as shown below. </p>

<img src = https://user-images.githubusercontent.com/41862477/49682841-10b9d580-fae1-11e8-851f-ec9fbf37dd92.JPG width = 1000>

<p align = "justify"> <i> The first image is the original one, while the remaining ones are the reconstructions when layers <b> Conv_1_2, Conv_2_2, Conv_3_2, Conv_4_2, and Conv_5_2 </b> (left to right and top to bottom) are chosen in the Content loss. </i> </p> 

<table>
  <tr>
<td><img src = https://user-images.githubusercontent.com/41862477/82235677-a8ffef00-9950-11ea-8e38-513055c487cf.jpg width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/82235677-a8ffef00-9950-11ea-8e38-513055c487cf.jpg width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/82235682-aac9b280-9950-11ea-8885-4b8775638bbe.jpg width = 285></td>
 </tr>
</table>

<table>
  <tr>
<td><img src = https://user-images.githubusercontent.com/41862477/82235683-abfadf80-9950-11ea-95b8-d9b8836ffa58.jpg width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/82235686-ac937600-9950-11ea-9fe3-14dd979106cc.jpg width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/82235688-ad2c0c80-9950-11ea-8a3b-d592d2bfee82.jpg width = 285></td>
 </tr>
</table>

### Style Cost

<p align = "justify"> To understand it better, we first need to know something about the <b> Gram Matrix </b>. In linear algebra, the Gram matrix G of a set of vectors  (v1, …, vn) is the matrix of dot products, whose entries are G(i, j) = np.dot(v<sub>i</sub>, v<sub>j</sub>). In other words, G(i, j) compares how similar v<sub>i</sub> is to v<sub>j</sub>. If they are highly similar, the outcome would be a large value, otherwise, it would be low suggesting a lower correlation. In Style Transfer, we can compute the Gram matrix by multiplying the <b> unrolled </b> filter matrix with its transpose as shown below: </p>

<img src = https://user-images.githubusercontent.com/41862477/49682895-f8968600-fae1-11e8-8fbd-b754c625542a.JPG width = 1000>

<p align = "justify"> The result is a matrix of dimension (n<sub>C</sub>, n<sub>C</sub>) where n<sub>C</sub> is the number of filters. The value G(i, j) measures how similar the activations of filter i are to the activations of filter j. One important part of the gram matrix is that the diagonal elements such as G(i, i) measures how active filter i is. For example, suppose filter i is detecting vertical textures in the image, then G(i, i)  measures how common vertical textures are in the image as a whole. </p>
 
<p align = "justify"> <i> By capturing the prevalence of different types of features G(i, i), as well as how much different features occur together G(i, j), the Gram matrix G measures the <b> Style </b> of an image. </i> Once we have the Gram matrix, we minimize the L2 distance between the Gram matrix of the Style image S and the Output image G. Usually, we take more than one layers in account to calculate the <b> Style cost </b> as opposed to Content cost (which only requires one layer), and the reason for doing so is discussed later on in the post. For a single hidden layer, the corresponding style cost is defined as: </p>

<img src = https://user-images.githubusercontent.com/41862477/49683030-54620e80-fae4-11e8-9f79-a500da7f12c3.JPG width = 500>

### Total Variation (TV) Cost

<p align = "justify"> It acts like a regularizer that encourages spatial smoothness in the generated image (G). This was not used in the original paper proposed by Gatys et al., but it sometimes improves the results. For 2D signal (or image), it is defined as follows: </p> 

<img src = https://user-images.githubusercontent.com/41862477/49683156-1b2a9e00-fae6-11e8-8321-34b3c1173175.JPG width = 500>

### Experiments

> What happens if we zero out the coefficients of the Content and TV loss, and consider only a single layer to compute the Style cost?

<p align = "justify"> As many of you might have guessed, the optimization algorithm will now only minimize the Style cost.  So, for a given <b> Style image </b>, we will see the different kinds of brush-strokes (depending on the layer used) that the model will try to enforce in the final generated image (G). Remember, we started with a single layer in the Style cost, so, running the experiments for different layers would give different kinds of brush-strokes. Suppose the style image is famous <b> The great wall of Kanagawa </b> shown below: </p>

<img src = https://user-images.githubusercontent.com/41862477/49683530-af97ff00-faec-11e8-9d30-e3bc15e9fa88.jpg width = 1000>

The brush-strokes that we get after running the experiment taking different layers one at a time are attached below.
<table>
  <tr>
<td><img src = https://user-images.githubusercontent.com/41862477/49683610-e15d9580-faed-11e8-8d3f-58de7ee88595.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49683611-e15d9580-faed-11e8-80d6-3d216487f678.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49683613-e15d9580-faed-11e8-836f-b8d3dab32f03.png width = 285></td>
 </tr>
</table>
<table>
  <tr>
<td><img src = https://user-images.githubusercontent.com/41862477/49683614-e1f62c00-faed-11e8-964f-6e0e4085cc3d.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49683615-e1f62c00-faed-11e8-9583-a6ca7cfc058b.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49683616-e1f62c00-faed-11e8-9cf2-cbc5c3f5e18b.png width = 285></td>
 </tr>
</table>
<table>
  <tr>
<td><img src = https://user-images.githubusercontent.com/41862477/49683617-e1f62c00-faed-11e8-9e09-4147889c3b01.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49683618-e28ec280-faed-11e8-92b3-f48787c98f8a.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49683619-e28ec280-faed-11e8-8076-85145ff382ea.png width = 285></td>
 </tr>
</table>

<p align = "justify"> <i> These are brush-strokes that the model learned when layers <b> Conv_2_2, Conv_3_1, Conv_3_2, Conv_3_3, Conv_4_1, Conv_4_3, Conv_4_4, Conv_5_1, and Conv_5_4 </b> (left to right and top to bottom) were used one at a time in the Style cost. </i> </p>

<p align = "justify"> The reason behind running this experiment was that the authors of the original paper gave equal weightage to the styles learned by different layers while calculating the <b> Total Style Cost. </b> Now, that's not intuitive at all after looking at these images, because we can see that styles learned by the shallower layers are much more aesthetically pleasing, compared to what deeper layers learned. So, we would like to assign a lower weight to the deeper layers and higher to the shallower ones (exponentially decreasing the weightage could be one way). </p>

### Results

<table>
  <tr>
<td><img src = https://user-images.githubusercontent.com/41862477/49685490-3b6c5400-fb0a-11e8-876a-526a95591cb5.jpg width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49685493-3c04ea80-fb0a-11e8-8a2a-822130da61d6.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49685494-3c04ea80-fb0a-11e8-8a9c-42e7173fdb1b.png width = 285></td>
  </tr>
</table>

<table>
  <tr>
<td><img src = https://user-images.githubusercontent.com/41862477/49685487-3ad3bd80-fb0a-11e8-833b-e34dfd340957.jpg width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49685493-3c04ea80-fb0a-11e8-8a2a-822130da61d6.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49685495-3c9d8100-fb0a-11e8-937f-b62c62a6016a.png width = 285></td>
  </tr>
</table>

<table>
  <tr>
<td><img src = https://user-images.githubusercontent.com/41862477/49685490-3b6c5400-fb0a-11e8-876a-526a95591cb5.jpg width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49685491-3b6c5400-fb0a-11e8-9161-1c6940d5e6bc.jpg width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49685496-3c9d8100-fb0a-11e8-9240-39be822aee63.png width = 285></td>
  </tr>
</table>

<table>
  <tr>
<td><img src = https://user-images.githubusercontent.com/41862477/49685490-3b6c5400-fb0a-11e8-876a-526a95591cb5.jpg width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49685492-3c04ea80-fb0a-11e8-8308-d770f4d0185d.jpg width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/49685497-3d361780-fb0a-11e8-8f13-57d2965ccbd0.png width = 285></td>
  </tr>
</table>

***

## [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf)

<img width="1000" src="https://user-images.githubusercontent.com/41862477/82241656-2aa84a80-995a-11ea-9968-686294f97414.png">

> If you don't know what <b> Generative Adversarial networks</b> are, please refer to this [blog](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html) before going ahead; it explains the intuition and mathematics behind the GANs. </p>

<p align = "justify"> Authors of this paper investigated Conditional adversarial networks as a general-purpose solution to <b> Image-to-Image Translation </b> problems. These networks not only learn the mapping from the input image to output image but also learn a loss function to train this mapping. If we take a naive approach and ask CNN to minimize just the Euclidean distance between predicted and ground truth pixels, it tends to produce blurry results; minimizing Euclidean distance averages all plausible outputs, which causes blurring. </p>

<p align = "justify"> <i> In Generative Adversarial Networks settings, we could specify only a high-level goal, like “make the output indistinguishable from reality”, and then it automatically learns a loss function appropriate for satisfying this goal. The conditional generative adversarial network, or cGAN for short, is a type of GAN that involves the conditional generation of images by a generator model. </i>Like other GANs, Conditional GAN has a discriminator (or critic depending on the loss function we are using) and a generator, and the overall goal is to learn a mapping, where we condition on an input image and generate a corresponding output image. In analogy to automatic language translation, automatic image-to-image translation is defined as the task of translating one possible representation of a scene into another, given sufficient training data. </p>

<p align = "justify"> Most formulations treat the output space as “unstructured” in the sense that each output pixel is considered conditionally independent from all others given the input image. Conditional GANs instead learn a structured loss. Structured losses penalize the joint configuration of the output. Mathematically, CGANs learn a mapping from observed image X and random noise vector z, to y, <i> G: {x,z} → y. </i> The generator G is trained to produce output that cannot be distinguished from the <b> real </b> images by an adversarially trained discriminator, D, which in turn is optimized to perform best at identifying the <b> fake </b> images generated by the generator. The figure shown below illustrates the working of GAN in the Conditional setting. </p>

<img width="1000" src="https://user-images.githubusercontent.com/41862477/82243881-f3d43380-995d-11ea-8877-5ccdf4828680.png">


### Loss Function

The objective of a conditional GAN can be expressed as:

<code>
L<sub>cGAN</sub> (G,D) = <b>E</b><sub>x,y</sub> [log D(x, y)] + <b>E</b><sub>x,z</sub> [log (1 − D(x, G(x, z))], 
</code> 

<p align = "justify"> where G tries to minimize this objective against an adversarial D that tries to maximize it. It is beneficial to mix the GAN objective with a more traditional loss, such as L1 distance to make sure that, the ground truth and the output are close to each other in L1 sense.
<code>
L<sub>L1</sub> (G) = E<sub>x,y,z</sub> [ ||y − G(x, z)||<sub>1</sub> ].

</code></p>

<p align = "justify"> Without z, the net could still learn a mapping from x to y, but would produce deterministic output, and therefore would fail to match any distribution other than a <b> delta function. </b> So, the authors provided noise in the form of <b> dropout; </b> applied it on several layers of the generator at both the <b>training</b> and <b>test</b> time. Despite the dropout noise, there is only minor stochasticity in the output. The complete objective is now, </b>
<code>
G<sup>∗</sup> = <b>arg</b> min<sub>G</sub> max<sub>D</sub> L<sub>cGAN</sub> (G,D) + &lambda;L<sub>L1</sub> (G)
</code></p>

<p align = "justify"> The Min-Max objective mentioned above was proposed by <b> Ian Goodfellow </b> in 2014 in his original paper, but unfortunately, it doesn't perform well because of vanishing gradients problem. Since then, there has been a lot of development, and many researchers have proposed different kinds of loss formulations (LS-GAN, WGAN, WGAN-GP) to alleviate vanishing gradients. Authors of this paper used <b> Least-square </b> objective function while optimizing the networks, which can be expressed as:</p>
<p align = "justify"> 
<code>
min L<sub>LSGAN</sub> (D) = 1/2 <b>E</b><sub>x,y</sub> [(D(x, y) - 1)<sup>2</sup>] + 0.5 * <b>E</b><sub>x,z</sub> [D(x, G(x, z))<sup>2</sup>]
</code> <br />
<code>
min L<sub>LSGAN</sub> (G) = 1/2 <b>E</b><sub>x,z</sub> [(D(x, G(x, z)) - 1)<sup>2</sup>]
</code>
</p>


### Network Architecture

#### Generator:

<p align = "justify"> <b>Assumption:</b> The input and output differ only in surface appearance and are renderings of the same underlying structure. Therefore, structure in the input is roughly aligned with the structure in the output. The generator architecture is designed around these considerations only. For many image translation problems, there is a great deal of low-level information shared between the input and output, and it would be desirable to shuttle this information directly across the net. To give the generator a means to circumvent the bottleneck for information like this, skip connections are added following the general shape of a <b>U-Net.</b> </p> 

<p align = "justify"> Specifically, skip connections are added between each layer i and layer n − i, where n is the total number of layers. Each skip connection simply concatenates all channels at layer i with those at layer n − i. The U-Net encoder-decoder architecture consists of <b>Encoder:</b> <code> C64-C128-C256-C512-C512-C512-C512-C512</code>, and <b>U-Net Decoder:</b> <code> C1024-CD1024-CD1024-CD1024-C512-C256-C128,</code> where Ck denote a <i>Convolution-BatchNorm-ReLU</i> layer with k filters, and CDk denotes a <i>Convolution-BatchNorm-Dropout-ReLU</i> layer with a dropout rate of 50%. </p>

#### Discriminator:

<p align = "justify"> The GAN discriminator models high-frequency structure term, and relies on the L1 term to force low-frequency correctness. To model high-frequencies, it is sufficient to restrict the attention to the structure in local image patches. Therefore, discriminator architecture was termed <b> PatchGAN </b> – that only penalizes structure at the scale of patches. This discriminator tries to classify if each N × N patch in an image is real or fake. The discriminator is run convolutionally across the image, and the responses get averaged out to provide the ultimate output. </p>

<p align = "justify"> Patch GANs discriminator effectively models the image as a Markov random field, assuming independence between pixels separated by more than a patch diameter. The receptive field of the discriminator used was 70 x 70 and was performing best compared to other smaller and larger receptive fields. <code> The 70 x 70 discriminator architecture is: C64 - C128 - C256 - C512 </code> </p>

> The [diagrams](https://www.tensorflow.org/tutorials/generative/pix2pix) attached below show the forward and backward propagation through the generator and discriminator!

<table>
  <tr>
<td><img width="450" height="500" src="https://user-images.githubusercontent.com/41862477/85034441-bc94b480-b19f-11ea-87ba-d2a7c5e8c559.png"></td>
<td><img width="450" height="500" src="https://user-images.githubusercontent.com/41862477/85034450-be5e7800-b19f-11ea-8c81-1898f2d04442.png"></td>
  </tr>
</table>

### Training Details:

- All convolution kernels are of size 4 × 4.
- Dropout is used both at the training and <b>test</b> time.
- **Instance** normalization is used instead of batch normalization.
- Normalization is not applied to the first layer in the encoder and discriminator.
- **Adam** is used with a learning rate of 2e-4, with momentum parameters β1 = 0.5, β2 = 0.999.
- All ReLUs in the encoder and discriminator are leaky, with slope **0.2**, while ReLUs in the decoder are not leaky. 

### Results

#### Cityscapes:

<img src="https://user-images.githubusercontent.com/41862477/84529965-fb3df100-acff-11ea-85db-a98e65bc61ad.png" width = 1000>
<img src="https://user-images.githubusercontent.com/41862477/84529974-fed17800-acff-11ea-81a4-28dbfd15b80c.png" width = 1000>
<img src="https://user-images.githubusercontent.com/41862477/84529984-009b3b80-ad00-11ea-8b53-15a77459658a.png" width = 1000>

#### Facades:

<img width="1000" alt="Image_1" src="https://user-images.githubusercontent.com/41862477/84530276-7bfced00-ad00-11ea-883f-a1ef5cecce12.png">
<img width="1000" alt="Image_2" src="https://user-images.githubusercontent.com/41862477/84530281-80c1a100-ad00-11ea-8ddc-1d3eb77209bd.png">
<img width="1000" alt="Image_4" src="https://user-images.githubusercontent.com/41862477/84530289-828b6480-ad00-11ea-9590-173c478b433e.png">
<img width="1000" alt="Image_5" src="https://user-images.githubusercontent.com/41862477/84530294-84552800-ad00-11ea-9cc2-85bb268e9351.png">

***

## [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)

<img width="1000" src="https://user-images.githubusercontent.com/41862477/82723149-4c9f2580-9cea-11ea-98cf-bf80e2428a4b.png">

<p align = "justify"> The image-to-Image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data is not available, so, authors of this paper presented an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. </p> 

<p align = "justify"> <i> The goal is to learn a mapping <b> G: X → Y </b> such that the distribution of images G(X) is indistinguishable from the distribution Y using an adversarial loss. </i> Because this mapping is highly under-constrained, they coupled it with an inverse mapping <b> F: Y → X </b> and introduced a cycle consistency loss to enforce <b> F(G(X)) ≈ X </b>(and vice-versa). </p>

### Motivation:

<p align = "justify"> Obtaining paired training data can be difficult and expensive. For example, only a couple of datasets exist for tasks like semantic segmentation, and they are relatively small. Obtaining input-output pairs for graphics tasks like <b> artistic stylization </b> can be even more difficult since the desired output is highly complex, and typically requires artistic authoring. For many tasks, like <b> object transfiguration </b> (e.g., zebra <-> horse), the desired output is not even well-defined. Therefore, the authors tried to present an algorithm that can learn to translate between domains without paired input-output examples. The primary assumption is that there exists some underlying relationship between the domains. </p>

<p align = "justify"> Although there is a lack of supervision in the form of paired examples, supervision at the level of sets can still be exploited: <i> one set of images in domain X and a different set in domain Y. </i> The optimal <b>G</b> thereby translates the domain <b>X</b> to a domain <b>Y</b> distributed identically to <b>Y</b>. However, such a translation does not guarantee that an individual input x and output y are paired up in a meaningful way – <i>there are infinitely many mappings <b>G</b> that will induce the same distribution over <b>y.</b></i></p>

<img src = https://user-images.githubusercontent.com/41862477/85051305-f9b77180-b1b4-11ea-982a-e6b54f6d8fa9.png width = 1000>

<p align = "justify"> As illustrated in the figure, the model includes two mappings <b> G: X → Y</b> and <b>F: Y → X. </b> Besides, two adversarial discriminators are introduced, <b>D<sub>X</sub></b> and <b>D<sub>Y</sub></b>; task of D<sub>X</sub> is to discriminate images <b>x</b> from translated images <b>F(y)</b>, whereas D<sub>Y</sub> aims to discriminate <b>y</b> from <b>G(x)</b>. So, the final objective has two different loss terms: adversarial loss for matching the distribution of generated images to the data distribution in the target domain, and cycle consistency loss to prevent the learned mappings <b>G</b> and <b>F</b> from contradicting each other. </p>

### Loss Formulation

#### Adversarial Loss:

<p align = "justify"> Adversarial loss is applied to both the mapping functions -  <b>G: X → Y</b> and <b>F: Y → X.</b> <b>G</b> tries to generate images <b>G(x)</b> that look similar to images from domain <b>Y</b>, and <b>D<sub>Y</sub></b> tries to distinguish the translated samples <b>G(x)</b> from real samples y (similar argument holds for the other one). </p>

- Generator (G) tries to minimize: <code> E<sub>[x∼p<sub>data</sub>(x)]</sub> (D(G(x)) − 1)<sup>2</sup> </code>
- Discriminator (D<sub>Y</sub>) tries to minimize: <code> E<sub>[y∼p<sub>data</sub>(y)]</sub> (D(y) − 1)<sup>2</sup> + E<sub>[x∼p<sub>data</sub>(x)]</sub> D(G(x))<sup>2</sup> </code>
- Generator (F) tries to minimize <code> E<sub>[y∼p<sub>data</sub>(y)]</sub> (D(G(y)) − 1)<sup>2</sup> </code>
- Discriminator (D<sub>X</sub>) tries to minimize: <code> E<sub>[x∼p<sub>data</sub>(x)]</sub> (D(x) − 1)<sup>2</sup> + E<sub>[y∼p<sub>data</sub>(y)]</sub> D(G(y))<sup>2</sup> </code>

#### Cycle Consistency Loss:

<p align = "justify"> Adversarial training can, in theory, learn mappings G and F that produce outputs identically distributed as target domains Y and X respectively (strictly speaking, this requires G and F to be stochastic functions). However, with large enough capacity, a network can map the same set of input images to any random permutation of images in the target domain, where any of the learned mappings can induce an output distribution that matches the target distribution. Thus, adversarial losses alone cannot guarantee that the learned function can map an individual input x<sub>i</sub> to a desired output y<sub>i</sub>. To further reduce the space of possible mapping functions, learned functions should be cycle-consistent. <code> L<sub>cyc</sub> (G, F) = E<sub>[x∼p<sub>data</sub>(x)]</sub> || F(G(x)) − x|| + E<sub>[y∼p<sub>data</sub>(y)]</sub> || G(F(y)) − y || </code> </p>

#### Full Objective:

<p align = "justify"> The full objective is: <code> L (G, F, D<sub>X</sub>, D<sub>Y</sub>) = L<sub>GAN</sub> (G, D<sub>Y</sub> , X, Y) + L<sub>GAN</sub> (F, D<sub>X</sub>, Y, X) + λ L<sub>cyc</sub> (G, F) </code> , where lambda controls the relative importance of the two objectives. <i><b>λ</b> is set to 10 in the final loss equation.</i> For <b>painting → photo</b>, authors found that it was helpful to introduce an additional loss to encourage the mapping to preserve color composition between the input and output. In particular, they regularized the generator to be near an identity mapping when real samples of the target domain are provided as the input to the generator i.e., <code> L<sub>identity</sub> (G, F) = E<sub>[y∼p<sub>data</sub>(y)]</sub> || G(y) − y || + E<sub>[x∼p<sub>data</sub>(x)]</sub> || F(x) − x ||. </code></p>

#### Key Takeaways:

- <p align = "justify"> It is difficult to optimize adversarial objective in isolation - standard procedures often lead to the well-known problem of mode collapse. Both the mappings G and F are trained simultaneously to enforce the structural assumption.</p>
- <p align = "justify"> The translation should be <b> Cycle consistent; </b> mathematically, translator G: X → Y and another translator F: Y → X, should be inverses of each other (and both mappings should be bijections). </p>
- <p align = "justify"> It is similar to training two <b>autoencoders</b> - <b>F ◦ G:</b> X → X jointly with <b>G ◦ F:</b> Y → Y. These autoencoders have special internal structure - map an image to itself via an intermediate repr that is a translation of the image into another domain. </p>
- <p align = "justify"> It can also be treated as a special case of <b> adversarial autoencoders </b>, which use an adversarial loss to train the bottleneck layer of an autoencoder to match an arbitrary target distribution. </p>

### Network Architecture

#### Generator:
<p align = "justify"> Authors adopted the Generator's architecture from the neural style transfer and super-resolution paper. The network contains two stride-2 convolutions, several residual blocks, and two fractionally-strided convolutions with stride 1/2. 6 or 9 ResBlocks are used in the generator depending on the size of the training images. <b>Instance</b> normalization is used instead of <b>batch</b> normalization.
<code> <b>128 x 128</b> images: c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3 </code>
<code> <b>256 x 256</b> images: c7s1-64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256, R256, u128, u64, c7s1-3 </code> </p>

#### Discriminator:
<p align = "justify"> The same 70 x 70 PatchGAN discriminator is used, which aims to classify whether 70 x 70 overlapping image patches are real or fake (more parameter efficient compared to full-image discriminator). To reduce model oscillations, discriminators are updated using a history of generated images rather than the latest ones with a probability of <b>0.5</b>.<code> <b>70 x 70 PatchGAN: </b>C64-C128-C256-C512 </code></p>

> <p align = "justify"> c7s1-k denote a 7×7 Convolution - InstanceNorm - ReLU Layer with k filters and stride 1. dk denotes a 3 × 3 Convolution - InstanceNorm - ReLU layer with k filters and stride 2. Reflection padding is used to reduce artifacts. Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters on both layer. uk denotes a 3 × 3 Deconv - InstanceNorm - ReLU layer with k filters and stride 1/2. Ck denote a 4 × 4 Convolution - InstanceNorm - LeakyReLU layer with k filters and stride 2. After the last layer, a convolution is applied to produce a 3-channels output for generator and 1-channel output for discriminator. No InstanceNorm in the first C64 layer.</p>


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
<img src = https://user-images.githubusercontent.com/41862477/85164051-68fa9780-b281-11ea-87ae-2d255f910fed.png height = 500 width = 1000>

<table>
  <tr>
<td><img src = https://user-images.githubusercontent.com/41862477/84597918-b1324800-ae84-11ea-82d4-aa4aaeb930d4.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/84620003-66e8af80-aef4-11ea-8b70-69c8ed492f33.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/84620018-74059e80-aef4-11ea-8f26-4f65002ed7c1.png width = 285></td>
 </tr>
</table>
<table>
  <tr>
<td><img src = https://user-images.githubusercontent.com/41862477/84620032-8089f700-aef4-11ea-8453-755b00d867a1.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/84620043-8c75b900-aef4-11ea-9858-5706ba5851b0.png width = 285></td>
<td><img src = https://user-images.githubusercontent.com/41862477/84620048-8ed81300-aef4-11ea-8a07-7fdbd831fd32.png width = 285></td>
 </tr>
</table>

***
