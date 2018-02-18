A very popular type of generative model is the **generative adverserial network**, or **GAN**. A GAN is actually two networks: the **generator** which attempts to produce convinving output and the **discriminator** which compares the output of the generator to real samples. Practically, what is happening here is there are two networks training each other to cross purposes based upon the other's feedback. 

[illustration]

So we basically have neural networks evaluating each other, improving each other, and this is a spectacularly powerful concept. Additionally, they are quite flexible like other network architectures in that the networks can be anything from simple, single layer networks to deep convolutional networks (also referred to as a **DCGAN**) or any other architecture, as well as being able to stack and combine them in a variety of ways. GANs have become especially popular in image generation but are equally useful in any generative task from text to numerical data.

```python
from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tflearn

# load & prep data
import tflearn.datasets.mnist as mnist
X_train, _, _, _ = mnist.load_data()

# define input dimensions
image_dim = 784 # 28x28 pixels
z_dim = 200 # noise
total_samples = len(X_train)

def generator(x, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        x = tflearn.fully_connected(x, 256, activation='relu')
        x = tflearn.fully_connected(x, image_dim, activation='sigmoid')
        return x

def discriminator(x, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        x = tflearn.fully_connected(x, 256, activation='relu')
        x = tflearn.fully_connected(x, 1, activation='sigmoid')
        return x

# define networks
gen_input = tflearn.input_data(shape=[None, z_dim], name='input_noise')
disc_input = tflearn.input_data(shape=[None, image_dim], name='disc_input')

gen_sample = generator(gen_input)
disc_real = discriminator(disc_input)
disc_fake = discriminator(gen_sample, reuse=True)

# get loss
disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))
gen_loss = -tf.reduce_mean(tf.log(disc_fake))

# training operations
gen_vars = tflearn.get_layer_variables_by_scope('generator')
gen_model = tflearn.regression(gen_sample, placeholder=None, optimizer='adam',
                               loss=gen_loss, trainable_vars=gen_vars,
                               batch_size=64, name='target_gen', op_name='GEN')
disc_vars = tflearn.get_layer_variables_by_scope('discriminator')
disc_model = tflearn.regression(disc_real, placeholder=None, optimizer='adam',
                                loss=disc_loss, trainable_vars=disc_vars,
                                batch_size=64, name='target_disc', op_name='DISC')

# define GAN model, that output the generated images.
gan = tflearn.DNN(gen_model)

# training
# generate noise to feed to the generator
z = np.random.uniform(-1., 1., size=[total_samples, z_dim])
# start training, feed both noise and real images.
gan.fit(X_inputs={gen_input: z, disc_input: X_train},
        Y_targets=None,
        n_epoch=100)

# generate images from noise, using the generator network.
f, a = plt.subplots(2, 10, figsize=(10, 4))
for i in range(10):
    for j in range(2):
        # noise input.
        z = np.random.uniform(-1., 1., size=[1, z_dim])
        # generate image from noise. Extend to 3 channels for matplot figure.
        temp = [[ii, ii, ii] for ii in list(gan.predict([z])[0])]
        a[j][i].imshow(np.reshape(temp, (28, 28, 3)))
f.show()
plt.draw()
plt.waitforbuttonpress()
```

# Training

The internal operations of GANs are really not much more complicated than previously stated. The generator model receives random data as input and outputs samples which are initially not expected to bear any resemblance to the intended data to be produced. The discriminator model, on the other hand, is a classifier that receives input from samples of the data to be produced as well as the output of the generator and learns to discriminate between the two. These two networks proceed to iterate, with the generator model receiving feedback from the discriminator by backpropogating gradient information in order to improve it's output to more closely match the real samples, thus forcing the discriminator model to become better and better at discerning the difference between the two. This process yields a powerful network organization which has been shown to have a remarkable ability to reflect higher-order semantic logic.

Given the interdependent nature of this arrangement, training GANs can prove somewhat difficult when the sophistication of the two networks are unmatched. The generator model is dependent upon valid and relevant feedback from the discriminator in order to improve it's output, so if the discriminator is over powered the generator will be insufficiently instructed in how to improve the output and thus not train effectively. Additionally, GANs have a tendency to fail in particular scenarios wherein it is necessary to understand holistic underlying architectures, dimensional perspectives, and in feature frequency. These failures can result in such oddities as the creation of images of dogs simultaneously standing on 2 legs and 4, both front and back facing, or with 6 or 7 eyes.

# Applications

GANs can be applied to virtually any generative task, but have perhaps become best known in image generation where they have become quite useful in a number of difficult tasks such as noise reduction or removal, content-aware fill, inpainting or any other task associated with reproducing corrupted or removed portions of an image. More generally, what these tasks are accomplishing is reproducing convincing synthetic data to fill gaps in the sample. This technique can be applied more broadly in the cases of any missing data points, even within observations in a numerical dataset or text just as well as it is applied to images.

# DCGANs



- image completion in 3 steps
   	1. first interpret images as being samples from a probability distribution
  	2. interpretation lets us learn how to generate fake images
  	3. then find the best fake image for completion
- interpreting images as samples from a probability distribution
  - 2 types of information:
    1. **contextual information**: infer what missing pixels are based on information provided by surrounding pixels
    2. **perceptual information**: interpret the filled in portions as being "normal", like what you've seen in real life or from other pictures
  - without perceptual information, there are many valid completions for a context
  - something that looks "normal" to a machine learning system might not look normal to humans
  - we can interpret images as samples from a high-dimensional probability distribution
  - PDF is easy to recover for simple distributions, it's difficult and often intractable for more complex distributions over images; complexity partly comes from intricate conditional dependencies (pixel depends on the values of other pixels in the image)
- quickly generating fake images
  - in addition to **GANs**, there are other ways to train generative models with deep learning, like **variational auto encoders** 
  - DCGANs use _fractionally-strided convolutions_ to _upsample images_
    - as opposed to a standard convolutional window sliding over a standard input matrix...
    - a fractionally-strided convolution as expanding the pixels so that there are zeros in-between the pixels; the convolution over this expanded space will result in a larger output space; ie, 3x3 to 5x5 
    - many names for convolutional layers that upsample: full convolution, in-network upsampling, fractionally-strided convolution, backwards convolution, deconvolution, up convolution, or transposed convolution



# References

- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434v2.pdf)
- [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Fantastic GANs and Where to Find Them](http://guimperarnau.com/blog/2017/03/Fantastic-GANs-and-where-to-find-them)
- [Image Completion with Deep Learning in Tensorflow](http://bamos.github.io/2016/08/09/deep-completion/)
- [Enhancing images using Deep Convolutional Generative Adversarial Networks (DCGANs)](https://swarbrickjones.wordpress.com/2016/01/13/enhancing-images-using-deep-convolutional-generative-adversarial-networks-dcgans/)
- [srez](https://github.com/david-gpu/srez)