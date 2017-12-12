Autoencoders are neural networks that attempt to learn the identity function, which is to say that it trains to output the input. The important factor here is what's happening in between the input and the output. Autoencoders typically reduce the dimensionality of the input within the network, then reconstructs the input to form as the output. The optimization of the loss function is then the evaluation of the output against the original input.

![Autoencoder architecture illustration](images/architecture_autoencoder.png)

This can be useful in that the process of training the reconstruction of the input causes the network to learn important, latent characteristicis of the data. In this way, autoencoders perform a similar function to PCA in that it reduces the dimensions of the input to the most important subset of the input. However, unlike PCA, autoencoders can handle nonlinear data. Additionally, since the autoencoder is a network architecture, autoencoders can be stacked like any other network layer to create a deep network architecture. This dimensionality reduction or feature learning is the traditional purpose of autoencoders.

Autoencoders consist of two steps: the _encoder_ and the _decoder_. The _encoder_ is the first step wherein the dimension of the input is most often reduced. This compression is performed by hidden layers which are smaller, or have fewer neurons, than the input. The _decoder_ is the second step wherein the network attempts to expand compressed data to it's original state by connecting the often reduced dimension of the hidden layer to a layer of the same size as the input. This process practically causes the input to be approximately copied to the output having been given only a portion of the input data.

# Applications

As previously stated, autoencoders perform well at dimensionality reduction of nonlinear data, but are also well suited for generative applications including reconstruction of noisy or corrupted images, colorization of black and white images, as well as increasing image resolution and filling the additional pixels with enhanced detail. These same methods can be used on other types of data beyond images in order to decrease or minimize noise within the dataset.

Generative applications of autoencoders have become quite popular recently with the exploration of the theoretical connection between such architectures and latent variable models. 

# Types of Autoencoders

The above described architecture where the hidden layer has fewer neurons than the input and output layers is referred to as an __undercomplete autoencoder__. This constraint of the hidden layer results in learning the most important subset of the dimensions of the data.

One weakness of an undercomplete autoencoder is that it can fail to reveal useful information if the hidden layer is given too much leeway, thus not sufficiently compressing the data. This can be overcome using __regularized autoencoders__ which can accomodate overcomplete architectures. Regularized autoencoders use a different loss function which are capable of optimizing additional properties beyond approximating the input such as robust compensation for noise within the data, sparse representations, or reduction of the derivative of the representation.

In addition to learning feature learning and dimension reduction, autoencoders can be used in to generate content in this same fashion by applying a __denoising autoencoder__ to fill gaps in the input to perform such tasks as removing the noise from an image or, more generally, fill gaps in corupted data. 

An alternative training method known as __recirculation__ may be used with autoencoders. This training method is generally thouhgt of as more inspired by true biological processes than back-propogation and functions by comparing the network activations of the original inputs and the reconstructed inputs.


```python
'''
decoder reconstructs input
'''
def decoder(code, n_code, phase_train):
    with tf.variable_scope("decoder"):
        with tf.variable_scope("hidden_1"):
            hidden_1 = layer(code, [n_code, n_decoder_hidden_1], [n_decoder_hidden_1], phase_train)
        
        with tf.variable_scope("hidden_2"):
            hidden_2 = layer(hidden_1, [n_decoder_hidden_1, n_decoder_hidden_2], [n_decoder_hidden_2], phase_train)
        
        with tf.variable_scope("hidden_3"):
            hidden_3 = layer(hidden_2, [n_decoder_hidden_2, n_decoder_hidden_3], [n_decoder_hidden_3], phase_train)
        
        with tf.variable_scope("output"):
            output = layer(hidden_3, [n_decoder_hidden_3, 784], [784], phase_train)
    
    return output


def layer(input, weight_shape, bias_shape, phase_train):
    weight_init = tf.random_normal_initializer(stddev=(1.0 / weight_shape[0])**0.5)
    biase_init = tf.constant_initializer(value=0)

    W = tf.get_variable("W", weight_shape, initializer=weight_init)
    b = tf.get_variable("b", bias_shape, initializer=bias_init)
    logits = tf.matmul(input, W) + b

    return tf.nn.sigmoid(layer_batch_norm(logits, weight_shape[1], phase_train))


def loss(output, x):
    with tf.variable_scope("training"):
        l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(output, x)), 1))
        train_loss = tf.reduce_mean(l2)
        train_summary_op = tf.scalar_summary("train_cost", train_cost)

        return train_loss, train_summary_op


def training(cost, global_step):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
    train_op = optimizer.minimize(cost, global_step=global_step)

    return train_op


'''
collect image summaries to compare input reconstructions
'''
def image_summary(summary_label, tensor):
    tensor_reshaped = tf.reshape(tensor, [-1, 28, 29, 1])
    return tf.image_summary(summary_label, tensor_reshaped)


'''
l2 norm evaluation
'''
def evaluate(output, x):
    with tf.variable_scope("validation"):
        in_im_op = image_summary("input_image", x)
        out_im_op = image_summary("output_image", output)
        l2 = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(output, x, name="val_diff")), 1))

        val_loss = tf.reduce_mean(l2)
        val_summary_op = tf.scalar_summary("val_cost", val_loss)

        return val_loss, in_im_op, out_im_op, val_summary_op


def layer_batch_norm(x, n_out, phase_train):
    beta_init = tf.constant_initializer(value=0.0, dtype=tf.float32)
    gamma_init = tf.constant_initializer(value=1.0, dtype=tf.float32)

    beta = tf.get_variable("beta", [n_out], initializer=beta_init)
    gamma = tf.get_variable("gamma", [n_out], initializer=gamma_init)

    batch_mean, batch_var = tf.nn.moments(x, [0], name="moments")
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    ema_apply_op = ema.apply([batch_mean, batch_var])
    ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

    def mean_var_with_update():
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)
    
    mean, var = control_flow_ops.cond(phase_train, mean_var_with_update, lambda:(ema_mean, ema_var))

    x_r = tf.reshape(x, [-1, 1, 1, n_out])
    normed = tf.nn.batch_norm_with_global_normalization(x_r, mean, var, beta, gamma, 1e-3, True)

    return tf.reshape(normed, [-1, n_out])


def run_network(n_code):
    mnist = input_data.read_data_sets("data/", one_hot=True)

    with tf.Graph().as_default():
        with tf.variable_scope("autoencoder_model"):
            # mnist data image of shape 28*28=784
            x = tf.placeholder("float", [None, 784])
            phase_train = tf.placeholder(tf.bool)
            code = encoder(x, int(n_code), phase_train)
            output = decoder(code, int(n_code), phase_train)

            cost, train_summary_op = loss(output, x)

            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = training(cost, global_step)

            eval_op, in_im_op, out_im_op, val_summary_op = evaluate(output, x)

            summary_op = tf.merge_all_summaries()
            saver = tf.train.Saver(max_to_keep=200)
            sess = tf.Session()

            train_writer = tf.train.SummaryWriter("mnist_autoencoder_hidden=" + n_code + "_logs/", graph=sess.graph)
            val_writer = tf.train.SummaryWriter("mnist_autoencoder_hidden=" + n_code + "_logs/", graph=sess.graph)

            init_op = tf.initialize_all_variables()

            sess.run(init_op)

            # training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.0
                total_batch = int(mnist.train.num_examples / batch_size)

                # loop over all batches
                for i in range(total_batch):
                    mbatch_x, mbatch_y = mnist.train.next_batch(batch_size)

                    # fit training using batch data
                    _, new_cost, train_summary = sess.run([train_op, cost, train_summary_op], feed_dict={x: mbatch_x, phase_train: True})

                    train_writer.add_summary(train_summary, sess.run(global_step))

                    # compute average loss
                    avg_cost += new_cost / total_batch
                
                # display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch: {}; cost={:.9f}".format(epoch + 1, avg_cost))

                    train_writer.add_summary(train_summary, sess.run(global_step))
                    val_images = mnist.validation.images
                    validation_loss, in_im, out_im, val_summary = sess.run([eval_op, in_im_op, out_im_op, val_summary_op], feed_dict={x: val_images, phase_train: False})

                    val_writer.add_summary(in_im, sess.run(global_step))
                    val_writer.add_summary(out_im, sess.run(global_step))
                    val_writer.add_summary(val_summary, sess.run(global_step))
                    print("Validation loss: {}".format(validation_loss))

                    saver.save(sess, "mnist_autoencoder_hidden=" + n_code + "_logs/model-checkpoint-" + '%04d' % (epoch + 1), global_step=global_step)
            
            print("Optimization finished")

            test_loss = sess.run(eval_op, feed_dict={x: mnist.test.images, phase_train: False})

            print("Test loss: {}".format(test_loss))
```