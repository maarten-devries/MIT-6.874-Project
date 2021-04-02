#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf

def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out + 1))
    return tf.random.uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)

class scTranslate(object):
    def __init__(self, network_architecture, transfer_fct=tf.nn.relu,
        learning_rate=0.001, learning_rate_decay=0.999,
        batch_size=100,
        lambda_nonzero=0, lambda_reconst_rna=0, lambda_reconst_atac=0,
        lambda_trans=0, lambda_latent=0,
        lambda_atac_cross=0, lambda_rna_cross=0,
        bound_atac_output=False, verbose=False,
        rna_importances=None, atac_cross_entropy=False):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = tf.Variable(learning_rate)
        self.learning_rate_decay = learning_rate_decay
        self.batch_size = batch_size
        self.lambda_nonzero = lambda_nonzero
        self.lambda_reconst_rna = lambda_reconst_rna
        self.lambda_reconst_atac = lambda_reconst_atac
        self.lambda_trans   = lambda_trans
        self.lambda_latent  = lambda_latent
        self.lambda_atac_cross = lambda_atac_cross
        self.lambda_rna_cross = lambda_rna_cross
        self.verbose = verbose
        self.bound_atac_output = bound_atac_output
        self.atac_cross_entropy = atac_cross_entropy
        if rna_importances is not None:
            self.rna_importances = rna_importances
        else:
            self.rna_importances = np.ones((network_architecture["n_input_rna"]))

        # tf Graph input
        self.x_atac = tf.compat.v1.placeholder(tf.float32,
            [None, network_architecture["n_input_atac"]])
        self.x_rna = tf.compat.v1.placeholder(tf.float32,
            [None, network_architecture["n_input_rna"]])

        # Create autoencoder network
        self._create_network()
        # Define loss function based variational upper-bound and corresponding optimizers.
        self._create_loss_optimizers()

        # Initializing the tensor flow variables
        init = tf.compat.v1.global_variables_initializer()

        # Launch the session
        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(init)
        all_variables = tf.compat.v1.get_collection_ref(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.compat.v1.train.Saver(var_list=all_variables)
    
    def _create_network(self):
        # Inialize autoencode network weights and biases.
        network_weights = self._initialize_weights(self.network_architecture)

        # ATAC Network
        self.atac_z_mean, self.atac_z_sigma_sq =             self._atac_recognition_network(network_weights["atac_weights_recog"],
                                           network_weights["atac_biases_recog"])

        # Draw one sample z from Gaussian distribution.
        # TODO: Should this be random?
        self.rna_z_mean, self.rna_z_sigma_sq =             self._rna_recognition_network(network_weights["rna_weights_recog"],
                                          network_weights["rna_biases_recog"])

        self.atac_reconst =             self._atac_generator_network(network_weights["atac_weights_gener"],
                                         network_weights["atac_biases_gener"],
                                         self.atac_z_mean)
        # RNA Network
        # Draw one sample z from Gaussian distribution.
        # TODO: Should this be random?
        self.rna_reconst =             self._rna_generator_network(network_weights["rna_weights_gener"],
                                        network_weights["rna_biases_gener"],
                                        self.rna_z_mean)
        #self.rna_recontr_mean = tf.compat.v1.Print(self.rna_recontr_mean, [self.rna_recontr_mean], "rna_recontr_mean: ")

        # Build translator network.
        #self.rna_z_hat = self._atac_trans_network(network_weights["atac_weights_trans"],
        #                                          network_weights["atac_biases_trans"])
        #self.atac_z_hat = self._rna_trans_network(network_weights["rna_weights_trans"],
        #                                          network_weights["rna_biases_trans"])
        self.rna_z_hat = self.atac_z_mean
        self.rna_reconst_from_atac = self._rna_generator_network(
            network_weights["rna_weights_gener"],
            network_weights["rna_biases_gener"],
            self.rna_z_hat)

        self.atac_z_hat = self.rna_z_mean
        self.atac_reconst_from_rna = self._atac_generator_network(
            network_weights["atac_weights_gener"],
            network_weights["atac_biases_gener"],
            self.atac_z_hat)
    
    def _initialize_weights(self, network_architecture):
        n_hidden_recog_1 = network_architecture["n_hidden_recog_1"]
        n_hidden_recog_2 = network_architecture["n_hidden_recog_2"]
        n_hidden_recog_3 = network_architecture["n_hidden_recog_3"]
        n_hidden_gener_1 = network_architecture["n_hidden_gener_1"]
        n_hidden_gener_2 = network_architecture["n_hidden_gener_2"]
        n_hidden_gener_3 = network_architecture["n_hidden_gener_3"]
        n_input_atac = network_architecture["n_input_atac"]
        n_input_rna = network_architecture["n_input_rna"]
        atac_n_z = network_architecture["atac_n_z"]
        rna_n_z = network_architecture["rna_n_z"]

        all_weights = dict()
        with tf.compat.v1.variable_scope('atac'):
            # ATAC Weights
            all_weights['atac_weights_recog'] = {
                'h1': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_input_atac, n_hidden_recog_1)), 'h1_recog not finite'),
                'h2': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2), dtype=tf.float32),
                    'h2_recog not finite'),
                'h3': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_hidden_recog_2, n_hidden_recog_3), dtype=tf.float32),
                    'h3_recog not finite'),
                'out_mean': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_hidden_recog_3, atac_n_z), dtype=tf.float32),
                    'out_mean_recog not finite'),
            }
            all_weights['atac_biases_recog'] = {
                'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
                'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
                'b3': tf.Variable(tf.zeros([n_hidden_recog_3], dtype=tf.float32)),
                'out_mean': tf.Variable(tf.zeros([atac_n_z], dtype=tf.float32)),
                #'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
            }
            all_weights['atac_weights_gener'] = {
                'h1': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(atac_n_z, n_hidden_gener_1)),
                    "h1_gener not finite"),
                'h2': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
                    "h2_gener not finite"),
                'h3': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_hidden_gener_2, n_hidden_gener_3)),
                    "h3_gener not finite"),
                'out_mean': tf.Variable(xavier_init(n_hidden_gener_3, n_input_atac))
            }
            all_weights['atac_biases_gener'] = {
                'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
                'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
                'b3': tf.Variable(tf.zeros([n_hidden_gener_3], dtype=tf.float32)),
                'out_mean': tf.Variable(tf.zeros([n_input_atac], dtype=tf.float32)),
            }

        with tf.compat.v1.variable_scope("rna"):
            # RNA Weights
            all_weights['rna_weights_recog'] = {
                'h1': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_input_rna, n_hidden_recog_1)), 'h1_recog not finite'),
                'h2': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2), dtype=tf.float32),
                    'h2_recog not finite'),
                'h3': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_hidden_recog_2, n_hidden_recog_3), dtype=tf.float32),
                    'h3_recog not finite'),
                'out_mean': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_hidden_recog_3, rna_n_z), dtype=tf.float32),
                    'out_mean_recog not finite'),
            }
            all_weights['rna_biases_recog'] = {
                'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
                'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
                'b3': tf.Variable(tf.zeros([n_hidden_recog_3], dtype=tf.float32)),
                'out_mean': tf.Variable(tf.zeros([rna_n_z], dtype=tf.float32)),
                #'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
            }
            all_weights['rna_weights_gener'] = {
                'h1': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(rna_n_z, n_hidden_gener_1)),
                    "h1_gener not finite"),
                'h2': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
                    "h2_gener not finite"),
                'h3': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(n_hidden_gener_2, n_hidden_gener_3)),
                    "h3_gener not finite"),
                'out_mean': tf.Variable(xavier_init(n_hidden_gener_3, n_input_rna))
            }
            all_weights['rna_biases_gener'] = {
                'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
                'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
                'b3': tf.Variable(tf.zeros([n_hidden_gener_3], dtype=tf.float32)),
                'out_mean': tf.Variable(tf.zeros([n_input_rna], dtype=tf.float32)),
            }

        with tf.compat.v1.variable_scope('trans'):
            all_weights['atac_weights_trans'] = {
                'out_mean': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(atac_n_z, rna_n_z)), "h1_atac_trans not finite")
            }
            all_weights['atac_biases_trans'] = {
                'out_mean': tf.Variable(tf.zeros([rna_n_z], dtype=tf.float32))
            }

            all_weights['rna_weights_trans'] = {
                'out_mean': tf.debugging.assert_all_finite(
                    tf.Variable(xavier_init(rna_n_z, atac_n_z)), "h1_rna_trans not finite")
            }
            all_weights['rna_biases_trans'] = {
                'out_mean': tf.Variable(tf.zeros([atac_n_z], dtype=tf.float32))
            }

        return all_weights

    def _recognition_network(self, weights, biases, x):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(x, weights['h1']),
                                           biases['b1']))
        layer_1_safe = tf.debugging.assert_all_finite(layer_1, "recog layer 1 not finite")
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1_safe, weights['h2']),
                                           biases['b2']))
        layer_2_safe = tf.debugging.assert_all_finite(layer_2, "recog layer 2 not finite")
        layer_3 = self.transfer_fct(tf.add(tf.matmul(layer_2_safe, weights['h3']),
                                           biases['b3']))
        layer_3_safe = tf.debugging.assert_all_finite(layer_3, "gen layer 3 not finite")
        z_mean = tf.add(tf.matmul(layer_3_safe, weights['out_mean']),
                        biases['out_mean'])
        return z_mean, tf.square(z_mean - tf.reduce_mean(z_mean, axis=0))

    def _atac_recognition_network(self, weights, biases):
        return self._recognition_network(weights, biases, self.x_atac)

    def _rna_recognition_network(self, weights, biases):
        return self._recognition_network(weights, biases, self.x_rna)

    def _generator_network(self, weights, biases, z):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto data space.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(z, weights['h1']),
                                           biases['b1']))
        layer_1_safe = tf.debugging.assert_all_finite(layer_1, "gen layer 1 not finite")
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1_safe, weights['h2']),
                                           biases['b2']))
        layer_2_safe = tf.debugging.assert_all_finite(layer_2, "gen layer 2 not finite")
        layer_3 = self.transfer_fct(tf.add(tf.matmul(layer_2_safe, weights['h3']),
                                           biases['b3']))
        layer_3_safe = tf.debugging.assert_all_finite(layer_3, "gen layer 3 not finite")
        x_reconstr_mean =                 tf.add(tf.matmul(layer_3_safe, weights['out_mean']),
                                  biases['out_mean'])
        return x_reconstr_mean

    def _atac_generator_network(self, weights, biases, z):
        if self.bound_atac_output:
            return tf.math.sigmoid(self._generator_network(weights, biases, z))
        else:
            return self._generator_network(weights, biases, z)

    def _rna_generator_network(self, weights, biases, z):
        return self._generator_network(weights, biases, z)

    def _trans_network(self, weights, biases, z1):
        return tf.add(tf.matmul(z1, weights['out_mean']),
            biases['out_mean'])

    def _atac_trans_network(self, weights, biases):
        return self._trans_network(weights, biases, self.atac_z_mean)

    def _rna_trans_network(self, weights, biases):
        return self._trans_network(weights, biases, self.rna_z_mean)

    def _create_reconst_loss(self, x, x_hat, cross_entropy=False, feat_importances=None):
        # TODO: upweight nonzeros?
        # TODO: This cross-entropy is wrong.
        if cross_entropy:
            #loss = tf.multiply(x, tf.math.exp(tf.clip_by_value(1-x_hat, 1e-5, 1.0))) \
            #     + tf.multiply((1-x), tf.math.exp(tf.clip_by_value(x_hat, 1e-5, 1.0)))
            loss = tf.multiply(x, tf.math.log(x_hat)) + tf.multiply((1-x), tf.math.log(1-x_hat))
        else:
            loss = tf.square(x - x_hat)
            #nonzero = zero*tf.compat.v1.to_float(tf.abs(x) > 1e-1)
            #zero_loss    = (1.-self.lambda_nonzero)*tf.reduce_mean(zero, axis=1)
            #nonzero_loss = self.lambda_nonzero*tf.reduce_mean(nonzero, axis=1)
        if feat_importances is not None:
            # TODO:
            return tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)
        else:
            return tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)

    def _create_latent_loss(self, mean, sigma_sq):
        # a = tf.square(1. - tf.reduce_sum(sigma_sq, 1))
        a = tf.square(1. - tf.reduce_mean(sigma_sq, 1))
        b = tf.reduce_sum(tf.square(mean), 1)
        #c = tf.square(tf.reduce_sum(sigma_sq, 1))
        #a_print = tf.compat.v1.Print(a, [a], "A: ")
        return tf.reduce_mean(0.5*(a+b), axis=0)

    def _create_trans_loss(self):
        #rna_error = tf.square(self.rna_z - self.rna_z_hat)
        #atac_error = tf.square(self.atac_z - self.atac_z_hat)
        #return tf.reduce_mean(tf.reduce_mean(rna_error, axis=1)
        #                      + tf.reduce_mean(atac_error, axis=1), axis=0)
        #return tf.reduce_mean(
        #    tf.square(self.rna_z - self.atac_z))
        return tf.norm(self.rna_z_hat - self.rna_z_mean, ord='euclidean') +             tf.norm(self.atac_z_hat  - self.atac_z_mean, ord='euclidean')

    def _create_loss_optimizers(self):
        # Loss is composed of:
        # 1) Reconstruction loss.
        # 2) Latent loss.
        # 4) Translation loss.

        self.atac_reconstr_loss = self.lambda_reconst_atac*(
            self._create_reconst_loss(self.x_atac, self.atac_reconst, self.atac_cross_entropy))
        self.atac_cross_loss = self.lambda_atac_cross*(
            self._create_reconst_loss(self.x_atac, self.atac_reconst_from_rna, self.atac_cross_entropy))
        
        self.rna_reconstr_loss  = self.lambda_reconst_rna*(
            self._create_reconst_loss(self.x_rna, self.rna_reconst, feat_importances=self.rna_importances))
        self.rna_cross_loss = self.lambda_rna_cross*(
            self._create_reconst_loss(self.x_rna, self.rna_reconst_from_atac, feat_importances=self.rna_importances))

        self.atac_latent_loss = self.lambda_latent*self._create_latent_loss(self.atac_z_mean, self.atac_z_sigma_sq)
        self.rna_latent_loss  = self.lambda_latent*self._create_latent_loss(self.rna_z_mean, self.rna_z_sigma_sq)

#         self.atac_loss    = self.atac_reconstr_loss + self.atac_cross_loss #self.atac_latent_loss
#         self.rna_loss     = self.rna_reconstr_loss + self.rna_cross_loss #self.rna_latent_loss
    
    ################################
    ###reoncstruction loss + KL loss 
        self.atac_loss    = self.atac_reconstr_loss + self.atac_latent_loss
        self.rna_loss     = self.rna_reconstr_loss + self.rna_latent_loss
    ################################
        
        self.trans_loss = self.lambda_trans*self._create_trans_loss()
        
        if self.verbose:
            self.atac_loss    = tf.compat.v1.Print(self.atac_loss, [self.atac_loss], "ATAC Loss: ")
            self.rna_loss     = tf.compat.v1.Print(self.rna_loss, [self.rna_loss], "RNA Loss: ")
            self.trans_loss = tf.compat.v1.Print(self.trans_loss, [self.trans_loss], "Trans Loss: ")

        self.matched_loss = self.atac_loss + self.rna_loss + self.trans_loss
        if self.verbose:
            self.matched_loss = tf.compat.v1.Print(self.matched_loss, [self.matched_loss], "Loss: ")

        atac_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                      "atac")
        rna_vars  = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                      "rna")
        trans_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                      "trans")

        self.atac_optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(self.atac_loss,
            var_list=atac_vars)
        self.rna_optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(self.rna_loss,
            var_list=rna_vars)
        self.matched_optimizer = tf.compat.v1.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate).minimize(self.matched_loss)


    def train_atac(self, data):
        empty_rna = np.zeros((data.shape[0], self.network_architecture["n_input_rna"]))
        opt, cost = self.sess.run((self.atac_optimizer, self.atac_loss),
                  feed_dict={self.x_atac: data, self.x_rna: empty_rna})
        return cost

    def train_rna(self, data):
        empty_atac = np.zeros((data.shape[0], self.network_architecture["n_input_atac"]))
        opt, cost = self.sess.run((self.rna_optimizer, self.rna_loss),
                  feed_dict={self.x_rna: data,
                             self.x_atac: empty_atac})
        return cost

    def train_matched(self, atac_data, rna_data, silence=False):
        opt, cost, atac_reconstr_loss, atac_latent_loss,          rna_reconstr_loss, rna_latent_loss,          atac_cross_loss, rna_cross_loss,          trans_loss = self.sess.run(
            (self.matched_optimizer, self.matched_loss,
                self.atac_reconstr_loss, self.atac_latent_loss,
                self.rna_reconstr_loss, self.rna_latent_loss,
                self.atac_cross_loss, self.rna_cross_loss,
                self.trans_loss),
                  feed_dict={self.x_atac: atac_data, self.x_rna: rna_data})
        #print(cost)
        if self.verbose and not silence:
            print("atac_reconstr_loss: ", atac_reconstr_loss)
            print("atac_latent_loss: ", atac_latent_loss)
            print("rna_reconstr_loss: ", rna_reconstr_loss)
            print("rna_latent_loss: ", rna_latent_loss)
            print("atac_cross_loss: ", atac_cross_loss)
            print("rna_cross_loss: ", rna_cross_loss)
            print("trans_loss", trans_loss)
        return cost, atac_reconstr_loss, atac_latent_loss, rna_reconstr_loss, rna_latent_loss, atac_cross_loss, rna_cross_loss, trans_loss

    def transform_atac(self, X):
        return self.sess.run(self.atac_z_mean, feed_dict={self.x_atac: X})

    def transform_rna(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.rna_z_mean, feed_dict={self.x_rna: X})

    def generate_atac(self, z_mu=None):
        """ Generate data by sampling from latent space.
        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["atac_n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.atac_reconst,
                             feed_dict={self.atac_z_mean: z_mu})

    def generate_rna(self, z_mu=None):
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["rna_n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.rna_reconst,
                             feed_dict={self.rna_z_mean: z_mu})

    def reconstruct_atac(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.atac_reconst,
                             feed_dict={self.x_atac: X})

    def reconstruct_rna(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.rna_reconst,
                             feed_dict={self.x_rna: X})

    def translate_atac(self, X):
        """ Translate ATAC-seq data into RNA-seq data. """
        return self.sess.run(self.rna_reconst_from_atac,
            feed_dict={self.x_atac: X})

    def translate_rna(self, X):
        """ Translate RNA-seq data into ATAC-seq data. """
        return self.sess.run(self.atac_reconst_from_rna,
            feed_dict={self.x_rna: X})

    def translate_atac_to_z(self, X):
        """ Translate ATAC-seq data into RNA-seq data. """
        return self.sess.run(self.rna_z_hat,
            feed_dict={self.x_atac: X})

    def translate_rna_to_z(self, X):
        """ Translate RNA-seq data into ATAC-seq data. """
        return self.sess.run(self.atac_z_hat,
            feed_dict={self.x_rna: X})


    def save_weights(self, fname):
        self.saver.save(self.sess, fname)

    def load_weights(self, fname):
        self.saver.restore(self.sess, fname)

    def get_lr(self):
        lr = self.sess.run(self.learning_rate)
        return lr
        

