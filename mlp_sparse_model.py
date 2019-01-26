import tensorflow as tf
import numpy as np
import sys
from utils.general import init_dir, get_logger, random_mini_batches


def neural_net(tf_x, n_layer, n_neuron, lambd):
    """
    Args:
        tf_x: input placeholder
        n_layer: number of layers of hidden layer of the neural network
        lambd: regularized parameter

    """
    # Only apply l1 regularization on the 1st layer
    # Set seed for Xavier initializer for paper replication
    layer = tf_x
    for i in range(1, n_layer+1):
        if i == 1:
            layer = tf.layers.dense(layer, n_neuron, tf.nn.relu,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(seed=1),
                                    kernel_regularizer=tf.contrib.layers.l1_regularizer(float(lambd)))
        else:
            layer = tf.layers.dense(layer, n_neuron, tf.nn.relu,
                                    kernel_initializer = tf.contrib.layers.xavier_initializer(seed=1))
    output = tf.layers.dense(layer, 1)

    return output


class MLPSparseModel(object):
    """Generic class for tf l1-sparse mlp models"""

    def __init__(self, config, dir_output):
        """
        Args:
            config: Config instance defining hyperparams
            dir_ouput: output directory (store model and log files)
        """
        self._config = config
        self._dir_output = dir_output
        tf.reset_default_graph()  # Saveguard if previous model was defined
        tf.set_random_seed(1)    # Set tensorflow seed for paper replication


    def build_train(self):
        """Builds model for training"""
        self._add_placeholders_op()
        self._add_pred_op()
        self._add_loss_op()
        self._add_train_op(self.loss)

        self.init_session()


    def build_pred(self):
        """Builds model for predicting"""
        self._add_placeholders_op()
        self._add_pred_op()
        self._add_loss_op()

        self.init_session()


    def _add_placeholders_op(self):
        """ Add placeholder attributes """
        self.X = tf.placeholder("float", [None, self._config['num_input']])
        self.Y = tf.placeholder("float", [None, 1])
        self.lr = tf.placeholder("float")  # to schedule learning rate


    def _add_pred_op(self):
        """Defines self.pred"""
        self.output = neural_net(self.X,
                                 self._config['num_layer'],
                                 self._config['num_neuron'],
                                 self._config['lambda'])


    def _add_loss_op(self):
        """Defines self.loss"""
        l2_loss = tf.losses.get_regularization_loss()
        self.loss = l2_loss + tf.losses.mean_squared_error(self.Y, self.output)


    def _add_train_op(self, loss):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize

        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads, vs     = zip(*optimizer.compute_gradients(loss))
            grads, gnorm  = tf.clip_by_global_norm(grads, 1)
            self.train_op = optimizer.apply_gradients(zip(grads, vs))


    def init_session(self):
        """Defines self.sess, self.saver and initialize the variables"""
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def train(self, X_matrix, perf_value, lr_initial):
        """Global training procedure

        Calls method self.run_epoch and saves weights if score improves.
        All the epoch-logic must be done in self.run_epoch

        Args:
            X_matrix: Input matrix
            perf_value: Performance value
            lr_initial: Initial learning rate
        """
#        l_old = 0
        lr = lr_initial
        decay = lr_initial/1000

        m = X_matrix.shape[0]
        batch_size = m
        seed = 0    # seed for minibatches
        for epoch in range(1, 2000):

            minibatch_loss = 0
            num_minibatches = int(m/batch_size)
            seed += 1
            minibatches = random_mini_batches(X_matrix, perf_value, batch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, t_l, pred = self.sess.run([self.train_op, self.loss, self.output],
                                             {self.X: X_matrix, self.Y: perf_value, self.lr: lr})
                minibatch_loss += t_l/num_minibatches

            if epoch % 500 == 0 or epoch == 1:

                rel_error = np.mean(np.abs(np.divide(perf_value.ravel() - pred.ravel(), perf_value.ravel())))
                if self._config['verbose']:
                    print("Cost function: {:.4f}", minibatch_loss)
                    print("Train relative error: {:.4f}", rel_error)

#                if np.abs(minibatch_loss-l_old)/minibatch_loss < 1e-8:
#                    break;

#            # Store the old cost function
#            l_old = minibatch_loss

            # Decay learning rate
            lr = lr*1/(1 + decay*epoch)


    def save_session(self):
        """Saves session"""
#        # check dir one last time
#        dir_model = self._dir_output + "model.weights/"
#        init_dir(dir_model)
#
#        # logging
#        sys.stdout.write("\r- Saving model...")
#        sys.stdout.flush()
#
#        # saving
#        self.saver.save(self.sess, dir_model + 'model.ckpt')
#
#        # logging
#        sys.stdout.write("\r")
#        sys.stdout.flush()
#        self.logger.info("- Saved model in {}".format(dir_model))


    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
#        self.logger.info("Reloading the latest trained model...")
#        self.saver.restore(self.sess, dir_model)


    def predict(self, X_matrix_pred):
        """Predict performance value"""

        Y_pred_val = self.sess.run(self.output, {self.X: X_matrix_pred})

        return Y_pred_val
