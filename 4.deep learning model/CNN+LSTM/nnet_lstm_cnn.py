import numpy as np
import tensorflow as tf

class Config():
    B, W, H, C = 32, 32, 32, 9
    B2, W2, H2, C2 = 32, 4, 4, 512
    train_step = 25000
    lr = 1e-4
    weight_decay = 0.005
    lstm_layers = 1
    # hidden 256(default)
    lstm_H = 512
    dense = 256


    drop_out = 0.75
    load_path = 'load/'
    save_path = 'save/'


def conv2d(input_data, out_channels, filter_size,stride, in_channels=None, name="conv2d"):
    if not in_channels:
        in_channels = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [filter_size, filter_size, in_channels, out_channels],
                initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [1, 1, 1, out_channels])
        return tf.nn.conv2d(input_data, W, [1, stride, stride, 1], "SAME") + b


def pool2d(input_data, ksize, name="pool2d"):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_data, [1, ksize, ksize, 1], [1, ksize, ksize, 1], "SAME")


def conv_relu_batch(input_data, out_channels, filter_size,stride, in_channels=None, name="crb"):
    with tf.variable_scope(name):
        a = conv2d(input_data, out_channels, filter_size, stride, in_channels)
        b = batch_normalization(a,axes=[0,1,2])
        r = tf.nn.relu(b)
        return r

def dense(input_data, H, N=None, name="dense"):
    if not N:
        N = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [32,N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b", [32,1, H])
        return tf.matmul(input_data, W, name="matmul") + b
    
def dense2(input_data, H, N=None, name="dense2"):
    if not N:
        N = input_data.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W2", [N, H], initializer=tf.contrib.layers.variance_scaling_initializer())
        b = tf.get_variable("b2", [1, H])
        return tf.matmul(input_data, W, name="matmul2") + b

def batch_normalization(input_data, axes=[0], name="batch"):
    with tf.variable_scope(name):
        mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
        return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")

class batch_norm(object):
    """Code modification of http://stackoverflow.com/a/33950177"""
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name

    def __call__(self, x, axes=[0,1,2], train=True):
        shape = x.get_shape().as_list()

        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))

                batch_mean, batch_var = tf.nn.moments(x, axes, name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)

                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var

        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)

        return normed

def lstm_net(input_data,output_data,config,keep_prob = 1,name='lstm_net'):
    with tf.variable_scope(name):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(config.lstm_H,state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.lstm_layers,state_is_tuple=True)
        state = cell.zero_state(16, tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(cell, input_data, 
                       initial_state=state, time_major=True)
        print(outputs.get_shape().as_list())
        output_final = tf.squeeze(tf.slice(outputs, [config.H2-1,0,0] , [1,-1,-1]))
        print (output_final.get_shape().as_list())
        fc1 = dense(outputs, config.dense, name="dense")
        outputs2 = tf.squeeze(dense(fc1,1,name='outputs2'))
        
        logit = tf.squeeze(dense2(outputs2,1,name='logit'))
        
        print(logit.get_shape().as_list())
        print(output_data.get_shape().as_list())
        loss = tf.nn.l2_loss(logit - output_data)

        return logit,loss,fc1

class NeuralModel():
    def __init__(self, config, name):

        self.x = tf.placeholder(tf.float32, [32, config.W, config.H, config.C], name="x")
        self.y = tf.placeholder(tf.float32, [32])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        self.conv1_1 = conv_relu_batch(self.x, 128, 3,1, name="conv1_1")
        conv1_1_d = tf.nn.dropout(self.conv1_1, self.keep_prob)
        conv1_2 = conv_relu_batch(conv1_1_d, 128, 3,2, name="conv1_2")
        conv1_2_d = tf.nn.dropout(conv1_2, self.keep_prob)

        conv2_1 = conv_relu_batch(conv1_2_d, 256, 3,1, name="conv2_1")
        conv2_1_d = tf.nn.dropout(conv2_1, self.keep_prob)
        conv2_2 = conv_relu_batch(conv2_1_d, 256, 3,2, name="conv2_2")
        conv2_2_d = tf.nn.dropout(conv2_2, self.keep_prob)

        conv3_1 = conv_relu_batch(conv2_2_d, 512, 3,1, name="conv3_1")
        conv3_1_d = tf.nn.dropout(conv3_1, self.keep_prob)
        conv3_2= conv_relu_batch(conv3_1_d, 512, 3,1, name="conv3_2")
        conv3_2_d = tf.nn.dropout(conv3_2, self.keep_prob)
        conv3_3 = conv_relu_batch(conv3_2_d, 512, 3,2, name="conv3_3")
        conv3_3_d = tf.nn.dropout(conv3_3, self.keep_prob)
        print(conv3_3_d.get_shape().as_list())
        
        '''
        dim = np.prod(conv3_3_d.get_shape().as_list()[1:])
        print dim
        
        flattened = tf.reshape(conv3_3_d, [-1, dim])
        print(flattened.get_shape())
        self.fc6 = dense(flattened, 4096, name="fc6")
        '''
        
       
        input_data = tf.reshape(conv3_3_d, [32,16,512])
        print('lstm input shape',input_data.get_shape())
        
        with tf.variable_scope('LSTM') as scope:
            self.pred,self.loss_err,self.feature = lstm_net(input_data, self.y, config, keep_prob=self.keep_prob)
        self.loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        self.loss = self.loss_err+self.loss_reg
        
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
        '''
        self.logits = tf.squeeze(dense(self.fc6, 1, name="dense"))
        # l2
        self.loss_err = tf.nn.l2_loss(self.logits - self.y)
        '''
        
        with tf.variable_scope('LSTM/lstm_net/outputs2') as scope:
            scope.reuse_variables()
            self.dense_W = tf.get_variable('W')
            self.dense_B = tf.get_variable('b')
        
        with tf.variable_scope('conv1_1/conv2d') as scope:
            scope.reuse_variables()
            self.conv_W = tf.get_variable('W')
            self.conv_B = tf.get_variable('b')
        with tf.variable_scope('LSTM/lstm_net/logit') as scope:
            scope.reuse_variables()
            self.dense_W2 = tf.get_variable('W2')
            self.dense_B2 = tf.get_variable('b2')

        with tf.variable_scope('LSTM/lstm_net/dense') as scope:
            scope.reuse_variables()
            self.dense_W3 = tf.get_variable('W')
            self.dense_B3 = tf.get_variable('b')




