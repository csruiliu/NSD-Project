import tensorflow as tf

channel_num = 3

class perceptron(object):
    def __init__(self, net_name, model_layer, input_h, input_w, batch_size, num_classes, opt, is_training=True):
        self.net_name = net_name
        self.model_layer_num = 3
        self.img_h = input_h
        self.img_w = input_w
        self.input_size = input_h * input_w * channel_num
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.optimzier = opt
        self.model_size = 0
        self.cost = 0

    def perceptron_layer(self, input):
        weights = tf.Variable(tf.random_normal([int(input.shape[1]), self.num_classes]))
        biases = tf.Variable(tf.random_normal([self.num_classes]))
        layer = tf.matmul(input, weights) + biases
        layer_size = (int(input.shape[1]) + 1) * self.num_classes

        return layer, layer_size

    def build(self, input):
        with tf.variable_scope(self.net_name + '_instance'):
            input_image = tf.reshape(input, [-1, self.input_size])
            layer, layer_size = self.perceptron_layer(input_image)
            self.model_size += layer_size
            for _ in range(self.model_layer_num):
                layer, layer_size = self.perceptron_layer(input = layer)
                self.model_size += layer_size

        return layer

    def compute_grads(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        
        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.optimzier == "Adam":
                    train_optimizer = tf.train.AdamOptimizer(1e-4)
                elif self.optimzier == "SGD":
                    train_optimizer = tf.train.GradientDescentOptimizer(1e-4)
                train_grads_and_vars = tf.gradients(ys=cross_entropy_cost, xs=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.net_name+'_instance'), gate_gradients=True)
                #train_grads_and_vars = train_optimizer.compute_gradients(cross_entropy_cost,  gate_gradients=train_optimizer.GATE_NONE) 
                #train_grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None] 
        return train_grads_and_vars


    def train_step(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        self.cost = cross_entropy_cost
        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.optimzier == "Adam":
                    train_optimizer = tf.train.AdamOptimizer(1e-4)
                elif self.optimzier == "SGD":
                    train_optimizer = tf.train.GradientDescentOptimizer(1e-4)
                #train_grads_and_vars = train_optimizer.compute_gradients(cross_entropy_cost, tf.trainable_variables(), gate_gradients=train_optimizer.GATE_NONE)
                train_grads_and_vars = tf.gradients(ys=cross_entropy_cost, xs=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.net_name+'_instance'), gate_gradients=True)
                #train_steps = train_optimizer.apply_gradients(train_grads_and_vars)
                train_steps = train_optimizer.apply_gradients(zip(train_grads_and_vars, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.net_name+'_instance')))

        return train_optimizer, train_grads_and_vars, train_steps

    def train(self, logits, labels):
        with tf.name_scope('loss_'+self.net_name):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
            cross_entropy_cost = tf.reduce_mean(cross_entropy)
        self.cost = cross_entropy_cost
        with tf.name_scope('optimizer_'+self.net_name):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,scope=self.net_name+'_instance')
            with tf.control_dependencies(update_ops):
                if self.optimzier == "Adam":
                    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_cost)
                elif self.optimzier == "SGD":
                    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy_cost)

        return train_step

    def getCost(self):
        return self.cost

    def getModelInstanceName(self):
        return (self.net_name + " with layer: " + str(self.model_layer_num))

    def getModelMemSize(self):
        return self.model_size * 4 / (1024**2)
