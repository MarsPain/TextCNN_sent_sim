import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn


class TextCNN:
    def __init__(self, config):
        # 初始化参数
        self.num_classes = config["num_classes"]
        self.sequence_length = config["sequence_length"]
        self.vocab_size = config["vocab_size"]
        self.embed_size = config["embed_size"]
        self.hidden_size = config["embed_size"]
        self.is_training = config["is_training"]
        self.learning_rate = tf.Variable(config["learning_rate"], trainable=False, name="learning_rate")  # ADD learning_rate
        decay_rate_big = 0.50
        self.learning_rate_decay_half_op = tf.assign(self.learning_rate, self.learning_rate * decay_rate_big)
        self.filter_sizes = config["filter_sizes"]  # it is a list of int. e.g. [3,4,5]
        self.num_filters = config["num_filters"]
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.num_filters_total = self.num_filters * len(config["filter_sizes"])  # 卷积核filter的数量
        self.clip_gradients = config["clip_gradients"]
        self.top_k = config["top_k"]
        self.length_data_mining_features = config["features_vector_size"]
        # 设置占位符和变量
        self.Embedding = tf.get_variable("Embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer, trainable=False)
        self.input_x1 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x1")  # sentences_1
        # print("input_x1:", self.input_x1)
        self.input_x2 = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x2")  # sentences_2
        # print("input_x2:", self.input_x2)
        self.features_vector = tf.placeholder(tf.float32, [None, self.length_data_mining_features], name="features_vector")  # features_vector
        # print("features_vector:", self.features_vector)
        self.input_y = tf.placeholder(tf.int32, [None, ], name="input_y")  # labels:[None,num_classes]
        # print("input_y:", self.input_y)
        self.weights = tf.placeholder(tf.float32, [None, ], name="weights_label")  # 标签权重
        # print("weights:", self.weights)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.iter = tf.placeholder(tf.int32)  # 记录training iteration
        self.tst = tf.placeholder(tf.bool)
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = config["decay_steps"], config["decay_rate"]

        # 构造图
        self.logits = self.inference_cnn()   # 获得预测值（one-hot向量：[batch_size, num_classes]）
        # if not is_training:
        #     return
        self.loss_val = self.loss()  # 计算loss
        self.train_op = self.train()    # 更新参数
        self.predictions = tf.argmax(self.logits, 1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.input_y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def inference_cnn(self):
        h_bluescore = tf.layers.dense(self.features_vector, self.hidden_size / 2, use_bias=True)   # features_vector
        h_bluescore = tf.nn.relu(h_bluescore)
        # cnn features from sentences_1 and sentences_2
        x1 = self.conv_layers(self.input_x1, 1)  # [None,num_filters_total]
        x2 = self.conv_layers(self.input_x2, 1, reuse_flag=True)  # [None,num_filters_total]
        h_cnn = self.additive_attention(x1, x2, self.hidden_size / 2, "cnn_attention")
        h = tf.concat([h_cnn, h_bluescore], axis=1)  # concat feature
        h = tf.layers.dense(h, self.hidden_size, activation=tf.nn.relu, use_bias=True)  # fully connected layer
        h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)
        with tf.name_scope("output"):
            logits = tf.layers.dense(h, self.num_classes, use_bias=False)
        return logits

    def conv_layers(self, input_x, name_scope, reuse_flag=False):
        embedded_words = tf.nn.embedding_lookup(self.Embedding, input_x)    # [None,sentence_length,embed_size]
        # [None,sentence_length,embed_size,1] expand dimension so meet input requirement of 2d-conv
        sentence_embeddings_expanded = tf.expand_dims(embedded_words, -1)   # 词向量可以是多通道的
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope(str(name_scope)+"convolution-pooling-%s" % filter_size, reuse=reuse_flag):
                # 1.create filter
                filters = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters], initializer=self.initializer)
                # 2.conv operation: conv2d===>computes a 2-D convolution given 4-D `input` and `filter` tensors.
                conv = tf.nn.conv2d(sentence_embeddings_expanded, filters, strides=[1, 1, 1, 1],
                                    padding="VALID", name="conv")   # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # print("conv:", conv)
                # 3. apply nolinearity
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")  # [batch_size,sequence_length - filter_size + 1,1,num_filters]
                h = tf.reshape(h, [-1, self.sequence_length - filter_size + 1, self.num_filters])  # [batch_size,sequence_length - filter_size + 1,num_filters]
                h = tf.transpose(h, [0, 2, 1])  # [batch_size,num_filters,sequence_length - filter_size + 1]
                # 4. k-max pooling
                h = tf.nn.top_k(h, k=self.top_k, name='top_k')[0]  # [batch_size,num_filters,self.k]
                h = tf.reshape(h, [-1, self.num_filters*self.top_k])  # [batch_size,num_filters*self.k]
                pooled_outputs.append(h)
        # 5. combine all pooled features, and flatten the feature.output' shape is a [1,None]
        h_pool = tf.concat(pooled_outputs, 1)  # shape:[batch_size, num_filters_total*self.k]
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total*self.top_k])  # shape should be:[None,num_filters_total]
        # print("h_pool_flat:", h_pool_flat)
        # 6. add dropout
        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h_pool_flat, keep_prob=self.dropout_keep_prob)    # [None,num_filters_total]
        return h

    def additive_attention(self, x1, x2, dimension_size, vairable_scope):
        with tf.variable_scope(vairable_scope):
            g = tf.get_variable("attention_g", initializer=tf.sqrt(1.0 / self.hidden_size))
            b = tf.get_variable("bias", shape=[dimension_size], initializer=tf.zeros_initializer)
            x1 = tf.layers.dense(x1, dimension_size)  # [batch_size,hidden_size]
            x2 = tf.layers.dense(x2, dimension_size)  # [batch_size,hidden_size]
            h = g*tf.nn.relu(x1 + x2 + b)  # [batch_size,hidden_size]
        return h

    def loss(self, l2_lambda=0.0003):
        with tf.name_scope("loss"):
            # sparse_softmax_cross_entropy
            losses = tf.losses.sparse_softmax_cross_entropy(self.input_y, self.logits, weights=self.weights)
            loss_main = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss_main+l2_losses
        return loss

    def train(self):
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op
