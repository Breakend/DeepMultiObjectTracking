from tf_helper_ops import lrelu, reorg


class YOLOv2(object):
    """
    YOLOv2
    - It takes (416, 416, 3) sized image as input
    """

    def __init__(self, config):
        self.anchors = config["anchors"]
        self.n_boxes = config["n_boxes"]
        self.n_classes = config["n_classes"]
        self.is_training = config["is_training"]
        self.reuse = config["reuse"]

    def _create_layer(self, inputs, layer_num, filter_size, kernel_size, max_pool=True, activation=True):
        # layer1
        conv1 = tf.layers.conv2d(inputs, name='conv%d'%layer_num, use_bias=False,
                                      filters=32, kernel_size=3, strides=1,
                                      padding="SAME", reuse=self.reuse)
        bn1 = tf.layers.batch_normalization(conv1, name='bn%d'%layer_num,
                                                 center=False, epsilon=2e-5,
                                                 training=self.is_training,
                                                 reuse=self.reuse)
        _return_layer = bias1 = tf.contrib.layers.bias_add(bn1, name='bias%d'%layer_num)

        if activation:
            _return_layer = lrelu1 = lrelu(self.bias1, name="lrelu%d'%layer_num)
        if max_pool:
            _return_layer = tf.layers.max_pooling_2d(lrelu1, pool_size=2,
                                                    strides=2, padding='valid',
                                                    name='maxpool%d'%layer_num)
        return _return_layer

    def _create_layers(self, inputs, scope="yolov2"):
        cur_layer = self.layer1 = self._create_layer(inputs, layer_num=1, filter_size=32, kernel_size=3)
        cur_layer = self.layer2 = self._create_layer(cur_layer, layer_num=2, filter_size=64, kernel_size=3)
        cur_layer = self.layer3 = self._create_layer(cur_layer, layer_num=3, filter_size=128, kernel_size=3, max_pool=False)
        cur_layer = self.layer4 = self._create_layer(cur_layer, layer_num=4, filter_size=64, kernel_size=1, max_pool=False)
        cur_layer = self.layer5 = self._create_layer(cur_layer, layer_num=5, filter_size=128, kernel_size=3, max_pool=False)
        cur_layer = self.layer6 = self._create_layer(cur_layer, layer_num=6, filter_size=256, kernel_size=3)
        cur_layer = self.layer7 = self._create_layer(cur_layer, layer_num=7, filter_size=128, kernel_size=1, max_pool=False)
        cur_layer = self.layer8 = self._create_layer(cur_layer, layer_num=8, filter_size=256, kernel_size=3, max_pool=False)
        cur_layer = self.layer9 = self._create_layer(cur_layer, layer_num=9, filter_size=512, kernel_size=3, max_pool=False)
        cur_layer = self.layer10 = self._create_layer(cur_layer, layer_num=10, filter_size=256, kernel_size=1)
        cur_layer = self.layer11 = self._create_layer(cur_layer, layer_num=11, filter_size=512, kernel_size=3, max_pool=False)
        cur_layer = self.layer12 = self._create_layer(cur_layer, layer_num=12, filter_size=256, kernel_size=1, max_pool=False)
        cur_layer = self.layer13 = self._create_layer(cur_layer, layer_num=13, filter_size=512, kernel_size=3, max_pool=False)
        self.high_resolution_feature = reorg(self.layer13)
        cur_layer = self.layer14 = self._create_layer(cur_layer, layer_num=14, filter_size=256, kernel_size=1, max_pool=False)
        cur_layer = self.layer15 = self._create_layer(cur_layer, layer_num=15, filter_size=512, kernel_size=3)
        cur_layer = self.layer16 = self._create_layer(cur_layer, layer_num=16, filter_size=1024, kernel_size=3, max_pool=False)
        cur_layer = self.layer17 = self._create_layer(cur_layer, layer_num=17, filter_size=512, kernel_size=1, max_pool=False)
        cur_layer = self.layer18 = self._create_layer(cur_layer, layer_num=18, filter_size=1024, kernel_size=3, max_pool=False)
        cur_layer = self.layer19 = self._create_layer(cur_layer, layer_num=19, filter_size=512, kernel_size=1, max_pool=False)
        cur_layer = self.layer20 = self._create_layer(cur_layer, layer_num=20, filter_size=1024, kernel_size=3, max_pool=False)

        ## top layers

        cur_layer = tf.concat((high_resolution_feature, cur_layer), axis=1)  # output concatnation
        cur_layer = self.layer21 = self._create_layer(cur_layer, layer_num=21, filter_size=1024, kernel_size=3, max_pool=False)
        self.layer22 = tf.layers.conv2d(cur_layer, name='conv22', use_bias=True,
                                      filters=self.n_boxes * (5+self.n_classes), kernel_size=1, strides=1,
                                      padding="SAME", reuse=self.reuse)
    def _create_last_layer_and_losses(self):
        # TODO: all the anchor stuff here



#         def model(data, is_training=False, reuse=None, scope='my_model'):
#   # Define a variable scope to contain all the variables of your model
#   with tf.variable_scope(scope, 'model', data, reuse=reuse):
#     # 1 layer
#     net = tf.contrib.layers.conv2d(data, ....)
#     ....
#     net = tf.contrib.layers.batch_norm(net, is_training)
#    return net
#
# train_outputs = model(train_data, is_training=True)
# eval_outputs = model(eval_data, is_training=False, reuse=True)
#
# eval_predictions = sess.run(eval_outputs, feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
