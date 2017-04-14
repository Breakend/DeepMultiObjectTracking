import tensorflow as tf

def lrelu(x, leak=0.1, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def reorg(inputs, stride=2):
    # Not sure if this is what we want
    return tf.extract_image_patches(inputs, [1,stride,stride,1],
                                    [1,stride,stride,1], [1,1,1,1], 'VALID')
