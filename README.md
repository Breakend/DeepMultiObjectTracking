# DeepMultiObjectTracking
=======

Outline of method.

Run it through yolo-ish architecture?

so ((batch_size*seq_size), x, y),

then cut and run through convolutional layers (should use residual blocks to prevent gradient vanishing)

yolo9000

tf.reshape(batch_size, seq_len, x, y)

causal/lstm layers

then backprop the gradients through the causal layers?

Test with longer sequences. Also need pure YOLO detections.
>>>>>>> adding start of tf yolo rewrite
