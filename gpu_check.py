import tensorflow as tf

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
hello_world = tf.constant("Hello, world!")
print (sess.run(hello_world))
