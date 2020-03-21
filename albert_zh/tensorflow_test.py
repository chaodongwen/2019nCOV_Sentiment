import tensorflow as tf

a = tf.constant([2])
b = tf.constant([2])
c = tf.add(a,b)

if __name__ == "__main__":
    with tf.Session() as sess:
        print(sess.run(c))

