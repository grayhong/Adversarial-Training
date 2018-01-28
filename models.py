import tensorflow as tf

slim = tf.contrib.slim

def modelA(x, logits=False, is_training=True, scope='naive', reuse=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = slim.conv2d(x, 20, [5,5], scope='conv1')
        net = slim.max_pool2d(net, [2,2], scope='pool1')
        net = slim.conv2d(net, 50, [5,5], scope='conv2')
        net = slim.max_pool2d(net, [2,2], scope='pool2')
        net = slim.flatten(net, scope='flatten3')
        net = slim.fully_connected(net, 500, scope='fc4')
        logit = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
        y = tf.nn.softmax(logit, name='ybar')
    
    if logits:
        return y, logit
    
    return y

def modelB(x, logits=False, is_training=True, scope='naive', reuse=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = slim.dropout(x, 0.2, is_training=is_training, scope='dropout1')
        net = slim.conv2d(net, 64, [8,8], 2, padding='SAME', scope='conv1')
        net = slim.conv2d(net, 128, [6,6], 2, padding='VALID', scope='conv2')
        net = slim.conv2d(net, 128, [5,5], padding='VALID', scope='con3')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        net = slim.flatten(net, scope='flatten3')
        logit = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
        y = tf.nn.softmax(logit, name='ybar')
    
    if logits:
        return y, logit
    
    return y

def modelC(x, logits=False, is_training=True, scope='naive', reuse=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = slim.conv2d(x, 128, [3,3], scope='conv1')
        net = slim.conv2d(net, 64, [3,3], scope='conv2')
        net = slim.dropout(net, 0.25, is_training=is_training, scope='dropout2')
        net = slim.flatten(net, scope='flatten3')
        net = slim.fully_connected(net, 128, scope='fc4')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout3')
        logit = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
        y = tf.nn.softmax(logit, name='ybar')
    
    if logits:
        return y, logit
    
    return y

def modelD(x, logits=False, is_training=True, scope='naive', reuse=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = slim.flatten(x, scope='flatten1')
        net = slim.fully_connected(net, 300, scope='fc1')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
        net = slim.fully_connected(net, 300, scope='fc2')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        net = slim.fully_connected(net, 300, scope='fc3')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout3')
        net = slim.fully_connected(net, 300, scope='fc4')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout4')
        logit = slim.fully_connected(net, 10, activation_fn=None, scope='fc5')
        y = tf.nn.softmax(logit, name='ybar')
    
    if logits:
        return y, logit
    
    return y
'''

def modelA(x, logits=False, is_training=True, scope='naive', reuse=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = slim.conv2d(x, 20, [5,5], reuse=reuse, scope='conv1')
        net = slim.max_pool2d(net, [2,2], scope='pool1')
        net = slim.conv2d(net, 50, [5,5], reuse=reuse, scope='conv2')
        net = slim.max_pool2d(net, [2,2], scope='pool2')
        net = slim.flatten(net, scope='flatten3')
        net = slim.fully_connected(net, 500, reuse=reuse, scope='fc4')
        logit = slim.fully_connected(net, 10, activation_fn=None, reuse=reuse, scope='fc5')
        y = tf.nn.softmax(logit, name='ybar')
    
    if logits:
        return y, logit
    
    return y

def modelB(x, logits=False, is_training=True, scope='naive', reuse=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = slim.dropout(x, 0.2, is_training=is_training, scope='dropout1')
        net = slim.conv2d(net, 64, [8,8], 2, padding='SAME', reuse=reuse, scope='conv1')
        net = slim.conv2d(net, 128, [6,6], 2, padding='VALID', reuse=reuse, scope='conv2')
        net = slim.conv2d(net, 128, [5,5], padding='VALID', reuse=reuse, scope='con3')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        net = slim.flatten(net, scope='flatten3')
        logit = slim.fully_connected(net, 10, activation_fn=None, reuse=reuse, scope='fc5')
        y = tf.nn.softmax(logit, name='ybar')
    
    if logits:
        return y, logit
    
    return y

def modelC(x, logits=False, is_training=True, scope='naive', reuse=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = slim.conv2d(x, 128, [3,3], reuse=reuse, scope='conv1')
        net = slim.conv2d(net, 64, [3,3], reuse=reuse, scope='conv2')
        net = slim.dropout(net, 0.25, is_training=is_training, scope='dropout2')
        net = slim.flatten(net, scope='flatten3')
        net = slim.fully_connected(net, 128, reuse=reuse, scope='fc4')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout3')
        logit = slim.fully_connected(net, 10, activation_fn=None, reuse=reuse, scope='fc5')
        y = tf.nn.softmax(logit, name='ybar')
    
    if logits:
        return y, logit
    
    return y

def modelD(x, logits=False, is_training=True, scope='naive', reuse=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        net = slim.flatten(x, scope='flatten1')
        net = slim.fully_connected(net, 300, reuse=reuse, scope='fc1')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout1')
        net = slim.fully_connected(net, 300, reuse=reuse, scope='fc2')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout2')
        net = slim.fully_connected(net, 300, reuse=reuse, scope='fc3')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout3')
        net = slim.fully_connected(net, 300, reuse=reuse, scope='fc4')
        net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout4')
        logit = slim.fully_connected(net, 10, activation_fn=None, reuse=reuse, scope='fc5')
        y = tf.nn.softmax(logit, name='ybar')
    
    if logits:
        return y, logit
    
    return y
'''