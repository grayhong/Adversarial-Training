import tensorflow as tf
        
def fgsm(x, grad, eps=0.3, clipping=True):
    """
    FGSM attack.
    """

    # signed gradient
    normed_grad = tf.sign(grad)

    # Multiply by constant epsilon
    scaled_grad = eps * normed_grad

    # Add perturbation to original example to obtain adversarial example
    adv_x = tf.stop_gradient(x + scaled_grad)

    if clipping:
        adv_x = tf.clip_by_value(adv_x, 0, 1)
    return adv_x
