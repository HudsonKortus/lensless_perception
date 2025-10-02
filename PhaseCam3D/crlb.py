import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#WARNING: this code is heavliy inspired from chat so may have issues


def _depth_gradient(PSFs_stack, delta_w=1.0):
    f_plus = PSFs_stack[2:,:,:] #[n2,n3,...,n21]
    f_minus = PSFs_stack[:-2,:,:] #[]
    depth_grad = (f_plus - f_minus) / (2.0 * delta_w)

    centered_PSFs = PSFs_stack [1:-1,:,:]
    return depth_grad, centered_PSFs


def _lateral_gradient(PSFs_stack_centered):
    PSFs = tf.expand_dims(PSFs_stack_centered, axis=-1) # [D_mid, H, W, 1]

    y_gradient, x_gradient = tf.image.image_gradients(PSFs)

    return tf.squeeze(y_gradient, -1), tf.squeeze(x_gradient, -1)

def _flatten_channel(channel):
    return tf.reshape(channel, [tf.shape(channel)[0], -1])

def _fisher_matrix(centered_PSFs, y_gradient, x_gradient, depth_grad, beta=1e-6, ridge=1e-12):
    
    #flatten channels to make math simpler
    P   = _flatten_channel(centered_PSFs)
    Dy  = _flatten_channel(y_gradient)
    Dx  = _flatten_channel(x_gradient)
    Dz  = _flatten_channel(depth_grad)

    weight = 



def get_fisher_matrix(PSFs, wvls):
    for wavelength in range(len(wvls)): #compute for each color channel
        PSFs_stack = PSFs[:,:,:,wavelength]
        depth_grad, centered_PSFs = _depth_gradient(PSFs_stack=PSFs_stack)
        y_gradient, x_gradient = _lateral_gradient(PSFs_stack_centered=centered_PSFs)

        for pixle in PSFs
        psf_grad_y = 


