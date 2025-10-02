import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def depth_gradient(PSFs_stack, delta_w=1.0):
    f_plus = PSFs_stack[2:,:,:] #[n2,n3,...,n21]
    f_minus = PSFs_stack[:-2,:,:] #[]
    depth_grad = (f_plus - f_minus) / (2.0 * delta_w)

    centered_PSFs = PSFs_stack [1:-1,:,:]
    return depth_grad, centered_PSFs

def laterial_gradient(PSFs_stack_centered):
    PSFs = tf.expand_dims(PSFs_stack_centered, axis=-1) # [D_mid, H, W, 1]

    y_gradient, x_gradient = tf.image.image_gradients(PSFs)

    return tf.squeeze(y_gradient, -1), tf.squeeze(x_gradient, -1)

def fisher_matrix(centered_PSFs, y_gradient, x_gradient,depth_grad )

def get_fisher_matrix(PSFs, wvls):
    for wavelength in range(len(wvls)): #compute for each color channel
        PSFs_stack = PSFs[:,:,:,wavelength]
        depth_grad, centered_PSFs = depth_gradient(PSFs_stack=PSFs_stack)
        y_gradient, x_gradient = laterial_gradient(PSFs_stack_centered=centered_PSFs)

        for pixle in PSFs
        psf_grad_y = 


