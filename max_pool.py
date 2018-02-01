import tensorflow as tf
import PIL.Image as Image
tf.InteractiveSession()

#####################################################################################
max_pool_kernel = [2, 2]  # set to whatever pleases you to test the effect of
stride = 2                # set to values 1-10 to test effect
#####################################################################################

# Generate an array with random pixel values between 0 and 255
random_image_tensor = tf.random_uniform(shape=[1, 400, 400, 3], minval=0, maxval=255)

# Max Pool layer
max_pool1 = tf.layers.max_pooling2d(inputs=random_image_tensor, pool_size=max_pool_kernel, strides=stride,
                                    padding='valid', data_format='channels_last', name="max_pool1")

# Convert to uint8 as PIL requires this to show image
input_as_uint8 = random_image_tensor.eval().astype('uint8')[0]
max_pool1_as_uint8 = max_pool1.eval().astype('uint8')[0]

print "input size (before max pooling):", input_as_uint8.shape
print "input size (after max pooling:", max_pool1_as_uint8.shape

# Create PIL images from arrays
before_pool_img = Image.fromarray(input_as_uint8)
after_pool_img = Image.fromarray(max_pool1_as_uint8)

# Show images
before_pool_img.show()
after_pool_img.show()

