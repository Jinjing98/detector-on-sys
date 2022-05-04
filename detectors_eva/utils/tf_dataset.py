import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from detectors_eva.SuperPoint.superpoint.evaluations.evaluate import evaluate_keypoint_net_SP2
tf.config.set_visible_devices([], 'GPU')
#



def ratio_preserving_resize(image, resize):
    if resize:
        target_size = tf.convert_to_tensor(resize)
    scales = tf.to_float(tf.divide(target_size, tf.shape(image)[:2]))
    new_size = tf.to_float(tf.shape(image)[:2]) * tf.reduce_max(scales)
    image = tf.image.resize_images(image, tf.to_int32(new_size),
                                   method=tf.image.ResizeMethod.BILINEAR)
    return tf.image.resize_image_with_crop_or_pad(image, target_size[0], target_size[1])
