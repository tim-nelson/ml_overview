import tensorflow as tf

def scale(image, label):
    image = tf.divide(tf.cast(image, tf.float32), 255.0)
    return image, label

def flatten(image, label):
    image = tf.reshape(image, [28*28])
    return image, label

def augment(image, label):
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.squeeze(image, axis=-1)
    return image, label


# Preprocess the datasets
def preprocess_train(dataset):
    dataset = (
        dataset
        .map(augment)
        .map(scale)
        .map(flatten)
        .cache()
        .shuffle(len(dataset))
        .batch(64)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset

def preprocess_val_test(dataset):
    dataset = (
        dataset
        .map(scale)
        .map(flatten)
        .cache()
        .batch(64)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset