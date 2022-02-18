import tensorflow as tf
import scipy.io as io
import numpy as np
import sklearn.preprocessing as pre
import os

def get_general_image(path, name, num):
    data = io.loadmat(path)
    data = data[name].astype(np.float32)
    data = np.reshape(data, [num, 28, 28, 1], order='F')
    return data
def get_general_label(path, name):
    label = io.loadmat(path)
    label = label[name]
    label = np.squeeze(label.astype(np.int32))
    return label



def get_mnist_train_data():
    train_images_path = '/Users/yoprod/Desktop/Mes Recherches/Code/evocnn-master/mnist-original.mat'
    train_label_path = '/Users/yoprod/Desktop/Mes Recherches/Code/evocnn-master/mnist-original.mat'

    #train_data = get_general_image(train_images_path, 'data', 10000)
    #train_label = get_general_label(train_label_path, 'trainY')
    train_data= io.loadmat(train_images_path)
    train_data = train_data["data"].astype(np.float32)
    #train_data = np.reshape(train_data, [10000, 28, 28, 1], order='F')
    train_label=train_data["label"].astype(np.int32)

    return train_data, train_label


def get_mnist_test_data():
    test_images_path = '/Users/yoprod/Desktop/Mes Recherches/Code/evocnn-master/mnist-original.mat'
    test_label_path = '/Users/yoprod/Desktop/Mes Recherches/Code/evocnn-master/mnist-original.mat'


    test_data = get_general_image(test_images_path, 'testX', 50000)
    test_label = get_general_label(test_label_path, 'testY')

    return test_data, test_label


def get_mnist_validate_data():
    validate_images_path = '/Users/yoprod/Desktop/Mes Recherches/Code/evocnn-master/mnist-original.mat'
    validate_label_path = '/Users/yoprod/Desktop/Mes Recherches/Code/evocnn-master/mnist-original.mat'

    validate_data = get_general_image(validate_images_path, 'testX', 2000)
    validate_label = get_general_label(validate_label_path, 'testY')

    return  validate_data, validate_label


##def get_standard_train_data(name):
##    data_path = '/am/lido/home/yanan/training_data/back-{}/train_images.npy'.format(name)
##    label_path = '/am/lido/home/yanan/training_data/back-{}/train_label.npy'.format(name)
##    data = np.load(data_path)
##    label = np.load(label_path)
 ##   return data, label

##def get_standard_validate_data(name):
#    data_path = '/am/lido/home/yanan/training_data/back-{}/validate_images.npy'.format(name)
#    label_path = '/am/lido/home/yanan/training_data/back-{}/validate_label.npy'.format(name)
#    data = np.load(data_path)
#    label = np.load(label_path)
#    return data, label

#def get_standard_test_data(name):
#    data_path = '/am/lido/home/yanan/training_data/back-{}/test_images.npy'.format(name)
#    label_path = '/am/lido/home/yanan/training_data/back-{}/test_label.npy'.format(name)
#    data = np.load(data_path)
#    label = np.load(label_path)
#    return data, label



def get_train_data(batch_size):

    num_classes = 10
    (t_image, t_label), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    assert t_image.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert t_label.shape == (60000,)
    assert y_test.shape == (10000,)
    #train_image = tf.cast(t_image, tf.float32)
    #train_label = tf.cast(t_label, tf.int32)
    #single_image, single_label  = tf.train.slice_input_producer([train_image, train_label], shuffle=True)
    #single_image = tf.image.per_image_standardization(single_image)
    t_image = t_image.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
    t_image = np.expand_dims(t_image, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", t_image.shape)
    print("x_label shape:", t_label.shape)

    print(t_image.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    print("x_label shape:",len(t_label.shape))

# convert class vectors to binary class matrices
    #t_label = tf.keras.utils.to_categorical(t_label, num_classes)
    #y_test = tf.keras.utils.to_categorical(y_test, num_classes)
    #t_image = np.reshape(t_image, [10000, 28, 28, 1], order='F')
    t_image=t_image[0:10000]
    #t_image=t_image[0:10000]
    t_label=t_label[0:10000]
    #t_label = np.reshape(t_label, [10000])
    print("x_label shape:",len(t_label.shape))

    #t_image, t_label = tf.train.batch([t_image, t_label], batch_size=batch_size, num_threads=2, capacity=batch_size*3)
    return t_image,t_label

def get_validate_data(batch_size):

    num_classes=10
    (x_train, y_train), (t_image, t_label) = tf.keras.datasets.mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert t_image.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert t_label.shape == (10000,)
    #validate_image = tf.cast(t_image, tf.float32)
    #validate_label = tf.cast(t_label, tf.int32)
    #single_image, single_label  = tf.train.slice_input_producer([validate_image, validate_label], shuffle=False)
    #single_image = tf.image.per_image_standardization(single_image)
    t_image = t_image.astype("float32") / 255
    x_train = x_train.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
    t_image = np.expand_dims(t_image, -1)
    x_train = np.expand_dims(x_train, -1)
    print("x_train shape:", t_image.shape)
    print("x_label shape:",len(t_label.shape))

    print(t_image.shape[0], "train samples")
    print(x_train.shape[0], "test samples")


# convert class vectors to binary class matrices
    #t_label = tf.keras.utils.to_categorical(t_label, num_classes)
    #y_train = tf.keras.utils.to_categorical(y_train, num_classes)

     #t_image = np.reshape(t_image, [10000, 28, 28, 1], order='F')
   # t_image.shape=[10000, 28, 28, 1]
    t_image=t_image[0:2000]
    t_label=t_label[0:2000]
    
    #t_label = np.reshape(t_label, [2000])
   # t_image, t_label = tf.train.batch([t_image, t_label], batch_size=batch_size, num_threads=2, capacity=batch_size*3)
    return t_image,t_label


def get_test_data(batch_size):
    t_image, t_label = get_mnist_test_data()
    test_image = tf.cast(t_image, tf.float32)
    test_label = tf.cast(t_label, tf.int32)
    single_image, single_label  = tf.train.slice_input_producer([test_image, test_label], shuffle=False)
    single_image = tf.image.per_image_standardization(single_image)
    image_batch, label_batch = tf.train.batch([single_image, single_label], batch_size=batch_size, num_threads=2, capacity=batch_size*3)
    return image_batch, label_batch

def tf_standalized(data):
    image = tf.placeholder(tf.float32, shape=[28,28,1])
    scale_data = tf.image.per_image_standardization(image)
    data_list = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        data_length = data.shape[0]
        for i in range(data_length):
            standard_data = sess.run(scale_data, {image:data[i]})
            print(i, data_length)
            data_list.append(standard_data)
    return np.array(data_list)


#if __name__ =='__main__':
#    name = 'random'
#    data, label = get_standard_test_data(name)
#    print(data.shape, label.shape, data.dtype, label.dtype)








# def get_mnist_train_batch(batch_size, capacity=1000):
#     mnist = input_data.read_data_sets('MNIST_data', reshape=False, one_hot=True)
#     train_images = tf.cast(mnist.train.images, tf.float32)
#     train_labels = tf.cast(mnist.train.labels, tf.int32)
#     input_queue = tf.train.slice_input_producer([train_images, train_labels],shuffle=True, capacity=capacity, name='input_queue')
#     images_batch, labels_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=3, capacity=capacity)
#     return images_batch, labels_batch
# batch_images, batch_labels = get_mnist_train_batch(100)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess, coord)
#         i = 0
#         try:
#             while not coord.should_stop():
#
#                 batch_images_v, batch_labels_v = sess.run([batch_images, batch_labels])
#                 print(i)
#                 i += 1
#
#         except tf.errors.OutOfRangeError:
#             print('done')
#         finally:
#             coord.request_stop()
#             coord.join(threads)