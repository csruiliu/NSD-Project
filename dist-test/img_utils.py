import tensorflow as tf
import cv2
import os
import pickle
import numpy as np

def unpickle(path, file_name):
    unpick_data = os.path.join(path, file_name)
    with open(unpick_data, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data

def convert_image(raw_images, img_w, img_h, num_channels):
    raw_float = np.array(raw_images, dtype=float) / 255.0
    return raw_float

def load_data(path, file_name, img_w, img_h, num_channels):
    data = unpickle(path, file_name)
    raw_images = data[b'data']
    labels = np.array(data[b'labels'])
    images = convert_image(raw_images, img_w, img_h, num_channels)
    return images, labels

def one_hot_encoded(class_numbers, num_classes=None):
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1
    return np.eye(num_classes, dtype=float)[class_numbers]

def load_cifar_training(path, num_train_batch, img_w, img_h, num_channels, num_classes):
    images = np.zeros(shape=[num_train_batch, img_w, img_h, num_channels], dtype=float)
    labels = np.zeros(shape=[num_train_batch], dtype=int)
    begin = 0
    for i in range(num_train_batch):
        images_batch, cls_batch = load_data(path, "data_batch_" + str(i + 1), img_w, img_h, num_channels)
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        labels[begin:end] = cls_batch
        begin = end
    return images, labels, one_hot_encoded(class_numbers=labels, num_classes=num_classes)

def load_cifar_evaluation(path, img_w, img_h, num_channels, num_classes):
    images, labels = load_data(path, "test_batch", img_w, img_h, num_channels)
    print(images.shape)
    return images, labels, one_hot_encoded(class_numbers=labels, num_classes=num_classes)


def load_images(path, img_w, img_h):
    images = tf.gfile.ListDirectory(path)
    imgs_num = len(images)
    img_list = []
    for idx, img in enumerate(images):
        im = cv2.imread(path+"/"+img, cv2.IMREAD_COLOR)
        res = cv2.resize(im, dsize=(img_w, img_h))
        res_exp = np.expand_dims(res, axis=0)
        img_list.append(res_exp)
    img_data = np.concatenate(img_list, axis=0)
    return img_data


def load_images_tf(path, num_channels, img_w, img_h):
    images = tf.gfile.ListDirectory(path)
    imgs_num = len(images)
    img_data_list = []
    for idx, img in enumerate(images):
        image_raw_data = tf.gfile.GFile(path+"/"+img,'rb').read()
        img_data = tf.image.decode_jpeg(image_raw_data, channels = num_channels)
        img_data_float = tf.image.convert_image_dtype(img_data, dtype = tf.float32)
        img_data_resize = tf.image.resize_images(img_data_float, (img_w, img_h))
        img_data_exp = tf.expand_dims(img_data_resize, 0)
        img_data_list.append(img_data_exp)
    img_data_batch = tf.concat(img_data_list, 0)
    print(img_data_batch.shape)
    return img_data_batch

def load_images_bin(path, num_channels, img_w, img_h):
    with open(path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    raw_images = data['image']
    raw_float = np.array(raw_images, dtype=float) / 255.0
    #raw_float = raw_float.transpose([0, 3, 1, 2])
    return raw_float

def load_labels_onehot(path, num_classes):
    lines = open(path).readlines()
    labels_array = np.zeros((len(lines), num_classes))
    for idx, val in enumerate(lines):
        hot = int(val.rstrip('\n'))
        labels_array[idx, hot-1] = 1
    return labels_array

def parse_image(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    image_converted = tf.cast(image_decoded, tf.float32)
    image_scaled = tf.image.resize_image_with_crop_or_pad(image_converted, 224, 224)
    return image_scaled, label

def generate_image_dataset(image_dir, label_path):
    images_list = sorted(tf.gfile.ListDirectory(image_dir))
    images = [image_dir + '/' +s for s in images_list]
    labels = [int(line.rstrip('\n')) for line in open(label_path)]
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(parse_image)
    dataset = dataset.batch(batch_size=32)
    dataset_it = dataset.make_one_shot_iterator()
    return dataset_it

def load_image_dir(image_dir, batch_list, img_h, img_w):
    img_list = []
    for img in batch_list:
        #print(image_dir+"/"+img)
        im = cv2.imread(image_dir+"/"+img, cv2.IMREAD_COLOR)
        res = cv2.resize(im, dsize=(img_w, img_h))
        res_exp = np.expand_dims(res, axis=0)
        img_list.append(res_exp)
    img_data = np.concatenate(img_list, axis=0)
    return img_data