import tensorflow as tf
import glob


class TFRecordConverter:
    def __init__(self, img_path):
        self.image_paths = glob.glob(img_path)

    def convert_to_tfrecord(self, record_file: str):
        with tf.io.TFRecordWriter(record_file) as writer:
            labels = parse_labels(self.image_paths)
            for img_path, label in zip(self.image_paths, labels):
                tf_example = image_example(img_path)
                writer.write(tf_example.SerializeToString())


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def parse_labels(img_path: str):
    return img_path.split("/")[-2]

def get_feature_description():
    return {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }


def image_example(img_path):
    img = tf.compat.v1.read_file(img_path)
    image_shape = tf.io.decode_png(img, channels=3).shape
    label = parse_labels(img_path)
    image_string = open(img_path, 'rb').read()
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))
