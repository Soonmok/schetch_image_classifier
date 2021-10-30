import functools

import tensorflow as tf

from image_processors.image_processor import get_feature_description


class SchetchDataloader:
    def __init__(self, config: dict, tfrecord_path: str):
        self.tf_dataset = tf.data.TFRecordDataset(tfrecord_path)
        self.config = config
        # self.tf_dataset = self.tf_dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
        self.tf_dataset = self.tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        self.tf_dataset = self.tf_dataset.shuffle(self.config.data.shuffle_batch)
        feature_description = get_feature_description()
        self.tf_dataset = self.tf_dataset.map(
            functools.partial(self._parse_image_function, image_feature_description=feature_description),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.tf_dataset = self.tf_dataset.batch(batch_size=self.config.data.batch_size)

    def _parse_image_function(self, example_proto, image_feature_description):
        # Parse the input tf.Example proto using the dictionary above.
        image_feature = tf.io.parse_single_example(example_proto, image_feature_description)
        image = tf.io.decode_png(image_feature['image_raw'], channels=3)
        image = tf.cast(image, tf.float32)
        image = image / 127.5 - 1
        label = image_feature['label']
        image = tf.image.resize(image, size=(self.config.data.img_size,
                                             self.config.data.img_size))
        return image, label
