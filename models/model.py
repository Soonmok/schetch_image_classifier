import tensorflow as tf


def get_resnet50(config, num_classes):
    backbone = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)
    backbone.trainable = False
    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(config.trainer.dropout)(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax',
                              name='prediction')(x)
    model = tf.keras.models.Model(backbone.input, x, name='model')
    return model


def get_densenet(config, num_classes):
    if config.model.blocks == 121:
        backbone = tf.keras.applications.DenseNet121(weights="imagenet", include_top=False)
    elif config.model.blocks == 169:
        backbone = tf.keras.applications.DenseNet169(weights="imagenet", include_top=False)
    elif config.model.blocks == 201:
        backbone = tf.keras.applications.DenseNet201(weights="imagenet", include_top=False)
    else:
        raise ValueError("block number is not matched")
    backbone.trainable = False
    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(config.trainer.dropout)(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax',
                              name='prediction')(x)
    model = tf.keras.models.Model(backbone.input, x, name='model')
    return model


def get_inception_resnet(config, num_classes):
    backbone = tf.keras.applications.InceptionResNetV2(weights="imagenet", include_top=False)
    backbone.trainable = False
    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(config.trainer.dropout)(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax',
                              name='prediction')(x)
    model = tf.keras.models.Model(backbone.input, x, name='model')
    return model


def get_nasnet(config, num_classes):
    backbone = tf.keras.applications.NASNetLarge(weights="imagenet", include_top=False)
    backbone.trainable = False
    x = backbone.output
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dropout(config.trainer.dropout)(x)
    x = tf.keras.layers.Dense(num_classes, activation='softmax',
                              name='prediction')(x)
    model = tf.keras.models.Model(backbone.input, x, name='model')
    return model
