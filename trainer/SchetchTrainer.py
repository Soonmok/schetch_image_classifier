import tensorflow as tf

from dataloaders.schetch_dataloader import SchetchDataloader
from models.model import get_densenet, get_resnet50, get_nasnet
from trainer.losses import top1_smooth_svm, topk_smooth_svm, top1_hard_svm, topk_hard_svm
from trainer.metric import evaluate_top_k_score, evaluate_map


class SchetchTrainer:
    def __init__(self, config, tf_record_path):
        self.config = config
        # define dataloader
        total_data_loader = SchetchDataloader(config=self.config,
                                              tfrecord_path=tf_record_path).tf_dataset
        self.dev_data_loader = total_data_loader.take(10)
        self.dev_data_loader = self.dev_data_loader.__iter__()
        self.train_data_loader = total_data_loader.skip(10)
        self.train_data_loader = self.train_data_loader.__iter__()

        # define losses and optimizer
        self.categorical_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        self.categorical_train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.train_map_accuracy = tf.keras.metrics.Mean(name='train_map_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.test_map_accuracy = tf.keras.metrics.Mean(name='test_map_accuracy')
        self.test_top_k_accuracy = tf.keras.metrics.Mean(name='test_top_k_accuracy')

        self.global_step = tf.Variable(1, name="global_step")

        # degine image descriptor
        self.define_image_descriptor()

    def define_image_descriptor(self):
        if self.config.model.name == "densenet":
            self.descriptor = get_densenet(self.config, num_classes=self.config.data.class_num)
        elif self.config.model.name == "resnet50":
            self.descriptor = get_resnet50(self.config, num_classes=self.config.data.class_num)
        elif self.config.model.name == "nasnet":
            self.descriptor = get_nasnet(self.config, num_classes=self.config.data.class_num)
        print(self.descriptor.summary())

    @tf.function
    def supervised_step(self, x_train, y_train):
        with tf.GradientTape() as senet_tape:
            y_pred = self.descriptor(x_train, training=True)
            if self.config.trainer.classifier_loss == "smooth_svm":
                categorical_loss = top1_smooth_svm(y_pred, y_train, tau=self.config.trainer.tau)
            elif self.config.trainer.classifier_loss == "smooth_top_k_svm":
                categorical_loss = topk_smooth_svm(y_pred, y_train, tau=self.config.trainer.tau,
                                                   k=self.config.trainer.top_k)
            elif self.config.trainer.classifier_loss == "hard_svm":
                categorical_loss = top1_hard_svm(y_pred, y_train)
            elif self.config.trainer.classifier_loss == "hard_top_k_svm":
                categorical_loss = topk_hard_svm(y_pred, y_train, k=self.config.trainer.top_k)
            else:
                categorical_loss = self.categorical_loss_object(y_train, y_pred)
            loss = categorical_loss

        gradient_of_model = senet_tape.gradient(loss, self.descriptor.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient_of_model, self.descriptor.trainable_variables))
        self.categorical_train_loss(categorical_loss)
        self.train_accuracy(y_train, y_pred)
        self.top_k_accuracy = evaluate_top_k_score(y_train, y_pred)
        self.map_accuracy = evaluate_map(y_train, y_pred)
        self.global_step.assign_add(1)

    def dev_step(self, x_dev, y_dev):
        y_pred = self.descriptor(x_dev, training=False)
        t_loss = self.categorical_loss_object(y_dev, y_pred)

        top_k_score = evaluate_top_k_score(y_dev, y_pred)
        map_score = evaluate_map(y_dev, y_pred)
        self.test_loss(t_loss)
        self.test_accuracy(y_dev, y_pred)
        self.test_top_k_accuracy(top_k_score)
        self.test_map_accuracy(map_score)

    def train(self):
        train_steps = 61
        dev_steps = 10
        for epoch in range(self.config.trainer.epoch):
            for idx in range(train_steps):
                x_train, y_train = next(self.train_data_loader)
                self.supervised_step(x_train, y_train)

            for idx in range(dev_steps):
                x_dev, y_dev = next(self.dev_data_loader)
                self.dev_step(x_dev, y_dev)
            test_template = 'Epoch {}, Dev_Top 1 Accuracy: {} Dev_Top k Accuracy: {} Dev_Map accuracy: {}'
            print(test_template.format(epoch + 1,
                                       self.test_accuracy.result() * 100,
                                       self.test_top_k_accuracy.result() * 100,
                                       self.test_map_accuracy.result()))
