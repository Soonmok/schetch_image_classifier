from configs.config import get_config_from_json
from image_processors.image_processor import TFRecordConverter
from trainer.SchetchTrainer import SchetchTrainer

if __name__ == '__main__':
    tf_record_path = '/dataset/schetch_dataset.tfrecords'
    config = get_config_from_json("config.json")
    tf_record_converter = TFRecordConverter(img_path="dataset/png/**/**.png")
    tf_record_converter.convert_to_tfrecord(record_file=tf_record_path)

    trainer = SchetchTrainer(config=config,
                             tf_record_path=tf_record_path)
