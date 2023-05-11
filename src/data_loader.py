"""Helper script to load dataset"""
import numpy as np
import tensorflow as tf

from datasets import load_dataset
from parameters import Params  # pylint: disable=E0401
from utils.data_augmentation import image_classification


def preprocess_image(x):
    """Normalize image pixels"""
    return x / 255.0


class Data:
    """handles data preprocessing and data loading"""

    def __init__(self, split, data_path=Params.train_path) -> None:
        self.split = split
        self.data_path = data_path
        self._dataset = load_dataset("imagefolder", data_dir=self.data_path)
        self._train_transform = image_classification()
        self._val_test_transform = image_classification(train=False)
        self._transform_data()

        assert self.split in {
            "train",
            "test",
        }, "Split type should be either test or train"

    def _transform_data(self):
        """Transform data"""
        if self.split == "train":
            self.train = self._dataset["train"]
            self.train.set_transform(self.preprocess_train)
            self.train = self.train.to_tf_dataset(
                columns="image",
                label_cols="image",
                batch_size=Params.batch_size_train,
                shuffle=True,
            )
            # train steps = total #of images in training set / batch_size_train
            self.train_steps = tf.data.experimental.cardinality(self.train).numpy()
            self.train = self.train.repeat()
        else:
            self.test = self._dataset["train"]
            self.test.set_transform(self.preprocess_val_test)
            self.test = self.test.to_tf_dataset(
                columns="image",
                label_cols="label",
                batch_size=Params.batch_size_val_test,
                shuffle=True,
            )

    def preprocess_train(self, examples):
        """preprocess training data"""
        examples["image"] = [
            preprocess_image(
                self._train_transform(image=np.array(image.convert("L")))["image"]
            )
            for image in examples["image"]
        ]
        return examples

    def preprocess_val_test(self, examples):
        """preprocess vaidation/test data"""
        examples["image"] = [
            preprocess_image(
                self._val_test_transform(image=np.array(image.convert("L")))["image"]
            )
            for image in examples["image"]
        ]
        return examples
