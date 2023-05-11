"""Deep clustering network"""
import glob
import json
import os
import warnings
from typing import Any

import numpy as np
import tensorflow as tf
import tqdm
from PIL import Image

from data_loader import Data
from model import DeepClustering  # pylint: disable=E0611
from parameters import Params
from utils.confusion_matrix import plot_cm


class DCN:
    def __init__(self) -> None:
        self.dcn = DeepClustering(
            input_shape=Params.input_shape,
            filters=Params.filters,
            n_clusters=Params.num_classes,
        )
        self.trained = False
        self.__class_mappings = {}
        self.__class_name_mappings = {}

    @property
    def class_names(self):
        return list(self.__class_name_mappings.values())

    def id2class(self, ind):
        if ind not in self.__class_name_mappings:
            raise IndexError("Index out of range.")
        return self.__class_name_mappings[ind]

    def train(self):
        # require two losses, one each for clustering and autoencoder
        assert len(Params.loss) == 2 and len(Params.loss_weights) == 2

        self.dcn.compile(
            loss=Params.loss,
            loss_weights=Params.loss_weights,
            optimizer=Params.optimizer,
        )
        self.dcn.model.summary()

        # load training data
        data = Data(split="train", data_path=Params.train_path)

        self.dcn.fit(
            data.train,
            data.train_steps,
            epochs=Params.epochs,
            tol=Params.tolerance,
            max_iter=Params.max_iter,
            save_dir=Params.train_log_dir,
            cae_weights=Params.load_saved_autoencoder_model,
        )
        self.trained = True
        self.infer_class_mappings()
        self.save_class_mappings()

    def __call__(self, image) -> Any:
        if not self.trained:
            warnings.warn(
                "Inferencing on non-trained model, consider training a model first or load pretrained model weights."
            )
        return self.predict(image)

    def load_weights(self, saved_model_path):
        self.dcn.load_weights(saved_model_path)
        print("Model loaded successfully.")
        self.trained = True
        self.infer_class_mappings()
        self.save_class_mappings()

    def test(self, saved_model_path=None):
        if not saved_model_path and not self.trained:
            raise ValueError(
                "Train the model first or load model weights in order to run the test"
            )

        if saved_model_path:
            self.load_weights(saved_model_path)

        # create directory if not exist
        if not os.path.exists(Params.test_log_dir):
            os.makedirs(Params.test_log_dir)

        # load test data
        data = Data(split="test", data_path=Params.test_path)

        _pred = []
        _gt = []
        for sample in tqdm.tqdm(data.test):
            pred = np.argmax(tf.squeeze(self.dcn.model.predict_on_batch(sample[0])[0]))
            gt = self.__class_mappings[tf.squeeze(sample[1]).numpy()]

            # append the results
            _pred.append(pred)
            _gt.append(gt)

        # plot confusion matrix
        plot_cm(_gt, _pred, self.__class_name_mappings.values())

    def predict(self, image: str, format_output: bool = True):
        img = Image.open(image).convert("L").resize(Params.input_shape[:2])
        img = np.expand_dims(img, 0) / 255.0
        out, _ = self.dcn.model.predict_on_batch(img)
        if not format_output:
            return np.argmax(out), np.max(out)
        if not self.__class_name_mappings:
            self.infer_class_mappings()
        return f"Class: {self.__class_name_mappings[np.argmax(out)]}, Probability: {np.max(out)}"

    def infer_class_mappings(self):
        reference_images = glob.glob1(Params.reference_path, "*")
        if not reference_images:
            raise FileNotFoundError("Reference images not found")
        for i, ref_image in enumerate(sorted(reference_images)):
            ind, _ = self.predict(
                os.path.join(Params.reference_path, ref_image), format_output=False
            )
            self.__class_mappings[i] = ind
            self.__class_name_mappings[int(ind)] = os.path.basename(ref_image).split(
                "."
            )[0]

    def save_class_mappings(self):
        if self.trained:
            with open(Params.train_log_dir + "/class_mappings.json", "w") as outfile:
                json.dump(self.__class_name_mappings, outfile, indent=3)
        else:
            raise ValueError("Train model first or load saved model weights first")


if __name__ == "__main__":
    dcn = DCN()
    dcn.train()
    dcn.test()
