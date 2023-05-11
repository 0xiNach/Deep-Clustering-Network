"""Deep clustering network
This is a novel deep learning based unsupervised learning method that learns latent features from an image
and cluster them based on the similarity.
Idea behind this algorithm is, first we learn latent features from an image using autoencoder and by
using those latent features to create clusters. Traditional way to cluster any latent space is to use KMeans clustering
but by using Deep Clustering we can achieve higher performance in dividing each clusters.
Workflow:
        Train autoencoder (image-to-image translation where input image and output image is the same, idea is to
        learn latent features from an image)
                                                    |
                        Use encoder to extract features from an image (returns fixed size embeddings)
                                                    |
                        create N cluster and initialize with them random centroid
                                                    |
                        soft placement of given sample to corresponding cluster based on similarity
                            (Student`s t-distribution is used as a kernel to measure the
                             similarity between embedded point and centroid.)
                             NOTE: similarity can be interpreted as the probability
                                    of assigning sample i to cluster j
                                                    |
                            compute auxiliary target distribution from soft placement
                                                    |
                compute KL divergence loss between the soft assignments and the auxiliary distribution
                                                    |
                            update model parameters and cluster centroid from computed loss
"""
import os
from time import time

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.cluster import KMeans
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    InputSpec,
    Layer,
    Reshape,
)
from tensorflow.keras.models import Model, Sequential


def clone_model(model):
    """Create a copy of given model
    Arguments:
        Keras model
    Returns:
        Copy of given model with same weights
    """
    model_copy = tf.keras.models.clone_model(model)
    model_copy.build(model.input_shape)
    model_copy.set_weights(model.get_weights())
    return model_copy


def autoencoder(
    input_shape: tuple[int] = (160, 160, 1), filters: list[int] = [32, 64, 128, 10]
):
    """
    Creates convolution based autoencoder network from given input shape and different convolution filter sizes
    Arguments:
        input_shape: shape of an input image size
            if last element is 1, it means input image is Grayscale
            if last element is 3, it means input image is RGB
        filters: size of convolution output filters

    """
    # model layers
    model = Sequential()
    if input_shape[0] % 8 == 0:
        pad3 = "same"
    else:
        pad3 = "valid"

    # encoder
    model.add(
        Conv2D(
            filters[0],
            5,
            strides=2,
            padding="same",
            activation="relu",
            name="conv1",
            input_shape=input_shape,
        )
    )
    model.add(
        Conv2D(
            filters[1], 5, strides=2, padding="same", activation="relu", name="conv2"
        )
    )
    model.add(
        Conv2D(filters[2], 3, strides=2, padding=pad3, activation="relu", name="conv3")
    )

    # embedding
    model.add(Flatten())
    model.add(Dense(units=filters[3], name="embedding"))

    # decoder
    model.add(
        Dense(
            units=filters[2] * int(input_shape[0] / 8) * int(input_shape[0] / 8),
            activation="relu",
        )
    )
    model.add(Reshape((int(input_shape[0] / 8), int(input_shape[0] / 8), filters[2])))
    model.add(
        Conv2DTranspose(
            filters[1], 3, strides=2, padding=pad3, activation="relu", name="deconv3"
        )
    )
    model.add(
        Conv2DTranspose(
            filters[0], 5, strides=2, padding="same", activation="relu", name="deconv2"
        )
    )
    model.add(
        Conv2DTranspose(input_shape[2], 5, strides=2, padding="same", name="deconv1")
    )

    return model


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(
        self, n_clusters: int, weights: np.ndarray = None, alpha: float = 1.0, **kwargs
    ):
        if "input_shape" not in kwargs and "input_dim" in kwargs:
            kwargs["input_shape"] = (kwargs.pop("input_dim"),)
        super().__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape: tuple[int]):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(
            shape=(self.n_clusters, input_dim),
            initializer="glorot_uniform",
            name="clusters",
        )
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (
            1.0
            + (
                K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2)
                / self.alpha
            )
        )
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {"n_clusters": self.n_clusters}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeepClustering:
    """Novel deep learning based clustering algorithm"""

    def __init__(
        self,
        input_shape: tuple[int],
        filters: list[int] = [32, 64, 128, 50],
        n_clusters: int = 10,
        alpha: float = 1.0,
    ):
        """
        Arguments:
            input_shape: model input image size
            filters: size of convolution filters
            n_clusters: number of clusters required to map given data
            alpha: degrees of freedom of student's t-distribution
        """

        super().__init__()

        self.n_clusters = n_clusters
        self.input_shape = input_shape
        self.alpha = alpha
        self.pretrained = False
        self.y_pred = []

        # Define autoencoder model
        self.cae = autoencoder(input_shape, filters)
        hidden = self.cae.get_layer(name="embedding").output
        self.encoder = Model(inputs=self.cae.input, outputs=hidden)

        # Define DeepClustering model
        clustering_layer = ClusteringLayer(self.n_clusters, name="clustering")(hidden)
        self.model = Model(
            inputs=self.cae.input, outputs=[clustering_layer, self.cae.output]
        )

    def pretrain(
        self,
        data,
        train_steps: int,
        batch_size: int = 16,
        epochs: int = 10,
        optimizer: str = "adam",
        save_dir: str = "./reports/train",
    ):
        """train autoencoder
        NOTE: before optimizing clustering layer, we must train autoencoder network (image-to-image) first in order to
        learn latent feature embedding from an image
        Arguments:
            data: TensorFlow Dataset generator
            train_steps: number of iteration in one epoch (total_num_images/batch_size)
            batch_size: batch size
            epoch: number of epochs to train the model
            optimizer: name of an optimizer
            save_dir: path to directory to save logs
        """
        print("...Pretraining...")

        # compile the model with given optimizer and with Mean Squared Loss
        self.cae.compile(optimizer=optimizer, loss="mse")
        # callback function to save model training logs
        csv_logger = CSVLogger(save_dir + "/pretrain_log.csv")

        # begin training
        t0 = time()
        self.cae.fit(
            data,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=train_steps,
            callbacks=[csv_logger],
        )
        print("Pretraining time: ", time() - t0)
        self.cae.save(save_dir + "/pretrain_cae_model.h5")
        print(f"Pretrained weights are saved to {save_dir}/pretrain_cae_model.h5")
        self.pretrained = True

    def load_weights(self, weights_path: str):
        """Load saved model weights
        Arguments:
            weights_path: path to saved model weight file
        """
        self.model.load_weights(weights_path)
        self.pretrained = True

    def extract_feature(self, x: np.ndarray):
        """extract features from before clustering layer
        Arguments:
            x: input image array
        Returns:
            latent feature embeddings
        """
        return self.encoder.predict(x)

    def predict(self, x: np.ndarray):
        """Run infernece on model from given image
        Arguments:
            x: input image
        Returns:
            returns predicted class index
        """
        q, _ = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        """Compute auxiliary target distribution"""
        weight = q**2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(
        self,
        loss: list[str] = ["kld", "mse"],
        loss_weights: list[int] = [1, 1],
        optimizer: str = "adam",
    ):
        """Helper method to compile model"""
        self.model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer)

    def fit(
        self,
        data,
        train_steps: int,
        epochs: int = 10,
        batch_size: int = 16,
        tol: float = 1e-3,
        max_iter: int = 3000,
        cae_weights: str = None,
        save_dir: str = "./reports/train",
    ):
        """Start training model using tensorflow dataset generator
        Arguments:
            data: Tensorflow dataset generator
            train_steps: number of iteration in one epoch (total_num_images/batch_size)
            batch_size: batch size
            tol: tolerance, threshold to stop training
            max_iter: Maximum iteration, threshold to stop training
            cae_weights: saved model weights of autoencoder
            save_dir: path to directory to save logs
        """
        # create directory if not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        num_iterations = train_steps  # total_images/batch_size
        update_interval = num_iterations // 2
        print("Update interval", update_interval)
        save_interval = num_iterations * 8
        print("Save interval", save_interval)

        # Step 1: pretrain if necessary
        t0 = time()
        # start training autoencoder
        if not self.pretrained and cae_weights is None:
            print("...pretraining CAE using default hyper-parameters:")
            self.pretrain(
                data, train_steps, batch_size, epochs=epochs, save_dir=save_dir
            )
            self.pretrained = True
        elif cae_weights is not None:
            # if cae_weights defined, load model weights
            self.cae.load_weights(cae_weights)
            print("cae_weights is loaded successfully.")

        # Step 2: initialize cluster centers using k-means
        t1 = time()
        print("Initializing cluster centers with k-means.")
        # create clusters
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        # fit KMeans on latent feature embeddings
        self.y_pred = kmeans.fit_predict(self.encoder.predict(data, steps=train_steps))
        y_pred_last = np.copy(self.y_pred)
        # use computed centers values to set the weights of ClusteringLayer
        self.model.get_layer(name="clustering").set_weights([kmeans.cluster_centers_])

        # Step 3: deep clustering
        t2 = time()
        ite = 0
        print("training clustering.....")
        for x, _ in data:  # run infinitely
            if ite % update_interval == 0:
                print(f"Iterations finished: {ite}")
                # clone model
                model_clone = clone_model(self.model)
                q, _ = model_clone.predict(data, steps=train_steps, verbose=0)
                p = self.target_distribution(
                    q
                )  # update the auxiliary target distribution p

                # evaluate the clustering performance
                self.y_pred = q.argmax(1)

                # check stop criterion
                # calculate the difference between previous
                delta_label = (
                    np.sum(self.y_pred != y_pred_last).astype(np.float32)
                    / self.y_pred.shape[0]
                )
                y_pred_last = np.copy(self.y_pred)
                if (ite > 0 and delta_label < tol) or ite > max_iter:
                    print("delta_label ", delta_label, "< tol ", tol)
                    print("Reached tolerance threshold. Stopping training.")
                    break

            batched_q, _ = model_clone.predict_on_batch(x)
            batched_p = self.target_distribution(batched_q)
            # calculate loss
            loss = self.model.train_on_batch(x=x, y=[batched_p, x])

            # save intermediate model
            if ite % save_interval == 0:
                # save DeepClustering model checkpoints
                print("saving model to:", save_dir + "/dcec_model_" + str(ite) + ".h5")
                self.model.save_weights(save_dir + "/dcec_model_" + str(ite) + ".h5")

            ite += 1

        # save the trained model
        print("saving model to:", save_dir + "/dcec_model_final.h5")
        self.model.save_weights(save_dir + "/dcec_model_final.h5")
        t3 = time()
        print("Pretrain time:  ", t1 - t0)
        print("Clustering time:", t3 - t1)
        print("Total time:     ", t3 - t0)
