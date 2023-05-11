"""Parameters for classifier"""

import tensorflow as tf


class Params:
    """Parameters for classifier"""

    # base
    data_path = "./datasets/"
    train_path = data_path + "train"
    test_path = data_path + "test"
    reference_path = data_path + "reference"
    train_log_dir = f"./reports/train"
    test_log_dir = f"./reports/test"
    autotune = tf.data.experimental.AUTOTUNE
    num_classes = 4

    # train
    # autoencoder
    input_shape = (160, 160, 1)
    filters = [32, 64, 128, 50]  # last element is size of embeddings
    batch_size_train = 16
    batch_size_val_test = 1
    epochs = 10
    lr = 1e-2
    decay = 1e-6
    momentum = 0.95
    nesterov = True
    optimizer = "adam"
    load_saved_autoencoder_model = None

    # clustering
    tolerance = 0.001
    max_iter = 1000
    loss = [
        "kld",
        "mse",
    ]  # KL Divergence for clustering and Mean Squared Error for autoencoder
    loss_weights = [0.1, 1]
