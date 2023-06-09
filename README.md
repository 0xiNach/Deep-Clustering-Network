
Deep Clustering Network
==============================

Instructions
------------
1. Run `pip install -r requirements.txt` to install required python packages (Requires Python > 3.10)
2. Run `python src/run.py ` to train and test the model (Must run from root project folder where *README.md* is located)

Solution
------------
Novel deep learning based unsupervised learning method to train a classifier. Idea is to train an autoencoder model which tries to create exact same images from given images (image-to-image translation) and we can use embedding layer to extract the latent features from an image. Once we have achieved that we can use traditional KMeans clustering algorithm to create N (4 for our case because we have 4 distinct classes) clusters and group them into clusters based on the similarity of extracted embeddings from autoencoder. However, we can achieve better results by using similar approach in deep learning fashion where we create clustering layer to do the same job as KMeans.

Workflow
------------
                            Resize image to 160 x 160 and convert it to grayscale
                                                    |
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


Project Organization
------------

    ├── LICENSE
    │
    ├── README.md                         <- The top-level README for developers using this project.
    │
    ├── datasets
    │    ├── reference                    <- reference images to infer class mappings
    │    ├── test                         <- test dataset
    │    └── train                        <- train dataset
    │
    ├── reports
    │    ├── train
    │    │    ├── autoencoder_log.csv     <- training log 
    │    │    ├── model.h5                <- saved model files
    │    │    └── class_mappings.json     <- id to class mappings in JSON format
    │    │
    │    └── test
    │         └── test_cm.png             <- confusion matrix for test data
    │
    ├── src
    │    ├── data_loader.py               <- huggingface's datasets based data loader
    │    ├── model.py                     <- Model definition and helper functions to build a model
    │    ├── parameters.py                <- Class containing base parameters and hyper-parameters for training/testing
    │    ├── run.py                       <- Helper class to train/test/predict on model
    │    └── utils                        <- Utility script for augmentation and to create confusion matrix  
    │
    └── requirements.txt                  <- The requirements file for reproducing the analysis environment, e.g.
                                             generated with `pip freeze > requirements.txt`
     
