The folder contains the code needed to train the fmnist autoencoders, train the fmnist classifiers, and evaluate them. The train-x- code trains the individual autoencoders. There are 4 in total. There is a flag retrain which determines if the code will restart training of a model or not. The train_pgd_fmnist code trains an adversarial classifier. train_pgd_fmnist-n trains a classifier just on the raw fmnist images. The eval scripts evaluate the different classifiers. Note this code runs on TensorFlow 1.4