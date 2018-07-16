# Cross-stitch-Networks-for-Multi-task-Learning

This project is a TensorFlow implementation of a Multi Task Learning described in the paper
[Cross-stitch Networks for Multi-task Learning](https://arxiv.org/abs/1604.03539).

## Dataset

The dataset used here is [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), a dataset of Zalando's article images, consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes:

|Label|Description|
|-----|-----|
|0|T-shirt/top|
|1|Trouser|
|2|Pullover|
|3|Dress|
|4|Coat|
|5|Sandal|
|6|Shirt|
|7|Sneaker|
|8|Bag|
|9|Ankle boot|

For multi task learning, I added a second label for each image, which is based on the original labels:

|Label|Original Labels|Description|
|-----|-----|-----|
|0|5, 7, 9|Shoes|
|1|3, 6, 8|For Women|
|2|0, 1, 2, 4|Other|

The network will train two classifiers together, given an image it should classify it with both labels.

 ## Network
 
 ### Without task sharing
 As a baseline, a network without cross stitch is built, which simply concat two convolutional neural networks. Each network is for one task, although they have the same structure but parameters are not shared. The final loss function is the sum of two loss functions of sub networks.
 
 This is an overview of the structure:
 ![Network structure without task sharing](https://raw.githubusercontent.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning/master/img/network_without.png)
 
 Each convolutional neural network has this architecture:
 
|Layer|Output size|filter size / stride|
|-----|-----|-----|
|conv1|28x28x32|3x3 / 1|
|pool_1|14x14x32|2x2 / 2|
|conv2|14x14x64|3x3 / 1|
|pool_2|7x7x64|2x2 / 2|
|fc_3|1024||
|output|10 or 3 depends on task||
 
 ### With Cross Stitch
 Cross Stitch is a transformation applied between layers, it describes the relationship between different tasks with a linear combination of their activations. 
 ![linear combination](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/2976605dc3b73377696537291d45f09f1ab1fbf5/3-Figure3-1.png)
 
 The network should learn the relationship by itself, in comparison with manually tuning the shared network structure, this end-to-end approach should work better.
 
 This is an overview of the structure:
 ![Network strcture with Cross Stitch](https://raw.githubusercontent.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning/master/img/network_with.png)
 
 The convolutional sub networks have the same architecture as described above. As in paper suggested the cross stitch is added after Pool layers and Fully Connected layers.
 
 ## Training
 * The input images are standardized with z-score at first. 
 * To avoid overfitting L2 regularization is used for convolution layers and fully connected layers, lambda = 1e-5. 
 * Dropout has keep_prob = 0.8
 * Batch normalization is used
 * Weights of sub networks are initialized with He initialization, weights of Cross Stitch are initialized with identity matrix (i.e no sharing between tasks)
 * Learning rate is set to 0.001 constantly
 * Trained 30 epochs with batch size = 128  
 
 ## Evaluation
 The accuracy is calculated by averaging the accuracies of two classification tasks. With cross stitch transformation it get about 2% improvement on test dataset.
 
 Orange: without sharing. Blue: with cross stitch.
 ![test accuracy](https://raw.githubusercontent.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning/master/img/acc_test.png)
 ![total loss](https://raw.githubusercontent.com/helloyide/Cross-stitch-Networks-for-Multi-task-Learning/master/img/total_loss.png)
  
 ## Result
 In general Cross Stitch improved the accuracy as expect. Although this project only trained two tasks but it can be extended to more tasks easily. 
     
 I didn't pretrain the sub networks at first like in paper suggested and I also used a different initialization strategy. A better result might be found after with more tuning. Also the new labels are created based on the original labels, i.e. two tasks are highly related, tests with more independent tasks would be better.
   
