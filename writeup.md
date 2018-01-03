# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/Horace89/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 34799
* The size of the validation set is: 4410
* The size of test set is: 12630
* The shape of a traffic sign image is: (32,32,3)
* The number of unique classes/labels in the data set is: 43

#### 2. Include an exploratory visualization of the dataset.

* First, I printed 50 random images from the training set.
* Second, I printed a histogram showing the number of classes and them numbers of images of each class.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


First, I shuffled the images by using the 'shuffle' method from sklearn.utils.
Then I preprocessed the data by normalizing it with function normalize_data() in order to improve the accuracy of the trained model. I got an accuracy rate over 90% after 20 epochs, so in this project I just use one method for improving the accuracy.

I'll return to this project later and will use other methods, as suggested by Udacity: other network models, changing the dimensions of the LeNet layers, using dropout and/or L2 regularization, tuning hyperparameters, augmenting the training data by rotating, shifting images, changing colors, etc.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 1     	| 1x1 stride, VALID padding, output = 28x28x6 	|
| RELU			|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 14x14x6   |
| Convolution 2  	    | 1x1 stride, VALID padding, output = 10x10x16  |
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, output = 5x5x16    |
| Flatten				| output = 400									|
| Fully connected		| input = 400, output = 120       	            |
| RELU					|												|
| Fully connected		| input = 120, output = 84       	            |
| RELU					|												|
| Fully connected		| input = 84, output = 10       	            |




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used my personnal iMac.
I trained the model with 7 epochs, batch_size of 256, and learning rate of 0.002.


I first trained the model with 10 epochs (as shown in the course video). I noticed that after 6-7 epochs, though, the accuracy did not decrease a lot and was reaching a plateau. os I decided the training may be good enough at 7 epochs. The batch size of 256 was selected as it appears to be a good practice and managed to speed up the training compared to using 128.
I kept the learning rate of 0.001 as it was shown in the LeNet lab.

A learning rate of 0.002 was tried as it managed to speed up the learning with 7 epochs.

For the optimizer, I used softmax_cross_entropy_with_logits, tf.reduce_mean() and the AdamOptimizer. I have note tried other variants.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.955
* test set accuracy of 0.909

The well-known architecture LeNet classifier was chosen:
* it is a famous historical architecture
* it is known to be quite good on traffic sign recognition
* the test set accuracy of 90+% with just 8 epochs, without data augmentation and sophisticated regularization method confirms this hypothesis.

I did not meet unexpected difficulties, but saw quickly that training a deep learning network is highly empirical and iterative: I played a lot with changing the number of epochs and the learning rate. Training converged more slowly when a smaller learning rate was used, but a better accuracy could be achieved.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Pedestrians crossing](https://github.com/Horace89/CarND-Traffic-Sign-Classifier-Project/blob/master/test_signs/germany-road-signs-pedestrians-crossing.png)
![Slippery road](https://github.com/Horace89/CarND-Traffic-Sign-Classifier-Project/blob/master/test_signs/germany-road-signs-slippery.png)
![Wild animals crossing](https://github.com/Horace89/CarND-Traffic-Sign-Classifier-Project/blob/master/test_signs/germany-road-signs-wild-animals.png)
![Speed limit 60](https://github.com/Horace89/CarND-Traffic-Sign-Classifier-Project/blob/master/test_signs/germany-speed-limit-sign-60.png)
![End of speed limit 80](https://github.com/Horace89/CarND-Traffic-Sign-Classifier-Project/blob/master/test_signs/germany-end-of-speed-limit-80.png)


They are located in the [test_signs directory](https://github.com/Horace89/CarND-Traffic-Sign-Classifier-Project/tree/master/test_signs)

These images may be difficult to classify as they appear to not be photographs but drawings. As the training set is done on real photos, the risk is that there will be some kind of "data mismatch" between the training set and this new set.
On the other hand the quality of these pictures is good: they are not blurry, not distorted, and taken directly from the front side. So I think it will help the classification process.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit 60			| Speed limit 50     							|
| Pedestrians crossing    			| Pedestrians crossing 										|
| Wild animals crossing	      		| Wild animals crossing					 				|
| Slippery road					| Slippery road											|
| End of speed limit 80      		| End of speed limit 80    									|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

You may look at the Ipython notebook for the detailed softmax probabilities.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

The model is always very sure about its prediction, with a prob above 85%.
The only mistake it made is about the misclassification of "Speed limit 60" to "Speed limit 50". It may be due to the fact that there are not so many training examples


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
