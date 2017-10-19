#**Traffic Sign Recognition** 

##Writeup

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

[image1]: ./visualization1.png "Random images"
[image2]: ./visualization2.png "Histogram"
[image3]: ./before.png "Color image"
[image4]: ./grayscale.png "Grayscaling"
[image5]: ./normalized.png "Normalized"
[image6]: ./test_images/1.jpg "Slippery road"
[image7]: ./test_images/2.jpg "Speed limit (60km/h)"
[image8]: ./test_images/3.jpg "Speed limit (70km/h)"
[image9]: ./test_images/4.jpg "Stop"
[image10]: ./test_images/5.jpg "Pedestrians"
[image11]: ./test_images/6.jpg "Yield"
[image12]: ./test_images/7.jpg "Road Work"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/Iamashwanth/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
One contains a set of 25 images randomly picked from the dataset.
Second one is a histogram of label data.

![Random images][image1]
![Histogram][image2]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it would help reduce the number of input features.
Here is an example of a traffic sign image before and after grayscaling.

![Color image][image3]
![Gray scale][image4]

Then I normalized the data to center it around zero and have a std of 1.

![Normalized image][image5]

Initially I did not perform any data augmentation. But later when I was working with the test examples my model was giving poor results.
When I looked at the histogram I found out that the labels it was wrongly classifying had fewer training examples.
So I used Keras image generator to perform rotate, zoom in/out, shift horizontal/vertical operations and generate 10000 more samples.
Adding this to my pipeline resulted in a significant boost to my test image accuracy.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Scale image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x24      									
| RELU					|												
| Max pooling	      	| 2x2 stride,  outputs 5x5x24 				
| Flatten
| Dropout				|											|											
| Fully connected		| output 120        									
| RELU					|												
| Dropout				|											|											
| Fully connected		| output 84        									
| RELU					|												
| Dropout				|											|											
| Fully connected		| output 43        								
| Softmax				|         									|

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I have used adam optimizer similar to what has been used in the LenNet Lab.
After experimenting a bit with the model I have reduced the learning rate to 0.001 and increased the number of epochs to 25 to trade-off with the reduced learning rate. Other hyper parameters include dropout keepprob which I have set to 0.6 - achieved similar results with dropout between 0.5 - 0.7.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.948
* validation set accuracy of 0.966
* test set accuracy of 0.947

If an iterative approach was chosen:
* I started off with the model from Lenet Lab without any tweaks.
* It was giving validation accuracy around ~93. I increased the depth of the convolutional layers so that the model can fit more complex data.
* After doing this I saw an increase in the validation accuracy.
* Later I added dropout layers so that the model can generalize to new examples - This step did not have any negative impact on the accuracy.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10] ![alt text][image11]
![alt text][image12]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slipper road      		| Slipper road   									| 
| Speed limit (60km/h)  	| Speed limit (60km/h)  								|
| Speed limit (70km/h)  	| Speed limit (70km/h)  								|
| Stop				| Stop											|
| Pedestrians			| General Caution										|
| Yield				| Yield 										|
| Road Work			| Road Work     									|


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 86%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Most of the predictions have a very high confidence with probability close to 90%.
But Road Work test image has 55 % probablity which I belive is because of the less number of training examples.
My data augementation model can be improved so that number of traning examples for each label can meet a minimum value.
This way the recall value can be improved which I believe is the issue here with road work label.

Pedestrain image resulted in a very bad prediction - It could because it looks different from the examples in the training set.

Test image 1 Correct Label: Speed limit (70km/h)

| Probability         	|     Prediction	        		|
|:---------------------:|:---------------------------------------------:|
| 0.999999		| Speed limit (70km/h)				|
| 5.12039e-07           | Speed limit (30km/h)				|
| 1.04967e-07           | Speed limit (20km/h)				|
| 1.13572e-12           | Speed limit (50km/h)				|
| 1.11442e-13           | Speed limit (120km/h)				|

Test image 2 Correct Label: Slippery road

| Probability         	|     Prediction	        		|
|:---------------------:|:---------------------------------------------:|
| 0.999993		| Slippery road					|
| 0.3177e-06            | Wild animals crossing				|
| 0.9265e-06      	| Dangerous curve to the left			|
| 0.02424e-07           | Bicycles crossing				|
| 0.17604e-09     	| Dangerous curve to the right			|

Test image 3 Correct Label: Speed limit (60km/h)

| Probability         	|     Prediction	        		|
|:---------------------:|:---------------------------------------------:|
| 0.96438		| Speed limit (60km/h)				|
| 0.0348595     	| Speed limit (50km/h)				|
| 0.000715      	| Speed limit (80km/h)				|
| 0.56941e-05   	| Speed limit (30km/h)				|
| 0.05655e-09   	| Speed limit (20km/h)				|

Test image 4 Correct Label: Stop

| Probability         	|     Prediction	        		|
|:---------------------:|:---------------------------------------------:|
| 0.892326		| Stop						|
| 0.0334785             | Vehicles over 3.5 metric tons prohibited	|
| 0.0266952             | Speed limit (60km/h)				|
| 0.0165427             | No entry					|
| 0.00689282            | Turn right ahead				|

Test image 5 Correct Label: Road work

| Probability         	|     Prediction	        		|
|:---------------------:|:---------------------------------------------:|
| 0.554708		| Road work					|
| 0.376441              | Right-of-way at the next intersection		|
| 0.0679441             | Beware of ice/snow				|
| 0.00046721            | Pedestrians					|
| 0.000234691           | Children crossing				|

Test image 6 Correct Label: Pedestrians

| Probability         	|     Prediction	        		|
|:---------------------:|:---------------------------------------------:|
| 0.813415		| General caution				|
| 0.164054              | Road narrows on the right			|
| 0.0199954             | Pedestrians					|
| 0.00226824            | Children crossing				|
| 0.000253962           | Traffic signals				|

Test image 7 Correct Label: Yield

| Probability         	|     Prediction	        		|
|:---------------------:|:---------------------------------------------:|
| 1.0			| Yield						|
| 1.48186e-12           | No vehicles					|
| 4.64471e-13           | Speed limit (60km/h)				|
| 1.5452e-14            | Priority road					|
| 1.15605e-14           | Speed limit (70km/h)				|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


