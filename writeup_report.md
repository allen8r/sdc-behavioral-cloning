# **Behavioral Cloning** 

## Writeup Report

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia_model]: ./visuals/nvidia-architecture.png "Model Visualization"
[center_lane_driving]: ./visuals/center_lane_driving.jpg "Center Lane Driving"
[recovery_0_off_right]: ./visuals/recovery_0_off_right.jpg "Recovery - Off right"
[recovery_1_steering_left]: ./visuals/recovery_1_steering_left.jpg "Recovery - Steering left"
[recovery_2_back_on_track]: ./visuals/recovery_2_back_on_track.jpg "Recovery - Back on track"
[normal]: ./visuals/normal.jpg "Normal Image"
[flipped]: ./visuals/flipped.jpg "Flipped Image"
[angle_histogram]: ./visuals/steering-angle-histogram.png "Steering angle distribution"
[loss_history]: ./visuals/loss_history.png "Loss history"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results
* `video.mp4` recroding of the car driving in autonomous mode around the track


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Based on the Nvidia self-driving car model architecture my model starts with a normalization layer (model.py, line 93), followed by 5 convolutional layers (model.py, lines 96-100), and ends with 3 fully connected layers (model.py, lines 103-108). The model includes ReLu activations in the convolutional layers to introduce non-linearity.

#### 2. Attempts to reduce overfitting in the model

Between each of the fully connected layers are Dropout layers (model.py, lines 104, 106, 108) to help with minimizing overfitting during training.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The data set comprised of recorded center lane driving for two laps, a lap of recovery driving from the sides of the track, and a lap of driving smoothly around the curves.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 113). The Adam optimizer from Keras had the following default parameter settings: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovery from the left and right sides of the road, and driving smoothly around curves.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple model to make sure the car responds to changes in imagery input. Once verified that the simple model can be tested in the simulator, refine the model to make the car drive itself completely around the track.

My first step was to use a convolution neural network model based on the LeNet architecture. I thought this model might be appropriate because it is a classic architecture that is simple yet very good at learning from input images. When this architecture was stablized, I then updated the model to use the Nvidia self-driving car architecture.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To combat the overfitting, I modified the model so that it included Dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track and plunged right into the water. To improve the driving behavior in these cases, I collected more data samples by recording more manual driving runs. In particular, recording one full track of recovery from going off the edges really helped the car stay on the track. Also, to help stablize the car from constantly weaving left and right, I ended up adding the steering angle correction factor to the steering angle only of the steering angle was non-zero. This one code fix resulted an enormous improvement for when the car was on straightaways.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is based on the Nvidia self-driving car model architecture. My model starts with a normalization layer (model.py, line 93), followed by three 5x5 convolutional layers (model.py, lines 96-98), another two 3x3 convolutional layers (model.py, lines 99-100), and finally ends with three fully connected layers (model.py, lines 103, 105, 107). The model includes ReLu activations in the convolutional layers to introduce non-linearity.

![alt text][nvidia_model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center_lane_driving]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer back to center when it approaches the edges of the track. These images show what a recovery looks like starting from going off the right edge, steering back left towards center, and ending back near center of the track:

![alt text][recovery_0_off_right]

![alt text][recovery_1_steering_left]

![alt text][recovery_2_back_on_track]

To augment the data set, I also flipped the images and the angles thinking that this would be an easy way to double the data set by creating mirror images for the model to learn from. For example, here is a normal image(on the left) that has then been flipped(on the right):

![alt text][normal] ![alt text][flipped]

After the collection process, I had 28,392 number of data points (9,464 x 3 [center, left, right images]). I then preprocessed this data by shuffling them and allowing only 4,000 data rows with zero-degree steering angle. This filtering was done to help balancing out the data a bit. Originally, I had filtered out most of the zero-angle data, leaving only about 600 samples to match similar frequencies of the other non-zero degree angles. However, this massive truncation of the data led to a very unstable car during driving on straighaways, where the car would weave left and right very rapidly. Thus, by trial and error, 4,000 of the zero-degree data points were reintroduced back into the data set. Here's a figure showing the distribution of the angle data from the data set:

![alt text][angle_histogram]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][loss_history]

#### Notes from the model refining process
Below is a summary of the model iterations I went through. There are some comical recordings of the car going literally all over the place in the beginning.

| Run   | Recording                                                                                         | Notes                                     |
|------:|:------------------------------------------------------------------------------------------------|:------------------------------------------|
|  0    | [run0.mp4](https://github.com/allen8r/sdc-behavioral-cloning/blob/master/runs/run0.mp4?raw=true)| Simple regression model to predict steering angle based on center images only; model working in simulator, but with disastrous results |                   
|  1    | [run1.mp4](https://github.com/allen8r/sdc-behavioral-cloning/blob/master/runs/run1.mp4?raw=true)| Add preprocessing layer to normalize image RGB values |
|  2    | [run2.mp4](https://github.com/allen8r/sdc-behavioral-cloning/blob/master/runs/run2.mp4?raw=true)| Use LeNet architecture with Conv2D, relu, MaxPooling2D |
|  3    | [run3.mp4](https://github.com/allen8r/sdc-behavioral-cloning/blob/master/runs/run3.mp4?raw=true)| Augment image data by flipping center images horizontally. Auto run looks the best so far; staying in lane mostly and starting to make turns ok; but ran off track when running into sharp left turn |
|  4    | [run4.mp4](https://github.com/allen8r/sdc-behavioral-cloning/blob/master/runs/run4.mp4?raw=true)| Augment data by using right and left camera images. Performing worse; running offtrack to the right, right away; may be overfitting to the left turns |
|  5    | [run5.mp4](https://github.com/allen8r/sdc-behavioral-cloning/blob/master/runs/run5.mp4?raw=true)| Add dropout layers, increase steering correction for left and right camera views |
|  6    | [run6.mp4](https://github.com/allen8r/sdc-behavioral-cloning/blob/master/runs/run6.mp4?raw=true)| Update to 10 epochs in training. Remove most non-zero steering angle frames |
|  7    | [run7.mp4](https://github.com/allen8r/sdc-behavioral-cloning/blob/master/runs/run7.mp4?raw=true()| Add generator to supply batch samples. Add back some zero-angle data to stabalize car on straightaways because car was weaving left and right |
|  8    | [run8.mp4](https://github.com/allen8r/sdc-behavioral-cloning/blob/master/runs/run8.mp4?raw=true)| Implement Nvidia self-driving car architecture. Only make angle correction for left/right camera images if center angle is NOT zero |
|  **9**    | [**video.mp4**](https://github.com/allen8r/sdc-behavioral-cloning/blob/master/video.mp4?raw=true)| Final run; rename video.mp4 per naming convention for file submissions. Add back majority of zero-steering angle images to help stabalize the car on straightaways |
| sample| [sample_run.mp4](https://github.com/allen8r/sdc-behavioral-cloning/blob/master/runs/sample_run.mp4?raw=true)| Autonomous run using udacity provided sample data  |
