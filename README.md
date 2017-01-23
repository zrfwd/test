#Behavioral-Cloning
In this project, I need to use images gathered from simulator as input to train a neural network, and make the car can drive automatically on the simulator.

#First Look at Images
The images are taken by 3 different cameras: center camera, left camera, and right camera.
Below are sample images for these 3 cameras:
image taken from center camera:

image taken from left camera:

image taken from right camera:


Because my computer is slow, I use the image data provided by udacity. In this dataset, each camera takes 8036 pictures, which is not big number. Since each camera only takes 8036 snapshots, using images only from center camera is not large enough to train my neural network, so I use images from all 3 cameras instead. 

#Images Analysis and Augmentation
Because the track on the simulator only has one right turn case, rest of the track is either straight or left turn, the image data is very unbalanced, so we need to somehow modify the images in order to make the dataset more balanced. A very good strategy is to flip the image and negate the steering, so that I can produce more right turn cases to train my model. 

The second trick is to crop the image. As the images shown above, the top part of the image are trees and the bottom part is car’s hood, both these 2 parts have nothing to do with training, so I crop the image by removing these two parts. What I use is: image = image[55:135, :].

After doing these 2 augmentations, I tried to train my network, however the result is not good, so I tried some more image augmentations. Firstly I tried to use the edge extraction, my intent is since the track have lanes, I can extract these lanes and then I can train my model to drive by following these edges. I tried to use canny and sobel to extract edges and train my model, but none of these two methods work well. I guess the reason is because the image loses the color information after edge extraction, but color plays a very important role in identifying roads. Another improvement is somehow merge the original image and edge extracted image, then we can retain the color information, but amazon EC2 is extremely slow recently, 5 mins run before now takes 30 mins, so I may try this method later.

I read this blog:https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.glw8z1jiy . This blog gives several very good image augmentation methods. I choose to use the image brightness augmentation: adjusting the brightness of the image to make it darker, this method gives me a very good result.

sample image after brightness augmentation:


Finally, I resize the image size to 64x64x3, if I don’t resize, the input size is too big for my neutral network and run out of memory.

sample image after resize:




#Neural Network Architecture

My neural network is very similar to that in Nvidia’s paper. I use one max pooling at the end convolutional layers, otherwise I will run out of memory. I also use 3x3 kernel size instead of 5x5 in paper. And I use dropout in my network to make it robust, the dropout rate I use is 40%. Below is a detailed chart of my neural network:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 32, 32, 24)    1824        convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
elu_1 (ELU)                      (None, 32, 32, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 30, 30, 36)    7812        elu_1[0][0]                      
____________________________________________________________________________________________________
elu_2 (ELU)                      (None, 30, 30, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 30, 30, 36)    0           elu_2[0][0]                      
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 28, 28, 48)    15600       dropout_1[0][0]                  
____________________________________________________________________________________________________
elu_3 (ELU)                      (None, 28, 28, 48)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 28, 28, 48)    0           elu_3[0][0]                      
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 26, 26, 64)    27712       dropout_2[0][0]                  
____________________________________________________________________________________________________
elu_4 (ELU)                      (None, 26, 26, 64)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 13, 13, 64)    0           elu_4[0][0]                      
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 11, 11, 64)    36928       maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
elu_5 (ELU)                      (None, 11, 11, 64)    0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 5, 5, 64)      0           elu_5[0][0]                      
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1600)          0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1164)          1863564     flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 1164)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
elu_6 (ELU)                      (None, 1164)          0           dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           116500      elu_6[0][0]                      
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
elu_7 (ELU)                      (None, 100)           0           dropout_4[0][0]                  
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        elu_7[0][0]                      
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 50)            0           dense_3[0][0]                    
____________________________________________________________________________________________________
elu_8 (ELU)                      (None, 50)            0           dropout_5[0][0]                  
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         elu_8[0][0]                      
____________________________________________________________________________________________________
elu_9 (ELU)                      (None, 10)            0           dense_4[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          elu_9[0][0]                      
====================================================================================================
Total params: 2,075,511
Trainable params: 2,075,511
Non-trainable params: 0
____________________________________________________________________________________________________


#Training
For the whole data set, I randomly pick 80% of it as training set and 20% of it as validation set.

Because I need to randomly flip image to make the training set more balanced, I need to use fit_generator API from Keras.
	•	training_generator = get_data_generator(train, batch_size=32)
	•	validation_data_generator = get_data_generator(val, batch_size=32)
  
You can see the batch size for my data generator is 32. The image has 50% possibility to be flipped horizontally by data generator. I generate 20000 training images in one epoch, and I generate 3000 validation images per epoch. I tried 3 epochs, 5 epochs, and 8 epochs, among them 5 produces best result, so I use 5 epochs.

For the optimizer, I choose Adam optimizer and loss is measured by mean squared error.

For the images from left camera and right camera, I need to have a biased steering value for them. What I do is increase the steering by 0.25 for left camera and decrease the steering by 0.25 for right camera.

#Result
The car can run pretty well on the simulator using my model.


