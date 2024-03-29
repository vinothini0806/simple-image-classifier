# simple-image-classifier
### Results:
- Results after training 50,176 images of CIFA10 image data set:
    - number of epochs = 25
    - MODEL 
        - CONV 5x5 filter layers with batch norm - **12 x 12 x 24 x 24** 
        - lowest_train_loss: 0.0004581 
        - highet_train_accuracy: 0.0.4181 
        - lowest_test_loss: 0.001859 
        - highest_test_accuracy: 0.685

The performance of the model was very good and was able to predict the animals with 66-67% accuracy.

Plots for model accuracy and loss are following:

![alt text](https://raw.githubusercontent.com/vinothini0806/simple-image-classifier/master/output/accuracy_50176images_20epochs.PNG "Model accuracy of training images")

![alt text](https://raw.githubusercontent.com/vinothini0806/simple-image-classifier/master/output/loss_50176images_20epochs.PNG "Model loss of training images")

![alt text](https://raw.githubusercontent.com/vinothini0806/simple-image-classifier/master/output/accuracy_9824images_20epochs.PNG "Model accuracy of testing images")

![alt text](https://raw.githubusercontent.com/vinothini0806/simple-image-classifier/master/output/loss_9824images_20epochs.PNG "Model loss of testing images")



### Instructions to run the code:

- Go to directory:  _/_
- To start the training run: 
    - _$ python main.py_
