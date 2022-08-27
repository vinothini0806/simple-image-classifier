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

![alt text](./output/accuracy_5000images_15epochs.png?raw=true "Model accuracy with 50 images")

![alt text](./output/loss_5000images_15epochs.png?raw=true "Model loss with 5000 images")

![alt text](./output/accuracy_18000images_15epochs.png?raw=true "Model accuracy with 18000 images")

![alt text](./output/loss_18000images_15epochs.png?raw=true "Model loss with 18000 images")



### Instructions to run the code:

- Go to directory:  _/_
- To start the training run: 
    - _$ python main.py_
