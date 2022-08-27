# import matplotlib.pyplot as plt
import numpy as np
from data import get_data
from model import Network
from loss import get_loss
from train import *
from torch.optim import Adam
import wandb
import os
# Function to show the images
# def imageshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# Function to test the model with a batch of images and show the labels predictions
# def testBatch():
#     # get batch of images from the test DataLoader  
#     images, labels = next(iter(test_loader))

#     # show all images as one image grid
#     # imageshow(torchvision.utils.make_grid(images))
   
#     # Show the real labels on the screen 
#     print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
#                                for j in range(batch_size)))
  
#     # Let's see what if the model identifiers the  labels of those example
#     outputs = model(images)
    
#     # We got the probability for every 10 labels. The highest (max) probability should be correct label
#     _, predicted = torch.max(outputs, 1)
    
#     # Let's show the predicted labels on the screen to compare with the real ones
#     print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
#                               for j in range(batch_size)))

def main():
    
    wandb.init(project='cnn')
    
    train_loader,test_loader,classes = get_data()
    model = Network()
    criterion = get_loss()
    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    num_epochs = 20
    
    best_accuracy = 0.0
    
    wandb.config.update({'lr':0.001,'weight_decay':0.0001,'num_epochs':20})
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        train_loss, train_acc = train(train_loader,model, criterion,optimizer,device)
        test_loss, test_acc = test(test_loader,model, criterion,device)

        print('epoch:%d train_loss: %.3f train_accuracy: %.3f test_loss: %.3f test_accuracy: %.3f' % (epoch + 1, train_loss, train_acc,test_loss, test_acc))

        # we want to save the model if the accuracy is the best
        if test_acc > best_accuracy:
            saveModel(model)
            best_accuracy = test_acc
            
        wandb.log({
            'epoch':epoch,
            'train_loss':train_loss,
            'train_acc':train_acc,
            'test_loss':test_loss,
            'test_acc':test_acc
        })

if __name__ == "__main__":
    main()
        
        
