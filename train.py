import torch
from torch.autograd import Variable

# Function to save the model
def saveModel(model):
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)

# Function to test the model with the test dataset and print the accuracy for the test images
def test(test_loader,model,criterion,device):
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    running_loss = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
             # compute the loss based on model output and real labels
            loss = criterion(outputs, labels)
            running_loss += loss.item()     # extract the loss value
    
    return(running_loss/total,accuracy/total)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(train_loader, model, criterion,optimizer,device):
    
    accuracy = 0.0
    total = 0.0
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader, 0):

        # get the inputs
        images = Variable(images.to(device))
        labels = Variable(labels.to(device))

        # zero the parameter gradients
        optimizer.zero_grad()
        # predict classes using images from the training set
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        accuracy += (predicted == labels).sum().item()
        # compute the loss based on model output and real labels
        loss = criterion(outputs, labels)
        # backpropagate the loss
        loss.backward()
        # adjust parameters based on the calculated gradients
        optimizer.step()

        # Let's print statistics for every 1,000 images
        running_loss += loss.item()     # extract the loss value
        total += labels.size(0)
        
    return(running_loss/total,accuracy/total)
            
        
