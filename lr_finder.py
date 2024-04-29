import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torch.optim as optim
import copy

# min_lr: minimum learning rate
# max_lr: maximum learning rate
# n_steps: number of steps to take
# loss: loss function
# model: model
# data_loaders: data loaders
# Function to find the optimal learning rate, and return the loss at each learning rate

def lr_finder(min_lr,max_lr,n_steps,loss,model,data_loaders):
    # Save the model state
    torch.save(model.state_dict(),'tmp.pth')

    # creating an optimizer
    optimizer = optim.SGD(model.parameters(),lr=min_lr)

    # what we are tryint to accomplish by r is, 
    # to basically have a factor by which we multiply the learning rate at each step
    # and in the end it reaches to max_lr
    # so basically max_lr = min_lr * (r * r * r .... r) (n times)
    # so gradually increasing our learning rate 
    r = np.power(max_lr/min_lr,1/(n_steps - 1))

    def new_lr(epoch):
        return r ** epoch
    
    # Custom lambda based learning rate updater 
    lr_scheduler = LambdaLR(optimizer,new_lr)

    # set model in training mode
    model.train()

    losses = {}
    train_loss = 0.0

    # loop over the training data
    for batch_idx,(data,target) in enumerate(tqdm(
        enumerate(data_loaders("train")),
        total=len(data_loaders("train")),
        desc="Training",
        leave=True,
        ncols=80)):
        if torch.cuda.is_available():
            data,target = data.cuda(),target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss_val = loss(output, target)
        loss_val.backward()
        optimizer.step()

        
        # althoug we can calcaulte simple average by just appending the loss_val.data.item() and dividing by batch_idx + 1
        # but we are using the below formula to calculate the average loss
        # the benifit is that in this the latter loss values have more weightage than the previous ones
        # and gives a better estimate of the loss
        train_loss = train_loss + ((1 / (batch_idx + 1) * (loss_val.data.item() - train_loss)))

        # storing the loss for the current learning rate, as the lr is updated after each batch
        losses[lr_scheduler.get_last_lr()[0]] = train_loss

        # break the loop if the loss is greater than 10 times the minimum loss
        if train_loss / min(losses.values()) > 10:
            break

        if batch_idx == n_steps - 1:
            break
        else:
            # Increase the learning rate for the next iteration
            lr_scheduler.step()

        
    # Restore model to its initial state
    model.load_state_dict(torch.load('__weights_backup'))
    
    return losses

