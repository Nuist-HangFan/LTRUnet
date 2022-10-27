from utils.normalization import *

import torch
import torch.nn as nn
import torch.optim as optim


def Optimizer(model, lr, weight_decay):
    """ Set the Optimizer for training """
    return optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)

def Scheduler(optimizer, if_PGT=False):
    """ Set the CosineAnnealing scheduler for learning rate """
    if if_PGT:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.65)
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 32,)

def Loss_func():
    """ Define the loss function """
    return nn.MSELoss()



# Trian 
def Train_step(model, DEVICE, EPOCHS, BATCH_SIZE,lr, weight_decay,best_save_location, writer, Train_loader, Validation_loader, if_PGT=False):
    """ Function for training the model
    Args:
        model: model to be trained
        DEVICE: the device for training
        best_save_location: path to save the best parameters
        fina_save_location: path to save the final parameters of the training
        writter: writter to record the parameters during training
        Trian_loader: data loader of training data
        Validation_loader: data loader of validation data
        loss_fuc: loss function
    """
    min_loss = 99999
    best_epoch = 0
    optimizer = Optimizer(model, lr, weight_decay)
    scheduler = Scheduler(optimizer, if_PGT)
    loss_func = Loss_func()
    for epoch in range(EPOCHS):
        for step, (data, target) in enumerate(Train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output,target)
            loss.backward()
            optimizer.step()

            # Summary per 1/4 epoch
            if (step+1)%(len(Train_loader.dataset)//BATCH_SIZE//4) == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, step * len(data), len(Train_loader.dataset),
                    100. * step / len(Train_loader), loss.item()))
                writer.add_scalar("Train_loss", loss, epoch + (step+1)%(len(Train_loader.dataset)//BATCH_SIZE//4))
        
        scheduler.step()
        
        # Validation per epoch
        valida_loss = 0
        model_err = 0
        with torch.no_grad():
            for step, (data, target) in enumerate(Validation_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                valida_loss += loss_func(output,target).item() * data.size(0)
                model_err += loss_func(turn_back_label(output), turn_back_label(target)).item() * data.size(0)

            valida_loss /= len(Validation_loader.dataset) 
            model_err /= len(Validation_loader.dataset)
            print('\nValidation set: Average loss: {:.4f}'.format(valida_loss))
            print('Validation set: Model err(RMSE): {:.4f}\n'.format(model_err**0.5))
        
        writer.add_scalar("Validation_loss", valida_loss, epoch)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Model RMSE", model_err**0.5, epoch)

        # Save the best parameters 
        if epoch > 0.3*EPOCHS and valida_loss  < min_loss*0.99:
            min_loss = valida_loss 
            best_epoch = epoch
            torch.save(model.state_dict(), best_save_location)

    print('best_epoch:',best_epoch)



def Test_step(model, DEVICE, best_save_location, Test_loader):
    """ Function for training the model
    Args:
        model: model to test
        DEVICE: the device for training
        best_save_location: path of the best parameters
        Test_loader: data loader of test data
        loss_fuc: loss function
    """
    model.load_state_dict(torch.load(best_save_location))
    model.eval()
    test_loss = 0
    model_err = 0
    loss_func = Loss_func()
    with torch.no_grad():
        for step, (data, target) in enumerate(Test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            loss = loss_func(output,target).item()
            test_loss += loss * data.size(0)
            model_err += loss_func(turn_back_label(output), turn_back_label(target)).item() * data.size(0)
            
        model_err /= len(Test_loader.dataset)
        test_loss /= len(Test_loader.dataset)
        print('********************************')
        print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
        print('Test set: Model err(RMSE): {:.4f}\n'.format(model_err**0.5))
    return model_err**0.5, test_loss, len(Test_loader.dataset)


