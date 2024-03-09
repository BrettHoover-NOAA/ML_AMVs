# import modules and dependencies
import numpy as np
from netCDF4 import Dataset as ncDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import glob
import pytorch_dependencies
import argparse
import pickle
# define SupErrNet: linear/ReLU stack NN estimating a single predictand from input predictors
class SupErrNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
                                               nn.Linear(9, 9),   # input layer
                                               nn.ReLU(),         # nonlinear transformation
                                               nn.Linear(9, 9),   # hidden layer 1
                                               nn.ReLU(),         # nonlinear transformation
                                               nn.Linear(9, 9),   # hidden layer 2
                                               nn.ReLU(),         # nonlinear transformation
                                               nn.Linear(9, 9),   # hidden layer 3
                                               nn.ReLU(),         # nonlinear transformation
                                               nn.Linear(9, 1)    # output layer
                                              )
    def forward(self, x):
        outputs = self.linear_relu_stack(x.float())
        return outputs
# define internal functions
# extract_predictors: given data from data-loader, compose predictor arrays into stacked tensor
# INPUTS:
#   dataset: torch.utils.data.DataLoader.dataset, batch of data extracted from data-loader (tuple of tensors)
#
# OUTPUTS:
#   predictors: tensor of stacked vectors representing tensors for each predictor
#
# DEPENDENCIES:
#   torch
#   torch.utils.data.DataLoader
#   torch.utils.data.Dataset
def extract_predictors(dataset):
    # extract tensors from dataset tuple
    lat = dataset[0]
    lon = dataset[1]
    pre = dataset[2]
    tim = dataset[3]
    uwd = dataset[4]
    vwd = dataset[5]
    nob = dataset[6]
    nty = dataset[7]
    uvr = dataset[8]
    vvr = dataset[9]
    pvr = dataset[10]
    tvr = dataset[11]
    dvr = dataset[12]
    # construct stacked tensor of predictors
    predictors = torch.stack([uwd, pre, nob, nty, uvr, vvr, pvr, tvr, dvr])
    # return predictors
    return predictors


# extract_predictands: given data and labels from data-loaders, compose predictand arrays into stacked tensor
# INPUTS:
#   dataset: torch.utils.data.DataLoader.dataset, batch of data extracted from data-loader (tuple of tensors)
#   labelset: torch.utils.data.DataLoader.dataset, batch of labels extracted from data-loader (tuple of tensors)
#
# OUTPUTS:
#   predictands: tensor of stacked vectors representing tensors for each predictand
#
# DEPENDENCIES:
#   torch
#   torch.utils.data.DataLoader
#   torch.utils.data.Dataset
def extract_predictands(dataset, labelset):
    # extract tensors from dataset tuple
    uwd = dataset[4]
    vwd = dataset[5]
    # extract tensors from labelset tuple
    ulb = labelset[0]
    vlb = labelset[1]
    # construct predictands tensor (length of vector difference between data and labels)
    predictands = torch.sqrt(torch.pow(ulb-uwd,2.))
    # return predictands
    return predictands


# train_and validate_epoch: train a given model with training/validation data-loders, storing average loss in epoch for both training and valid data
#                           in appending lists
# INPUTS:
#   model: instance of torch model class to train
#   loss_fn: torch loss-function
#   optimizer: torch optimizer
#   dataTrainLoader: data-loader of training data
#   labelTrainLoader: data-loader of training-label data
#   dataValidLoader: data-loader of validation data
#   labelValidLoader: data-loader of validation-label data
#   lossTrain: appending list of loss values averaged across all training batches (defaults to empty list)
#   lossValid: appending list of loss values averaged across all validation batches (defaults to empty list)
#   epoch: integer-value of epoch (defaults to zero)
#
# OUTPUTS:
#   lossTrain: list of loss values averaged across all training batches, per epoch
#   lossValid: list of loss values averaged across all validation batches, per epoch
#
# DEPENDENCIES:
#   numpy as np
#   torch
#   torch.utils.data.DataLoader
#   torch.utils.data.Dataset
def train_and_validate_epoch(model, loss_fn, optimizer, dataTrainLoader, labelTrainLoader, dataValidLoader, labelValidLoader, lossTrain=[], lossValid=[], epoch=0):
    #
    # train model
    #
    # set model to training mode (track gradients)
    model.train(True)
    # loop through all batches in training data-loaders
    trainLossRunning = 0.
    print('epoch {:d}: training {:d} batches'.format(epoch, len(dataTrainLoader)))
    for i in range(len(dataTrainLoader)):
        # extract tuples of data and labels from data-loader batch
        inputs = dataTrainLoader.dataset[i]
        labels = labelTrainLoader.dataset[i]
        # extract predictors from inputs
        X = extract_predictors(inputs)
        # extract predictands from inputs and labels
        Y = extract_predictands(inputs, labels)
        # filter out NaN values from all predictors and predictands: retain only indices where all values are non-NaN
        kp = torch.where(torch.isnan(inputs[0])==False)[0].numpy()  # numpy array of indicies
        for j in range(1,len(inputs)):
            kp = np.intersect1d(kp,torch.where(torch.isnan(inputs[j])==False)[0].numpy())
        for j in range(len(labels)):
            kp = np.intersect1d(kp,torch.where(torch.isnan(labels[j])==False)[0].numpy())
        X = X[:,kp]
        if len(Y.shape) == 1:
            Y = Y[kp]
        else:
            Y = Y[:,kp]
        # zero your gradients for every batch!
        optimizer.zero_grad()
        # make predictions for this batch
        outputs = model(X.T.float())  # X is transposed and asserted as torch.float
        # compute the loss
        if len(Y.shape) == 1:
            loss = loss_fn(outputs.T, Y[None,:].float())  # use [None,:] to force dim=0 as a singleton dimension if Y is a vector
        else:
            loss = loss_fn(outputs.T, Y.float())
        # accumulate trainLossRunning
        trainLossRunning += loss.item()
        # compute gradients of loss
        loss.backward()
        # adjust learning weights
        optimizer.step()
    # compute average training loss across all batches and store in lossTrain
    lossTrain.append(trainLossRunning/float(i))
    #
    # validate model
    #
    # set to evaluation mode (disables drop-out)
    model.eval()
    # disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        # loop through all batches in validation data-loaders
        validLossRunning = 0.
        print('epoch {:d}: validating {:d} batches'.format(epoch, len(dataValidLoader)))
        for i in range(len(dataValidLoader)):
            # extract tuples of data and labels from data-loader batch
            inputsVal = dataValidLoader.dataset[i]
            labelsVal = labelValidLoader.dataset[i]
            # extract predictors from inputsVal
            vX = extract_predictors(inputsVal)
            # extract predictands from inputsVal and labelsVal
            vY = extract_predictands(inputsVal, labelsVal)
            # filter out NaN values from all predictors and predictands: retain only indices where all values are non-NaN
            kp = torch.where(torch.isnan(inputsVal[0])==False)[0].numpy()  # numpy array of indicies
            for j in range(1,len(inputsVal)):
                kp = np.intersect1d(kp,torch.where(torch.isnan(inputsVal[j])==False)[0].numpy())
            for j in range(len(labelsVal)):
                kp = np.intersect1d(kp,torch.where(torch.isnan(labelsVal[j])==False)[0].numpy())
            vX = vX[:,kp]
            if len(Y.shape) == 1:
                vY = vY[kp]
            else:
                vY = vY[:,kp]
            # make predictions for this batch
            vout = model(vX.T.float())
            # compute loss
            if len(Y.shape) == 1:
                vloss = loss_fn(vout.T, vY[None,:].float())
            else:
                vloss = loss_fn(vout.T, vY.float())
            # accumulate validLossRunning
            validLossRunning += vloss.item()
        # compute average validation loss across all batches and store in lossValid
        lossValid.append(validLossRunning/float(i))
    # return lossTrain and lossValid
    return lossTrain, lossValid
#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define full-path to training/validation directories and number of training epochs')
    parser.add_argument('trainDir', metavar='TRAINDIR', type=str, help='full path to training data/label directory')
    parser.add_argument('validDir', metavar='VALIDDIR', type=str, help='full path to validation data/label directory')
    parser.add_argument('nEpochs', metavar='NEPOCHS', type=int, help='number of epochs to train')
    parser.add_argument('saveDir', metavar='SAVEDIR', type=str, help='full path to save directory to save model, training stats')
    parser.add_argument('saveName', metavar='SAVEDIR', type=str, help='model-name to save model, training stats')
    userInputs = parser.parse_args()
    # quality control saveDir, add '/' at end if not present
    saveDir = userInputs.saveDir if userInputs.saveDir[-1]=='/' else userInputs.saveDir + '/'
    # define model on hardware-type
    model = SupErrNet().to('cpu')
    # define loss function
    lossFunc = torch.nn.MSELoss()
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Might be the superior optimizer for a regression like this
    # define data-loaders from dataset classes
    #   training datasets
    superobTrainDataSet = pytorch_dependencies.SuperobDataset(dataDir=userInputs.trainDir)
    labelTrainDataSet = pytorch_dependencies.LabelDataset(dataDir=userInputs.trainDir)
    #   training data-loaders
    superobTrainLoader = DataLoader(superobTrainDataSet, batch_size=None, shuffle=False, num_workers=0)
    labelTrainLoader = DataLoader(labelTrainDataSet, batch_size=None, shuffle=False, num_workers=0)
    #   validation datasets
    superobValidDataSet = pytorch_dependencies.SuperobDataset(dataDir=userInputs.validDir)
    labelValidDataSet = pytorch_dependencies.LabelDataset(dataDir=userInputs.validDir)
    #   validation data-loaders
    superobValidLoader = DataLoader(superobValidDataSet, batch_size=None, shuffle=False, num_workers=0)
    labelValidLoader = DataLoader(labelValidDataSet, batch_size=None, shuffle=False, num_workers=0)
    nEpochs = userInputs.nEpochs
    # train epoch 1, passing empty lists to lossTrain and lossValid inputs
    lossTrain, lossValid = train_and_validate_epoch(model,
                                                    lossFunc,
                                                    optimizer,
                                                    superobTrainLoader,
                                                    labelTrainLoader,
                                                    superobValidLoader,
                                                    labelValidLoader,
                                                    lossTrain=[],
                                                    lossValid=[],
                                                    epoch=0)
    # train remaining epochs, passing existing lossTrain and lossValid as inputs
    if nEpochs > 1:
        for ep in range(1,nEpochs):
            lossTrain, lossValid = train_and_validate_epoch(model,
                                                            lossFunc,
                                                            optimizer,
                                                            superobTrainLoader,
                                                            labelTrainLoader,
                                                            superobValidLoader,
                                                            labelValidLoader,
                                                            lossTrain=lossTrain,
                                                            lossValid=lossValid,
                                                            epoch=ep)
    # report training statistics
    for ep in range(nEpochs):
        print('epoch {:d}: train={:.6E}, valid={:.6E}'.format(ep, lossTrain[ep], lossValid[ep]))
    # save model to saveDir
    torch.save(model.state_dict(), saveDir + userInputs.saveName)
    # save pickle-file containing training statistics
    picklePayload = (lossTrain, lossValid)
    with open(saveDir + userInputs.saveName + '.pkl', 'wb') as f:
        pickle.dump(picklePayload, f)
#
# end
#
