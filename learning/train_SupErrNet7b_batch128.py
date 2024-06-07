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
# set random seed for reproducibility
#torch.manual_seed(0)
# define SupErrNet: linear/ReLU stack NN estimating a single predictand from input predictors
class SupErrNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(33,23),  # input layer
            nn.LeakyReLU(),
            nn.Linear(23, 23),  # hidden layer 1
            nn.LeakyReLU(),
            nn.Linear(23, 23),  # hidden layer 2
            nn.LeakyReLU(),
            nn.Linear(23, 23),  # hidden layer 3
            nn.LeakyReLU(),
            nn.Linear(23, 1)    # output layer
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
    has_240 = dataset[13]
    has_241 = dataset[14]
    has_242 = dataset[15]
    has_243 = dataset[16]
    has_244 = dataset[17]
    has_245 = dataset[18]
    has_246 = dataset[19]
    has_247 = dataset[20]
    has_248 = dataset[21]
    has_249 = dataset[22]
    has_250 = dataset[23]
    has_251 = dataset[24]
    has_252 = dataset[25]
    has_253 = dataset[26]
    has_254 = dataset[27]
    has_255 = dataset[28]
    has_256 = dataset[29]
    has_257 = dataset[30]
    has_258 = dataset[31]
    has_259 = dataset[32]
    has_260 = dataset[33]
    # normalize predictors based on known min/max values, or estimated values based on ~100M superobs
    # lat: normalize by known bounds of [-90,90]
    xmin = -90.
    xmax = 90.
    lat = (lat - xmin) / (xmax - xmin)
    # lon: normalize by known bounds of [-180,180]
    xmin = -180.
    xmax = 180.
    lon = (lon - xmin) / (xmax - xmin)
    # pre: standardize by estimated mean/std of [65520.,29367.]
    xmean = 65520.
    xstdv = 29367.
    pre = (pre - xmean) / (xstdv)
    # tim: normalize by known bounds of [-3,3]
    xmin = -3.
    xmax = 3.
    tim = (tim - xmin) / (xmax - xmin)
    # uwd: standardize by estimated mean/std of [3.,14.]
    xmean = 3.
    xstdv = 14.
    uwd = (uwd - xmean) / (xstdv)
    # vwd: standardize by estimated mean/std of [1.,8.8]
    xmean = 1.
    xstdv = 8.8
    vwd = (vwd - xmean) / (xstdv)
    # nob: normalize by estimated bounds of [0,500]
    xmin = 0.
    xmax = 550.
    nob = (nob - xmin) / (xmax - xmin)
    # nty: normalize by estimated bounds of [1,7]
    xmin = 1.
    xmax = 7.
    nty = (nty - xmin) / (xmax - xmin)
    # uvr: standardize by estimated mean/std of [0.217,0.738]
    xmean = 0.217
    xstdv = 0.738
    uvr = (uvr - xmean) / (xstdv)
    # vvr: standardize by estimated mean/std of [0.210,0.723]
    xmean = 0.210
    xstdv = 0.723
    vvr = (vvr - xmean) / (xstdv)
    # pvr: normalize by known bounds of [0,6.25e+06] (with 25000Pa max search distance)
    xmin = 0.
    xmax = 6.25e+06
    pvr = (pvr - xmin) / (xmax - xmin)
    # tvr: normalize by known bounds of [0,0.25] (with 0.5 hr max search distance)
    xmin = 0.
    xmax = 0.25
    tvr = (tvr - xmin) / (xmax - xmin)
    # dvr: normalize by known bounds of [0,1e+10] (with 100000 m max search distance)
    xmin = 0.
    xmax = 1.0e+10
    dvr = (dvr - xmin) / (xmax - xmin)
    # NOTE: has_* variables are binary [0,1] and are normalized to [-1,1] instead
    has_240[torch.where(has_240==0)] = -1
    has_241[torch.where(has_241==0)] = -1
    has_242[torch.where(has_242==0)] = -1
    has_243[torch.where(has_243==0)] = -1
    has_244[torch.where(has_244==0)] = -1
    has_245[torch.where(has_245==0)] = -1
    has_246[torch.where(has_246==0)] = -1
    has_247[torch.where(has_247==0)] = -1
    has_248[torch.where(has_248==0)] = -1
    has_249[torch.where(has_249==0)] = -1
    has_250[torch.where(has_250==0)] = -1
    has_251[torch.where(has_251==0)] = -1
    has_252[torch.where(has_252==0)] = -1
    has_253[torch.where(has_253==0)] = -1
    has_254[torch.where(has_254==0)] = -1
    has_255[torch.where(has_255==0)] = -1
    has_256[torch.where(has_256==0)] = -1
    has_257[torch.where(has_257==0)] = -1
    has_258[torch.where(has_258==0)] = -1
    has_259[torch.where(has_259==0)] = -1
    has_260[torch.where(has_260==0)] = -1
    # construct stacked tensor of predictors
    predictors = torch.stack([uwd, vwd, lat, pre, tim, nob, nty, uvr, vvr, pvr, tvr, dvr,
                              has_240, has_241, has_242, has_243, has_244, has_245,
                              has_246, has_247, has_248, has_249, has_250, has_251,
                              has_252, has_253, has_254, has_255, has_256, has_257,
                              has_258, has_259, has_260])
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
    predictands = torch.sqrt(torch.pow(ulb-uwd,2.)+torch.pow(vlb-vwd,2.))
    # return predictands
    return predictands

# train_and validate_epoch: train a given model with training/validation data-loders, storing average loss in epoch for both training and valid data
#                           in appending lists
def train_and_validate_epoch(model, loss_fn, optimizer, dataTrainLoader, labelTrainLoader, dataValidLoader, labelValidLoader, miniBatchSize):
    #
    # initialize lists of loss values and prediction/predictand ranges, to track progress
    #
    lossTrain = []  # loss value, computed against training data (per-batch)
    pMinTrain = []  # minimum predicted value, computed against training data (per-batch)
    pMaxTrain = []  # maximum predicted value, computed against training data (per-batch)
    vMinTrain = []  # minimum real value, in training data (per-batch)
    vMaxTrain = []  # maximum real value, in training data (per-batch)
    lossValid = []  # loss value, computed against validation data (mean across all validation batches, per-epoch)
    pMinValid = []  # minimum predicted value, computed against validation data (all validation batches, per-epoch)
    pMaxValid = []  # maximum predicted value, computed against validation data (all validation batches, per-epoch)
    vMinValid = []  # minimum real value, computed against validation data (all validation batches, per-epoch)
    vMaxValid = []  # maximum real value, computed against validation data (all validation batches, per-epoch)
    #
    # train model
    #
    # loop through all batches in training data-loaders
    for i in range(len(dataTrainLoader)):
        # set model to training mode (track gradients)
        model.train(True)
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
        print('Training batch {:d} of {:d} ({:d} obs)'.format(i+1, len(dataTrainLoader), len(kp)))
        X = X[:,kp]
        if len(Y.shape) == 1:
            Y = Y[kp]
        else:
            Y = Y[:,kp]
        # start mini-batch loop
        kpBeg = 0
        kpEnd = 0
        nMiniBatches = int(np.ceil(len(kp)/miniBatchSize))
        print('     training {:d} mini-batches (<={:d} obs), ({:.2f} < Y < {:.2f})'.format(nMiniBatches,
                                                                                           miniBatchSize,
                                                                                           Y.min().item(),
                                                                                           Y.max().item()))
        for k in range(nMiniBatches):
            kpBeg = kpEnd
            kpEnd = np.min([kpEnd + miniBatchSize, len(kp)-1])
            # make predictions for this batch
            outputs = model(X[:,kpBeg:kpEnd].T.float())  # X is transposed and asserted as torch.float
            # compute the loss
            if len(Y.shape) == 1:
                loss = loss_fn(outputs.T, Y[None,kpBeg:kpEnd].float())  # use [None,:] to force dim=0 as a singleton dimension if Y is a vector
            else:
                loss = loss_fn(outputs.T, Y[:,kpBeg:kpEnd].float())
            optimizer.zero_grad()
            # compute gradients and adjust weights if loss is not NaN (skips learning problematic mini-batches)
            if not torch.isnan(loss):
                # compute gradients of loss
                loss.backward()
                # adjust learning weights
                optimizer.step()
        # re-test entire batch for training loss and prediction/predictand ranges
        model.eval()
        with torch.no_grad():
            outputs = model(X.T.float())
            loss = loss_fn(outputs.T, Y[None,:].float())
            print('     training loss for batch: {:.4E} ({:.2E} < Z < {:.2E})'.format(loss.item(),
                                                                                      outputs.min().item(),
                                                                                      outputs.max().item()))
            # store loss, predictor min/max values, and predictand min/max values
            lossTrain.append(loss.item())
            pMinTrain.append(outputs.min().item())
            pMaxTrain.append(outputs.max().item())
            vMinTrain.append(Y.min().item())
            vMaxTrain.append(Y.max().item())
    #
    # validate model
    #
    # set to evaluation mode (disables drop-out)
    model.eval()
    # disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        # loop through all batches in validation data-loaders
        validLossRunning = 0.
        pMinRunning = 1.0e+18   # absurdly large value
        pMaxRunning = -1.0e+18  # absurdly small value
        vMinRunning = 1.0e+18   # absurdly large value
        vMaxRunning = -1.0e+18  # absurdly small value
        for i in range(len(dataValidLoader)):
            print('Validating batch {:d} of {:d}...'.format(i+1, len(dataValidLoader)))
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
            # update min/max
            pMinRunning = vout.min().item() if vout.min().item() < pMinRunning else pMinRunning
            pMaxRunning = vout.max().item() if vout.max().item() > pMaxRunning else pMaxRunning
            vMinRunning = vY.min().item() if vY.min().item() < vMinRunning else vMinRunning
            vMaxRunning = vY.max().item() if vY.max().item() > vMaxRunning else vMaxRunning
        # store validation statistics
        lossValid.append(validLossRunning/float(len(dataValidLoader)))
        pMinValid.append(pMinRunning)
        pMaxValid.append(pMaxRunning)
        vMinValid.append(vMinRunning)
        vMaxValid.append(vMaxRunning)
    # return statistics
    return lossTrain, vMinTrain, vMaxTrain, pMinTrain, pMaxTrain, lossValid, vMinValid, vMaxValid, pMinValid, pMaxValid
#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define full-path to training/validation directories and number of training epochs')
    parser.add_argument('trainDir', metavar='TRAINDIR', type=str, help='full path to training data/label directory')
    parser.add_argument('validDir', metavar='VALIDDIR', type=str, help='full path to validation data/label directory')
    parser.add_argument('epoch', metavar='EPOCH', type=int, help='current epoch to train')
    parser.add_argument('anneal', metavar='ANNEAL', type=float, help='annealing-rate applied per-epoch')
    parser.add_argument('saveDir', metavar='SAVEDIR', type=str, help='full path to save directory to save model, training stats')
    parser.add_argument('saveName', metavar='SAVEDIR', type=str, help='model-name to save model, training stats')
    userInputs = parser.parse_args()
    # quality control saveDir, add '/' at end if not present
    saveDir = userInputs.saveDir if userInputs.saveDir[-1]=='/' else userInputs.saveDir + '/'
    # define model on hardware-type
    model = SupErrNet().to('cpu')
    # for epoch > 0, load prior model state
    epoch = userInputs.epoch
    if epoch > 0:
        model.load_state_dict(torch.load(saveDir + userInputs.saveName + "E{:d}".format(epoch-1)))
    # define loss function
    lossFunc = torch.nn.MSELoss()
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
    anneal = userInputs.anneal
    baseLearningRate = 4e-5
    learningRate = baseLearningRate
    # epoch 0: define optimizer based on current (base) learningRate
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    # epoch >0: define optimizer based on prior epoch's optimizer state-dictionary and apply annealing
    if epoch > 0:
        optimizer.load_state_dict(torch.load(saveDir + userInputs.saveName + "E{:d}_optimizer".format(epoch-1)))
        for g in optimizer.param_groups:
            g['lr'] = anneal * g['lr']
    # train epoch
    (lossTrain, vMinTrain, vMaxTrain, pMinTrain, pMaxTrain, 
     lossValid, vMinValid, vMaxValid, pMinValid, pMaxValid) = train_and_validate_epoch(model,
                                                                                       lossFunc,
                                                                                       optimizer,
                                                                                       superobTrainLoader,
                                                                                       labelTrainLoader,
                                                                                       superobValidLoader,
                                                                                       labelValidLoader,
                                                                                       miniBatchSize=128)
    # save model to saveDir
    torch.save(model.state_dict(), saveDir + userInputs.saveName + "E{:d}".format(epoch))
    # save optimizer state to saveDir
    torch.save(optimizer.state_dict(), saveDir + userInputs.saveName + "E{:d}_optimizer".format(epoch))
    # save pickle-file containing training statistics
    picklePayload = (lossTrain, vMinTrain, vMaxTrain, pMinTrain, pMaxTrain,
                     lossValid, vMinValid, vMaxValid, pMinValid, pMaxValid)
    with open(saveDir + userInputs.saveName + "E{:d}".format(epoch) + '.pkl', 'wb') as f:
        pickle.dump(picklePayload, f)
#
# end
#
