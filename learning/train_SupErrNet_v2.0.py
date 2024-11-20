# Using SupErrNetv2.0 core, run a full version conforming to SupErrNetv2.0 design:
#   CNN component: 21x21 grid of temperature data at observation's ERA5 model-level, processed to a flattened (3x3x3) 27-nodes
#   36 direct input predictors/nodes (plus 27 CNN nodes = 63 total input nodes):
#       uwd, vwd, lat, pre, tim, nob, nty, nid, qim, uvr, vvr, pvr, tvr, dvr, qvr,
#       has_240, has_241, has_242, has_243, has_244, has_245,
#       has_246, has_247, has_248, has_249, has_250, has_251,
#       has_252, has_253, has_254, has_255, has_256, has_257,
#       has_258, has_259, has_260
#   43 nodes per hidden layer
#   3 hidden layers
#
# This version will utilize predictor statistics [mean, stdv, minVal, maxVal] from ~800M observations
# provided in the superobs_stats.nc file
#
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
# define SupErrNetv2: ReLu stack CNN + linear/leakyReLU stack NN estimating a single predictand from both image and predictor data
class SupErrNetv2(nn.Module):
    def __init__(self):
        super(SupErrNetv2, self).__init__()
        # define convolutional neural network that processes 2d field into inputs to FC network
        self.cnn = nn.Sequential(
                                 # input (1,21,21) array, return unpadded (3,19,19) array with (3,3) kernel
                                 nn.Conv2d(in_channels=1,  # input features, e.g. number of 2D variables being passed
                                           out_channels=3, # output features: e.g. nodes of hidden layer
                                           kernel_size=(3, 3), # can adjust
                                           stride=1, # default
                                           padding=0), # padding added
                                 nn.ReLU(),
                                 # pool with (2,2) kernel to reduce to (3,9,9) array
                                 nn.MaxPool2d((2, 2)),
                                 # input (3,9,9) array, return padded (3,7,7) array with (3,3) kernel
                                 nn.Conv2d(in_channels=3,  # input features, e.g. number of 2D variables being passed
                                           out_channels=3, # output features: e.g. nodes of hidden layer
                                           kernel_size=(3, 3), # can adjust
                                           stride=1, # default
                                           padding=0), # padding added
                                 nn.ReLU(),
                                 # pool with (2,2) kernel to reduce to (3,3,3) array
                                 nn.MaxPool2d((2, 2)),
                                 # flatten to 3x3x3 = (1,27) output array
                                 nn.Flatten()
                                )
        # define FC network that processes input data + CNN output into prediction
        self.fc = nn.Sequential(
                                nn.Linear(27+36,43),  # input layer
                                nn.LeakyReLU(),
                                nn.Linear(43, 43),  # hidden layer 1
                                nn.LeakyReLU(),
                                nn.Linear(43, 43),  # hidden layer 2
                                nn.LeakyReLU(),
                                nn.Linear(43, 43),  # hidden layer 3
                                nn.LeakyReLU(),
                                nn.Linear(43, 1)    # output layer
                               )
        
    def forward(self, image, data):
        # process image through CNN, produce output x1
        x1 = self.cnn(image)
        # pass through data directly as x2
        x2 = data
        # concatenate outputs to master input array to FC network
        x = torch.cat((x1, x2), dim=1)
        # process inputs through FC network
        output = self.fc(x.float())
        return output


# define internal functions
# extract_predictors: given data from data-loader, compose predictor arrays into stacked tensor
# INPUTS:
#   dataset: torch.utils.data.DataLoader.dataset, batch of data extracted from data-loader (tuple of tensors)
#   stats: tuple containing [mean,stdv,minVal,maxVal] statistics for each predictor (computed offline from training data)
#
# OUTPUTS:
#   predictors: tensor of stacked vectors representing tensors for each predictor
#
# DEPENDENCIES:
#   torch
#   torch.utils.data.DataLoader
#   torch.utils.data.Dataset
def extract_predictors(dataset, statsTuple):
    # extract tensors from dataset tuple
    lat = dataset[0]
    lon = dataset[1]
    pre = dataset[2]
    tim = dataset[3]
    uwd = dataset[4]
    vwd = dataset[5]
    nob = dataset[6]
    nty = dataset[7]
    nid = dataset[8]
    qim = dataset[9]
    uvr = dataset[10]
    vvr = dataset[11]
    pvr = dataset[12]
    tvr = dataset[13]
    dvr = dataset[14]
    qvr = dataset[15]
    has_240 = dataset[16]
    has_241 = dataset[17]
    has_242 = dataset[18]
    has_243 = dataset[19]
    has_244 = dataset[20]
    has_245 = dataset[21]
    has_246 = dataset[22]
    has_247 = dataset[23]
    has_248 = dataset[24]
    has_249 = dataset[25]
    has_250 = dataset[26]
    has_251 = dataset[27]
    has_252 = dataset[28]
    has_253 = dataset[29]
    has_254 = dataset[30]
    has_255 = dataset[31]
    has_256 = dataset[32]
    has_257 = dataset[33]
    has_258 = dataset[34]
    has_259 = dataset[35]
    has_260 = dataset[36]
    # extract statistics from stats tuple
    latStats = statsTuple[0]
    lonStats = statsTuple[1]
    preStats = statsTuple[2]
    timStats = statsTuple[3]
    uwdStats = statsTuple[4]
    vwdStats = statsTuple[5]
    nobStats = statsTuple[6]
    ntyStats = statsTuple[7]
    nidStats = statsTuple[8]
    qimStats = statsTuple[9]
    uvrStats = statsTuple[10]
    vvrStats = statsTuple[11]
    pvrStats = statsTuple[12]
    tvrStats = statsTuple[13]
    dvrStats = statsTuple[14]
    qvrStats = statsTuple[15]
    iAVG, iSTD, iMIN, iMAX = [0, 1, 2, 3]
    # normalize predictors based on known min/max values, or estimated values from stats tuple
    # lat: normalize by known bounds of [-90,90]
    xmin = -90.
    xmax = 90.
    lat = (lat - xmin) / (xmax - xmin)
    # lon: normalize by known bounds of [-180,180]
    xmin = -180.
    xmax = 180.
    lon = (lon - xmin) / (xmax - xmin)
    # pre: standardize by estimated mean/std of [65520.,29367.]
    xmean = preStats[iAVG]  # 65520.
    xstdv = preStats[iSTD]  # 29367.
    pre = (pre - xmean) / (xstdv)
    # tim: normalize by known bounds of [-3,3]
    xmin = -3.
    xmax = 3.
    tim = (tim - xmin) / (xmax - xmin)
    # uwd: standardize by estimated mean/std of [3.,14.]
    xmean = uwdStats[iAVG]  #  3.
    xstdv = uwdStats[iSTD]  # 14.
    uwd = (uwd - xmean) / (xstdv)
    # vwd: standardize by estimated mean/std of [1.,8.8]
    xmean = vwdStats[iAVG]  # 1.
    xstdv = vwdStats[iSTD]  # 8.8
    vwd = (vwd - xmean) / (xstdv)
    # nob: normalize by estimated bounds of [0,500]
    xmin = nobStats[iMIN]  # 0.
    xmax = nobStats[iMAX]  # 550.
    nob = (nob - xmin) / (xmax - xmin)
    # nty: normalize by estimated bounds of [1,7]
    xmin = ntyStats[iMIN]  # 1.
    xmax = ntyStats[iMAX]  # 7.
    nty = (nty - xmin) / (xmax - xmin)
    # nid: normalize by estimated bounds of [1,3]
    xmin = nidStats[iMIN]  # 1.
    xmax = nidStats[iMAX]  # 3.
    nid = (nid - xmin) / (xmax - xmin)
    # qim: normalize by known bounds of [0.,100.]
    xmin = 0.
    xmax = 100.
    qim = (qim - xmin) / (xmax - xmin)
    # uvr: standardize by estimated mean/std of [0.217,0.738]
    xmean = uvrStats[iAVG]  # 0.217
    xstdv = uvrStats[iSTD]  # 0.738
    uvr = (uvr - xmean) / (xstdv)
    # vvr: standardize by estimated mean/std of [0.210,0.723]
    xmean = vvrStats[iAVG]  # 0.210
    xstdv = vvrStats[iSTD]  # 0.723
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
    # qvr: normalize by known bounds of [0,2500.] (with known bounds of [0.,100.] on qi)
    xmin = 0.
    xmax = 2500.
    qvr = (qvr - xmin) / (xmax - xmin)
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
    predictors = torch.stack([uwd, vwd, lat, pre, tim, nob, nty, nid, qim, uvr, vvr, pvr, tvr, dvr, qvr,
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
def train_and_validate_epoch(model, loss_fn, optimizer, dataTrainLoader, labelTrainLoader, cnnTrainLoader, dataValidLoader, labelValidLoader, cnnValidLoader, statsTuple, miniBatchSize):
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
    # initialize zero-array of size (100,100) to store 2D histogram of prediction vs label data from validation
    hist2D = np.zeros((100,100))
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
        # extract CNN image data
        cnn = cnnTrainLoader.dataset[i]
        # extract predictors from inputs
        X = extract_predictors(inputs, statsTuple)
        # extract predictands from inputs and labels
        Y = extract_predictands(inputs, labels)
        # filter out NaN values from all predictors, cnn, and predictands: retain only indices where all values are non-NaN
        kp1 = torch.where(torch.isnan(inputs[0])==False)[0].numpy()  # numpy array of indicies
        for j in range(1,len(inputs)):
            kp1 = np.intersect1d(kp1,torch.where(torch.isnan(inputs[j])==False)[0].numpy())
        for j in range(len(labels)):
            kp1 = np.intersect1d(kp1,torch.where(torch.isnan(labels[j])==False)[0].numpy())
        # further filter out any index where C.sum(axis=1)=0. (all zero values)
        C = cnn.reshape((cnn.size()[0], cnn.size()[1]*cnn.size()[2]))
        kp2 = torch.where(C.sum(axis=1) > 0.)[0].numpy()
        kp = np.intersect1d(kp1, kp2)
        print('Training batch {:d} of {:d} ({:d} obs)'.format(i+1, len(dataTrainLoader), len(kp)))
        X = X[:,kp]
        if len(Y.shape) == 1:
            Y = Y[kp]
        else:
            Y = Y[:,kp]
        # permute C to put index value in axis=3
        C = cnn[kp,:,:].permute(1,2,0)
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
            outputs = model(C[:,:,kpBeg:kpEnd].permute(2,0,1).float().unsqueeze(dim=1), X[:,kpBeg:kpEnd].T.float())  # X is transposed and asserted as torch.float
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
            else:
                print('bad mini-batch {:d}'.format(k))
        # re-test entire batch for training loss and prediction/predictand ranges
        model.eval()
        with torch.no_grad():
            outputs = model(C.permute(2,0,1).float().unsqueeze(dim=1), X.T.float())
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
            # extract CNN image data
            cnnVal = cnnValidLoader.dataset[i]
            # extract predictors from inputsVal
            vX = extract_predictors(inputsVal, statsTuple)
            # extract predictands from inputsVal and labelsVal
            vY = extract_predictands(inputsVal, labelsVal)
            # filter out NaN values from all predictors, cnn, and predictands: retain only indices where all values are non-NaN
            kp1 = torch.where(torch.isnan(inputsVal[0])==False)[0].numpy()  # numpy array of indicies
            for j in range(1,len(inputsVal)):
                kp1 = np.intersect1d(kp1,torch.where(torch.isnan(inputsVal[j])==False)[0].numpy())
            for j in range(len(labelsVal)):
                kp1 = np.intersect1d(kp1,torch.where(torch.isnan(labelsVal[j])==False)[0].numpy())
            # further filter out any index where vC.sum(axis=1)=0. (all zero values)
            vC = cnnVal.reshape((cnnVal.size()[0], cnnVal.size()[1]*cnnVal.size()[2]))
            kp2 = torch.where(vC.sum(axis=1) > 0.)[0].numpy()
            kp = np.intersect1d(kp1, kp2)
            vX = vX[:,kp]
            if len(Y.shape) == 1:
                vY = vY[kp]
            else:
                vY = vY[:,kp]
            # permute C to put index value in axis=3
            vC = cnnVal[kp,:,:].permute(1,2,0)
            # make predictions for this batch
            vout = model(vC.permute(2,0,1).float().unsqueeze(dim=1), vX.T.float())
            # compute loss
            if len(vY.shape) == 1:
                vloss = loss_fn(vout.T, vY[None,:].float())
            else:
                vloss = loss_fn(vout.T, vY.float())
            # store validation statistics
            lossValid.append(vloss.item())
            pMinValid.append(vout.min().item())
            pMaxValid.append(vout.max().item())
            vMinValid.append(vY.min().item())
            vMaxValid.append(vY.max().item())
            # accumulate counts in hist2D of prediction (dim=0) vs label (dim=1)
            H, xe, ye = np.histogram2d(x=vout.detach().numpy().squeeze(),y=vY.detach().numpy().squeeze(),bins=100,range=[[0.,100.],[0.,100.]])
            hist2D = hist2D + H
            # accumulate validLossRunning
            #validLossRunning += vloss.item()
            # update min/max
            #pMinRunning = vout.min().item() if vout.min().item() < pMinRunning else pMinRunning
            #pMaxRunning = vout.max().item() if vout.max().item() > pMaxRunning else pMaxRunning
            #vMinRunning = vY.min().item() if vY.min().item() < vMinRunning else vMinRunning
            #vMaxRunning = vY.max().item() if vY.max().item() > vMaxRunning else vMaxRunning
        # store validation statistics
        #lossValid.append(validLossRunning/float(len(dataValidLoader)))
        #pMinValid.append(pMinRunning)
        #pMaxValid.append(pMaxRunning)
        #vMinValid.append(vMinRunning)
        #vMaxValid.append(vMaxRunning)
    # return statistics
    return lossTrain, vMinTrain, vMaxTrain, pMinTrain, pMaxTrain, lossValid, vMinValid, vMaxValid, pMinValid, pMaxValid, hist2D
#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define full-path to training/validation directories and number of training epochs')
    parser.add_argument('trainDir', metavar='TRAINDIR', type=str, help='full path to training data/label directory')
    parser.add_argument('validDir', metavar='VALIDDIR', type=str, help='full path to validation data/label directory')
    parser.add_argument('statsFile', metavar='STATSFILE', type=str, help='full path to netCDF predictor stats file [mean,stdv,min,max]')
    parser.add_argument('epoch', metavar='EPOCH', type=int, help='current epoch to train')
    parser.add_argument('anneal', metavar='ANNEAL', type=float, help='annealing-rate applied per-epoch')
    parser.add_argument('saveDir', metavar='SAVEDIR', type=str, help='full path to save directory to save model, training stats')
    parser.add_argument('saveName', metavar='SAVEDIR', type=str, help='model-name to save model, training stats')
    userInputs = parser.parse_args()
    #userInputs = parser.parse_args([ 'training/',
    #                                 'training/',
    #                                 '/scratch1/NCEPDEV/da/Brett.Hoover/ML_AMVs/superob_stats.nc',
    #                                 '0',
    #                                 '0.25',
    #                                 './',
    #                                 'SupErrNetv2.0'
    #                               ])
    # quality control saveDir, add '/' at end if not present
    saveDir = userInputs.saveDir if userInputs.saveDir[-1]=='/' else userInputs.saveDir + '/'
    # load statsFile and extract predictor statistics
    hdl = ncDataset(userInputs.statsFile)
    latStats = np.asarray(hdl.variables['lat']).squeeze()
    lonStats = np.asarray(hdl.variables['lon']).squeeze()
    preStats = np.asarray(hdl.variables['pre']).squeeze()
    timStats = np.asarray(hdl.variables['tim']).squeeze()
    uwdStats = np.asarray(hdl.variables['uwd']).squeeze()
    vwdStats = np.asarray(hdl.variables['vwd']).squeeze()
    nobStats = np.asarray(hdl.variables['nob']).squeeze()
    ntyStats = np.asarray(hdl.variables['nty']).squeeze()
    nidStats = np.asarray(hdl.variables['nid']).squeeze()
    qimStats = np.asarray(hdl.variables['qim']).squeeze()
    uvrStats = np.asarray(hdl.variables['uvr']).squeeze()
    vvrStats = np.asarray(hdl.variables['vvr']).squeeze()
    pvrStats = np.asarray(hdl.variables['pvr']).squeeze()
    tvrStats = np.asarray(hdl.variables['tvr']).squeeze()
    dvrStats = np.asarray(hdl.variables['dvr']).squeeze()
    qvrStats = np.asarray(hdl.variables['qvr']).squeeze()
    statsTuple = (latStats, lonStats, preStats, timStats,
                  uwdStats, vwdStats, nobStats, ntyStats,
                  nidStats, qimStats, uvrStats, vvrStats,
                  pvrStats, tvrStats, dvrStats, qvrStats)
    # define model on hardware-type
    model = SupErrNetv2().to('cpu')
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
    cnnTrainDataSet = pytorch_dependencies.CNNDataset(dataDir=userInputs.trainDir)
    #   training data-loaders
    superobTrainLoader = DataLoader(superobTrainDataSet, batch_size=None, shuffle=False, num_workers=0)
    labelTrainLoader = DataLoader(labelTrainDataSet, batch_size=None, shuffle=False, num_workers=0)
    cnnTrainLoader = DataLoader(cnnTrainDataSet, batch_size=None, shuffle=False, num_workers=0)
    #   validation datasets
    superobValidDataSet = pytorch_dependencies.SuperobDataset(dataDir=userInputs.validDir)
    labelValidDataSet = pytorch_dependencies.LabelDataset(dataDir=userInputs.validDir)
    cnnValidDataSet = pytorch_dependencies.CNNDataset(dataDir=userInputs.validDir)
    #   validation data-loaders
    superobValidLoader = DataLoader(superobValidDataSet, batch_size=None, shuffle=False, num_workers=0)
    labelValidLoader = DataLoader(labelValidDataSet, batch_size=None, shuffle=False, num_workers=0)
    cnnValidLoader = DataLoader(cnnValidDataSet, batch_size=None, shuffle=False, num_workers=0)
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
     lossValid, vMinValid, vMaxValid, pMinValid, pMaxValid, hist2D) = train_and_validate_epoch(model,
                                                                                               lossFunc,
                                                                                               optimizer,
                                                                                               superobTrainLoader,
                                                                                               labelTrainLoader,
                                                                                               cnnTrainLoader,
                                                                                               superobValidLoader,
                                                                                               labelValidLoader,
                                                                                               cnnValidLoader,
                                                                                               statsTuple,
                                                                                               miniBatchSize=32)
    # save model to saveDir
    torch.save(model.state_dict(), saveDir + userInputs.saveName + "E{:d}".format(epoch))
    # save optimizer state to saveDir
    torch.save(optimizer.state_dict(), saveDir + userInputs.saveName + "E{:d}_optimizer".format(epoch))
    # save pickle-file containing training statistics
    picklePayload = (lossTrain, vMinTrain, vMaxTrain, pMinTrain, pMaxTrain,
                     lossValid, vMinValid, vMaxValid, pMinValid, pMaxValid)
    with open(saveDir + userInputs.saveName + "E{:d}".format(epoch) + '.pkl', 'wb') as f:
        pickle.dump(picklePayload, f)
    # save .npy file containing hist2D counts
    np.save(saveDir + userInputs.saveName + "E{:d}".format(epoch), hist2D)
#
# end
#
