# Using SupErrNetv2.0 core, run a slimmed-down verion conforming to the *modified* SupErrNetv1.0 design:
#   No CNN component
#   36 input predictors/nodes:
#       uwd, vwd, lat, pre, tim, nob, nty, nid, qim, uvr, vvr, pvr, tvr, dvr, qvr,
#       has_240, has_241, has_242, has_243, has_244, has_245,
#       has_246, has_247, has_248, has_249, has_250, has_251,
#       has_252, has_253, has_254, has_255, has_256, has_257,
#       has_258, has_259, has_260
#   25 nodes per hidden layer
#   3 hidden layers
#
# This version will utilize predictor statistics [mean, stdv, minVal, maxVal] from ~800M observations
# provided in the superobs_stats.nc file
#
#import modules and dependencies
import numpy as np
from netCDF4 import Dataset as ncDataset
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import glob
import pytorch_dependencies
import argparse
import pickle
import time
# set random seed for reproducibility
#torch.manual_seed(0)
# define SupErrNetv2: linear/leakyReLU stack NN estimating a single predictand from both image and predictor data
class SupErrNetv2(nn.Module):
    def __init__(self):
        super(SupErrNetv2, self).__init__()
        # define FC network that processes input data into prediction
        self.fc = nn.Sequential(
                                # SupErrNet v1.5 fc NN configuration
                                nn.Linear(36,25),  # input layer
                                nn.LeakyReLU(),
                                nn.Linear(25, 25),  # hidden layer 1
                                nn.LeakyReLU(),
                                nn.Linear(25, 25),  # hidden layer 2
                                nn.LeakyReLU(),
                                nn.Linear(25, 25),  # hidden layer 3
                                nn.LeakyReLU(),
                                nn.Linear(25, 1)    # output layer
                               )
        
    def forward(self, data):
        # pass data directly to FC network
        x = data
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

# validate_epoch: validate a given model with validation data-loders, storing average loss and per-batch prediction and validation predictand-ranges
#                 for epoch in appending lists
def validate_epoch(model, loss_fn, dataValidLoader, labelValidLoader, statsTuple):
    #
    # initialize lists of loss values and prediction/predictand ranges, to track progress
    #
    lossValid = []  # loss value, computed against validation data (per validation batch, per-epoch)
    pMinValid = []  # minimum predicted value, computed against validation data (per validation batch, per-epoch)
    pMaxValid = []  # maximum predicted value, computed against validation data (per validation batch, per-epoch)
    vMinValid = []  # minimum real value, computed against validation data (per validation batch, per-epoch)
    vMaxValid = []  # maximum real value, computed against validation data (per validation batch, per-epoch)
    # initialize zero-array of size (100,100) to store 2D histogram of prediction vs label data (all validation batches, per epoch)
    hist2D = np.zeros((100,100))
    #
    # validate model
    #
    # set to evaluation mode (disables drop-out)
    model.eval()
    # disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i in range(len(dataValidLoader)):
            print('Validating batch {:d} of {:d}...'.format(i+1, len(dataValidLoader)), flush=True)
            # extract tuples of data and labels from data-loader batch
            inputsVal = dataValidLoader.dataset[i]
            labelsVal = labelValidLoader.dataset[i]
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
            kp = kp1
            vX = vX[:,kp]
            if len(vY.shape) == 1:
                vY = vY[kp]
            else:
                vY = vY[:,kp]
            # make predictions for this batch
            vout = model(vX.T.float())
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
    # return statistics
    return lossValid, vMinValid, vMaxValid, pMinValid, pMaxValid, hist2D
#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define full-path to validation directories and model name + training epoch')
    parser.add_argument('validDir', metavar='VALIDDIR', type=str, help='full path to validation data/label directory')
    parser.add_argument('statsFile', metavar='STATSFILE', type=str, help='full path to netCDF predictor stats file [mean,stdv,min,max]')
    parser.add_argument('epoch', metavar='EPOCH', type=int, help='current epoch to validate')
    parser.add_argument('saveDir', metavar='SAVEDIR', type=str, help='full path to save directory to save model, training stats')
    parser.add_argument('saveName', metavar='SAVEDIR', type=str, help='model-name to save model, training stats')
    userInputs = parser.parse_args()
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
    # load model state for selected epoch
    epoch = userInputs.epoch
    model = SupErrNetv2().to('cpu')
    model.load_state_dict(torch.load(saveDir + userInputs.saveName + "E{:d}".format(epoch), map_location=torch.device('cpu')))
    # define loss function
    lossFunc = torch.nn.MSELoss()
    # define data-loaders from dataset classes
    #   validation datasets
    superobValidDataSet = pytorch_dependencies.SuperobDataset(dataDir=userInputs.validDir)
    labelValidDataSet = pytorch_dependencies.LabelDataset(dataDir=userInputs.validDir)
    #   validation data-loaders
    superobValidLoader = DataLoader(superobValidDataSet, batch_size=None, shuffle=False, num_workers=0)
    labelValidLoader = DataLoader(labelValidDataSet, batch_size=None, shuffle=False, num_workers=0)
    # train epoch
    (lossValid, vMinValid, vMaxValid, pMinValid, pMaxValid, hist2D) = validate_epoch(model,
                                                                                     lossFunc,
                                                                                     superobValidLoader,
                                                                                     labelValidLoader,
                                                                                     statsTuple)
    # save pickle-file containing validation statistics
    picklePayload = (lossValid, vMinValid, vMaxValid, pMinValid, pMaxValid)
    with open(saveDir + userInputs.saveName + "E{:d}".format(epoch) + '_validation.pkl', 'wb') as f:
        pickle.dump(picklePayload, f)
    # save .npy file containing hist2D counts
    np.save(saveDir + userInputs.saveName + "E{:d}".format(epoch), hist2D)
#
# end
#
