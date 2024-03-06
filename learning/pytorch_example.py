import numpy as np
from netCDF4 import Dataset as ncDataset
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import pytorch_dependencies

if __name__ == "__main__":
    superobDataset = pytorch_dependencies.SuperobDataset()
    labelDataset = pytorch_dependencies.LabelDataset()
    # "batch_size=None" appears to be critical here, or else the algorithm chokes when it tries to reconcile
    # files with different vector-lengths
    superobLoader = DataLoader(superobDataset, batch_size=None, shuffle=False, num_workers=0)
    labelLoader = DataLoader(labelDataset, batch_size=None, shuffle=False, num_workers=0)
    # predictors
    latTen = torch.from_numpy(np.asarray([]))
    lonTen = torch.from_numpy(np.asarray([]))
    preTen = torch.from_numpy(np.asarray([]))
    timTen = torch.from_numpy(np.asarray([]))
    uwdTen = torch.from_numpy(np.asarray([]))
    vwdTen = torch.from_numpy(np.asarray([]))
    nobTen = torch.from_numpy(np.asarray([]))
    ntyTen = torch.from_numpy(np.asarray([]))
    uvrTen = torch.from_numpy(np.asarray([]))
    vvrTen = torch.from_numpy(np.asarray([]))
    pvrTen = torch.from_numpy(np.asarray([]))
    tvrTen = torch.from_numpy(np.asarray([]))
    dvrTen = torch.from_numpy(np.asarray([]))
    for i in range(len(superobLoader)):
        (lat,lon,pre,tim,uwd,vwd,nob,nty,uvr,vvr,pvr,tvr,dvr) = superobLoader.dataset[i]
        latTen = torch.cat((latTen, lat))
        lonTen = torch.cat((lonTen, lon))
        preTen = torch.cat((preTen, pre))
        timTen = torch.cat((timTen, tim))
        uwdTen = torch.cat((uwdTen, uwd))
        vwdTen = torch.cat((vwdTen, vwd))
        nobTen = torch.cat((nobTen, nob))
        ntyTen = torch.cat((ntyTen, nty))
        uvrTen = torch.cat((uvrTen, uvr))
        vvrTen = torch.cat((vvrTen, vvr))
        pvrTen = torch.cat((pvrTen, pvr))
        tvrTen = torch.cat((tvrTen, tvr))
        dvrTen = torch.cat((dvrTen, dvr))
    
    print("lat tensor has shape: ", latTen.shape)
    print("lon tensor has shape: ", lonTen.shape)
    print("pre tensor has shape: ", preTen.shape)
    print("tim tensor has shape: ", timTen.shape)
    print("uwd tensor has shape: ", uwdTen.shape)
    print("vwd tensor has shape: ", vwdTen.shape)
    print("nob tensor has shape: ", nobTen.shape)
    print("nty tensor has shape: ", ntyTen.shape)
    print("uvr tensor has shape: ", uvrTen.shape)
    print("vvr tensor has shape: ", vvrTen.shape)
    print("pvr tensor has shape: ", pvrTen.shape)
    print("tvr tensor has shape: ", tvrTen.shape)
    print("dvr tensor has shape: ", dvrTen.shape)
    
    # labels
    ulbTen = torch.from_numpy(np.asarray([]))
    vlbTen = torch.from_numpy(np.asarray([]))
    for i in range(len(labelLoader)):
        (uwd,vwd) = labelLoader.dataset[i]
        ulbTen = torch.cat((ulbTen, uwd))
        vlbTen = torch.cat((vlbTen, vwd))
    
    print("ulb tensor has shape: ", ulbTen.shape)
    print("vlb tensor has shape: ", vlbTen.shape)
