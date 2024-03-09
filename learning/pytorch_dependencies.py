import numpy as np
from netCDF4 import Dataset as ncDataset
import torch
from torch.utils.data import Dataset, DataLoader
import glob

# SuperobDataset: Dataset class for superob tensors
#
# To use:
# > import numpy as np
# > from netCDF4 import Dataset as ncDataset
# > import torch
# > from torch.utils.data import Dataset, DataLoader
# > import glob
# >
# > superobDataset = SuperobDataset(dataDir="/path/to/data/directory/")
# > superobLoader = DataLoader(superobDataset, batch_size=None, shuffle=False, num_workers=0)
class SuperobDataset(Dataset):
    def __init__(self, dataDir):
        self.nc4_path = dataDir + "/" if dataDir[-1] != "/" else dataDir
        date_list = glob.glob(self.nc4_path + "*")
        self.data = []
        for date_path in date_list:
            date_name = date_path.split("/")[-1]
            for nc4_path in glob.glob(date_path + "/*_superobs.nc"):
                self.data.append([nc4_path, date_name])
        print(len(self.data), " files of ", len(date_list), " directories loaded")        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        nc4_path, class_name = self.data[idx]
        lat_nc = np.asarray(ncDataset(nc4_path).variables['lat']).squeeze()
        lon_nc = np.asarray(ncDataset(nc4_path).variables['lon']).squeeze()
        pre_nc = np.asarray(ncDataset(nc4_path).variables['pre']).squeeze()
        tim_nc = np.asarray(ncDataset(nc4_path).variables['tim']).squeeze()
        uwd_nc = np.asarray(ncDataset(nc4_path).variables['uwd']).squeeze()
        vwd_nc = np.asarray(ncDataset(nc4_path).variables['vwd']).squeeze()
        nob_nc = np.asarray(ncDataset(nc4_path).variables['nob']).squeeze()
        nty_nc = np.asarray(ncDataset(nc4_path).variables['nty']).squeeze()
        uvr_nc = np.asarray(ncDataset(nc4_path).variables['uvr']).squeeze()
        vvr_nc = np.asarray(ncDataset(nc4_path).variables['vvr']).squeeze()
        pvr_nc = np.asarray(ncDataset(nc4_path).variables['pvr']).squeeze()
        tvr_nc = np.asarray(ncDataset(nc4_path).variables['tvr']).squeeze()
        dvr_nc = np.asarray(ncDataset(nc4_path).variables['dvr']).squeeze()
        # will include typ later, need to build function to unpack binary
        latTen = torch.from_numpy(lat_nc)
        lonTen = torch.from_numpy(lon_nc)
        preTen = torch.from_numpy(pre_nc)
        timTen = torch.from_numpy(tim_nc)
        uwdTen = torch.from_numpy(uwd_nc)
        vwdTen = torch.from_numpy(vwd_nc)
        nobTen = torch.from_numpy(nob_nc)
        ntyTen = torch.from_numpy(nty_nc)
        uvrTen = torch.from_numpy(uvr_nc)
        vvrTen = torch.from_numpy(vvr_nc)
        pvrTen = torch.from_numpy(pvr_nc)
        tvrTen = torch.from_numpy(tvr_nc)
        dvrTen = torch.from_numpy(dvr_nc)
        return (latTen, lonTen, preTen, timTen, uwdTen, vwdTen, nobTen, ntyTen, uvrTen, vvrTen, pvrTen, tvrTen, dvrTen)


# LabelDataset: Dataset class for label tensors
#
# To use:
# > import numpy as np
# > from netCDF4 import Dataset as ncDataset
# > import torch
# > from torch.utils.data import Dataset, DataLoader
# > import glob
# >
# > labelDataset = LabelDataset(dataDir='/path/to/data/directory/')
# > labelLoader = DataLoader(labelDataset, batch_size=None, shuffle=False, num_workers=0)
class LabelDataset(Dataset):
    def __init__(self, dataDir):
        self.nc4_path = dataDir + "/" if dataDir[-1] != "/" else dataDir
        date_list = glob.glob(self.nc4_path + "*")
        self.data = []
        for date_path in date_list:
            date_name = date_path.split("/")[-1]
            for nc4_path in glob.glob(date_path + "/*_superobs_labels_ERA5.nc"):
                self.data.append([nc4_path, date_name])
        print(len(self.data), " files of ", len(date_list), " directories loaded")        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        nc4_path, class_name = self.data[idx]
        uwd_nc = np.asarray(ncDataset(nc4_path).variables['uwd']).squeeze()
        vwd_nc = np.asarray(ncDataset(nc4_path).variables['vwd']).squeeze()
        uwdTen = torch.from_numpy(uwd_nc)
        vwdTen = torch.from_numpy(vwd_nc)
        return (uwdTen, vwdTen)
