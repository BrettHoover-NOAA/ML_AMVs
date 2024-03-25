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
        typ_nc = np.asarray(ncDataset(nc4_path).variables['typ']).squeeze()
        typeValues_nc = np.asarray(ncDataset(nc4_path).variables['typeValues']).squeeze()
        # unpack binary typ and convert to array of 1/0 values
        q = lambda x: format(x, 'b').rjust(np.size(typeValues_nc),'0')
        typBinaryStr = np.array(list(map(q, typ_nc)))
        typBinaryVal = np.array(list(map(list, typBinaryStr)),dtype=np.int16)
        # initialize has_* variables as zeros
        has_240 = np.zeros(np.shape(typ_nc))
        has_241 = np.zeros(np.shape(typ_nc))
        has_242 = np.zeros(np.shape(typ_nc))
        has_243 = np.zeros(np.shape(typ_nc))
        has_244 = np.zeros(np.shape(typ_nc))
        has_245 = np.zeros(np.shape(typ_nc))
        has_246 = np.zeros(np.shape(typ_nc))
        has_247 = np.zeros(np.shape(typ_nc))
        has_248 = np.zeros(np.shape(typ_nc))
        has_249 = np.zeros(np.shape(typ_nc))
        has_250 = np.zeros(np.shape(typ_nc))
        has_251 = np.zeros(np.shape(typ_nc))
        has_252 = np.zeros(np.shape(typ_nc))
        has_253 = np.zeros(np.shape(typ_nc))
        has_254 = np.zeros(np.shape(typ_nc))
        has_255 = np.zeros(np.shape(typ_nc))
        has_256 = np.zeros(np.shape(typ_nc))
        has_257 = np.zeros(np.shape(typ_nc))
        has_258 = np.zeros(np.shape(typ_nc))
        has_259 = np.zeros(np.shape(typ_nc))
        has_260 = np.zeros(np.shape(typ_nc))
        # for each present typeValue in file, assign 1 to corresponding has_* variable where type is present
        if np.isin(240,typeValues_nc):
            j = np.where(typeValues_nc==240)[0][0]
            has_240 = typBinaryVal[:,j].squeeze()
        if np.isin(241,typeValues_nc):
            j = np.where(typeValues_nc==241)[0][0]
            has_241 = typBinaryVal[:,j].squeeze()
        if np.isin(242,typeValues_nc):
            j = np.where(typeValues_nc==242)[0][0]
            has_242 = typBinaryVal[:,j].squeeze()
        if np.isin(243,typeValues_nc):
            j = np.where(typeValues_nc==243)[0][0]
            has_243 = typBinaryVal[:,j].squeeze()
        if np.isin(244,typeValues_nc):
            j = np.where(typeValues_nc==244)[0][0]
            has_244 = typBinaryVal[:,j].squeeze()
        if np.isin(245,typeValues_nc):
            j = np.where(typeValues_nc==245)[0][0]
            has_245 = typBinaryVal[:,j].squeeze()
        if np.isin(246,typeValues_nc):
            j = np.where(typeValues_nc==246)[0][0]
            has_246 = typBinaryVal[:,j].squeeze()
        if np.isin(247,typeValues_nc):
            j = np.where(typeValues_nc==247)[0][0]
            has_247 = typBinaryVal[:,j].squeeze()
        if np.isin(248,typeValues_nc):
            j = np.where(typeValues_nc==248)[0][0]
            has_248 = typBinaryVal[:,j].squeeze()
        if np.isin(249,typeValues_nc):
            j = np.where(typeValues_nc==249)[0][0]
            has_249 = typBinaryVal[:,j].squeeze()
        if np.isin(250,typeValues_nc):
            j = np.where(typeValues_nc==250)[0][0]
            has_250 = typBinaryVal[:,j].squeeze()
        if np.isin(251,typeValues_nc):
            j = np.where(typeValues_nc==251)[0][0]
            has_251 = typBinaryVal[:,j].squeeze()
        if np.isin(252,typeValues_nc):
            j = np.where(typeValues_nc==252)[0][0]
            has_252 = typBinaryVal[:,j].squeeze()
        if np.isin(253,typeValues_nc):
            j = np.where(typeValues_nc==253)[0][0]
            has_253 = typBinaryVal[:,j].squeeze()
        if np.isin(254,typeValues_nc):
            j = np.where(typeValues_nc==254)[0][0]
            has_254 = typBinaryVal[:,j].squeeze()
        if np.isin(255,typeValues_nc):
            j = np.where(typeValues_nc==255)[0][0]
            has_255 = typBinaryVal[:,j].squeeze()
        if np.isin(256,typeValues_nc):
            j = np.where(typeValues_nc==256)[0][0]
            has_256 = typBinaryVal[:,j].squeeze()
        if np.isin(257,typeValues_nc):
            j = np.where(typeValues_nc==257)[0][0]
            has_257 = typBinaryVal[:,j].squeeze()
        if np.isin(258,typeValues_nc):
            j = np.where(typeValues_nc==258)[0][0]
            has_258 = typBinaryVal[:,j].squeeze()
        if np.isin(259,typeValues_nc):
            j = np.where(typeValues_nc==259)[0][0]
            has_259 = typBinaryVal[:,j].squeeze()
        if np.isin(260,typeValues_nc):
            j = np.where(typeValues_nc==260)[0][0]
            has_260 = typBinaryVal[:,j].squeeze()
        # generate tensors from numpy arrays
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
        has_240Ten = torch.from_numpy(has_240)
        has_241Ten = torch.from_numpy(has_241)
        has_242Ten = torch.from_numpy(has_242)
        has_243Ten = torch.from_numpy(has_243)
        has_244Ten = torch.from_numpy(has_244)
        has_245Ten = torch.from_numpy(has_245)
        has_246Ten = torch.from_numpy(has_246)
        has_247Ten = torch.from_numpy(has_247)
        has_248Ten = torch.from_numpy(has_248)
        has_249Ten = torch.from_numpy(has_249)
        has_250Ten = torch.from_numpy(has_250)
        has_251Ten = torch.from_numpy(has_251)
        has_252Ten = torch.from_numpy(has_252)
        has_253Ten = torch.from_numpy(has_253)
        has_254Ten = torch.from_numpy(has_254)
        has_255Ten = torch.from_numpy(has_255)
        has_256Ten = torch.from_numpy(has_256)
        has_257Ten = torch.from_numpy(has_257)
        has_258Ten = torch.from_numpy(has_258)
        has_259Ten = torch.from_numpy(has_259)
        has_260Ten = torch.from_numpy(has_260)
        return (latTen, lonTen, preTen, timTen, uwdTen, vwdTen, nobTen, ntyTen, uvrTen, vvrTen, pvrTen, tvrTen, dvrTen,
                has_240Ten, has_241Ten, has_242Ten, has_243Ten, has_244Ten, has_245Ten, has_246Ten, has_247Ten, has_248Ten, has_249Ten,
                has_250Ten, has_251Ten, has_252Ten, has_253Ten, has_254Ten, has_255Ten, has_256Ten, has_257Ten, has_258Ten, has_259Ten,
                has_260Ten)


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
