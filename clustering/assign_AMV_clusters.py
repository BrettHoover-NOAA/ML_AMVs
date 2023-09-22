#
#  reads AMV data from process_AMVs_from_BUFR.py (netCDF obs-space fields), assigns a clusterID to obs, and
#  fills clusterID field in netCDF file with appropriate values
#
# import all dependencies
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from netCDF4 import Dataset
import pyproj
import libpysal
from sklearn.neighbors import NearestNeighbors
import yaml
import argparse

#
# internal functions:
#
# parse_yaml: Given a YAML file and a requested tile, parse inputs and return variables
#
# INPUTS:
#    yamlFile: name of YAML file (string)
#    tileName: name of tile (string, "Tile_N" format)
#
# OUTPUTS:
#    tuple containing all tile data (see below for details)
#
# DEPENDENCIES:
#    yaml 
def parse_yaml(yamlFile, tileName):
    # YAML entries are nested for each Tile:
    # Tile_1:
    #      tileValue: ........................ numerical index of tile (int)
    #      tilePressureMin: .................. minimum pressure on tile (float, Pa)
    #      tilePressureMax: .................. maximum pressure on tile (float, Pa)
    #      tileTimeMin: ...................... minimum time on tile (float, fractional hours)
    #      tileTimeMax: ...................... maximum time on tile (float, fractional hours)
    #      haloPressureMin: .................. minimum pressure on halo (float, Pa)
    #      haloPressureMax: .................. maximum pressure on halo (float, Pa)
    #      haloTimeMin: ...................... minimum time on halo (float, fractional hours)
    #      haloTimeMax: ...................... maximum time on halo (float, fractional hours)
    # Tile_2:
    #      tileValue:
    #      ...
    with open(yamlFile, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as YAMLError:
            parsed_yaml = None
            print(f'YAMLError: {YAMLError}')
        if parsed_yaml is not None:
            # Select tile
            try:
                tileDict = parsed_yaml[tileName]
            except KeyError as MissingTileError:
                tileDict = None
                print(f'MissingTileError: {MissingTileError}')
            # Extract tile data 
            try:
                tileValue = tileDict['tileValue']
                tilePressureMin = tileDict['tilePressureMin']
                tilePressureMax = tileDict['tilePressureMax']
                tileTimeMin = tileDict['tileTimeMin']
                tileTimeMax = tileDict['tileTimeMax']
                haloPressureMin = tileDict['haloPressureMin']
                haloPressureMax = tileDict['haloPressureMax']
                haloTimeMin = tileDict['haloTimeMin']
                haloTimeMax = tileDict['haloTimeMax']
            except KeyError as MissingDataError:
                tileValue = None
                tilePressureMin = None
                tilePressureMax = None
                tileTimeMin = None
                tileTimeMax = None
                haloPressureMin = None
                haloPressureMax = None
                haloTimeMin = None
                haloTimeMax = None
                print(f'MissingDataError: {MissingDataError}')
    # return tile data
    return (tileValue, tilePressureMin, tilePressureMax, tileTimeMin, tileTimeMax,
            haloPressureMin, haloPressureMax, haloTimeMin, haloTimeMax)


# define_clusters: given a geopandas dataframe with shapely POINT geometry in EPSG:4087 format,
#                  performs a series of operations to cluster observations that collectively share
#                  neighborbood relationships in 2D distance, pressure, time, u-component wind, and
#                  v-component wind within defined threshold maximum values
#
# INPUTS:
#
# gdfE: geopandas dataframe containing observations to operate on, MUST have a gdfE['geometry'] in
#       shapely POINT values with crs=EPSG:4087 (cylindrical equidistant projection, differences in m).
#       Presumed dataframe columns
#         geometry (shapely.geometry.Point())
#         lat (float, degrees)
#         lon (float, degrees)
#         pre (float, hPa)
#         tim (float, fractional hours relative to analysis time, i.e. location in time-window)
#         uob (float, m/s)
#         vob (float, m/s)
#         obIndex (integer, index in full ob vector)
#         clusterIndex (integer, initialized to all -1 values)
#         onTile (integer, ==1 if observation is on tile, ==-1 if not)
# thresholdDistance: maximum allowable distance (m) between observations to define neighborhood
# thresholdPressure: maximum allowable pressure difference (hPa) between observations to define neighborhood
# thresholdTime: maximum allowable time difference (hrs) between observations to define neighborhood
# thresholdUwnd: maximum allowable u-wind difference (m/s) between observations to define neighborhood
# thresholdVwnd: maximum allowable v-wind difference (m/s) between observations to define neighborhood
#
# OUTPUTS:
#
# gdfE2: geopandas dataframe with 'clusterIndex' filled, defining each cluster in dataset
#
# DEPENDENCIES:
#
# numpy
# geopandas
# pandas
# shapely.geometry.Point
# pysal
# libpysal
# sklearn.neighbors.NearestNeighbors
def define_clusters(gdfE, thresholdDistance, thresholdPressure, thresholdTime, thresholdUwnd, thresholdVwnd):
    #########################################################################################################
    #
    # Compute 2D distance relationships between observations using libpysal weights
    #
    # use libpysal.weights.distance.DistanceBand.from_dataframe() to compute a system of weights
    # identifying observations that are within thresholdDistance of each other. Warnings are silenced
    # but with silence_warnings=False you will see warnings of the form:
    #
    # "UserWarning: The weights matrix is not fully connected:"
    # " There are {n1} disconnected components."
    # " There are {n2} islands with ids: ..."
    #
    # We will use both the disconnected components information and the islands information when
    # processing observations, but we can squelch the warnings since we don't need to report this
    # information to the screen.
    w = libpysal.weights.distance.DistanceBand.from_dataframe(df=gdfE,
                                                              threshold=thresholdDistance,
                                                              binary=True,
                                                              ids=None,
                                                              build_sp=True,
                                                              silence_warnings=True,
                                                              distance_metric='euclidean',
                                                              radius=None)
    # We can immediately trim the islands off of gdfE, these are observations with no neighbors
    # withing thresholdDistance and as a result will definitively be single-member clusters. These
    # will be appended back into gdfE at the end and assigned their own clusterIndex values, but
    # for now we can remove them from the rest of the algorithm's operations.
    # Create separate geopandas dataframe to contain islands
    gdfE_islands = gdfE.iloc[w.islands]
    # Reset the index numbering of gdfE_islands
    gdfE_islands = gdfE_islands.reset_index() # moves existing index-values to 'index'
    # Drop gdfE_islands indices from gdfE
    gdfE=gdfE.drop(index=w.islands)
    # Reset the index numbering of gdfE
    gdfE = gdfE.reset_index() # moves existing index-values to 'index'
    # Re-run libpysal.weights.distance.DistanceBand.from_dataframe() on gdfE without islands,
    # which will yield data only across non-island data
    w = libpysal.weights.distance.DistanceBand.from_dataframe(df=gdfE,
                                                              threshold=thresholdDistance,
                                                              binary=True,
                                                              ids=None,
                                                              build_sp=True,
                                                              silence_warnings=True,
                                                              distance_metric='euclidean',
                                                              radius=None)
    # w generates information on component-groups, which are collectively interconnected observations that
    # are not connected to any observations outside of the component-group. For example, a set of 5
    # observations that share distance < thresholdDistance relationships between each other, but none of
    # those 5 observations has a similar relationship outside of the group, would be a component-group. These
    # component-groups serve as natural dividing-lines between the observations in the dataset: knowing that
    # no observations within a component-group are connected to observations outside of the component-group,
    # clusters can be searched-for within each component-group individually. We will use this information
    # to break up the task of searching for neighbors in other spaces (pressure, time, u-wind, v-wind) only
    # within a component-group.
    #
    # Assign the w.component_label value for each observation to gdfE as 'compGroup'
    gdfE = gdfE.assign(compGroup=w.component_labels)
    # Assign a -1 value to 'compGroup' in all members in gdfE_islands to flag them as islands
    gdfE_islands = gdfE_islands.assign(compGroup=-1)
    #########################################################################################################
    #
    # Search for clusters within each component-group of the data
    #
    # Define a cluster index value
    clusterID = -1  # first valid cluster will increment, so clusterIndex will begin at 0
    # Loop over component-groups
    for ic in np.unique(gdfE['compGroup'].values):
        # Extract all component-neighbors from a component-group and place into a gdfEsub geopandas dataframe
        gdfEsub = gdfE.loc[gdfE['compGroup']==ic]
        # Compute proximity-neighbor lists among members of component-group
        # pressure
        neighPres = NearestNeighbors(radius=thresholdPressure)
        neighPres.fit(np.reshape(gdfEsub['pre'].values,(-1,1)))
        neighPresList = neighPres.radius_neighbors(np.reshape(gdfEsub['pre'].values,(-1,1)),return_distance=False)
        # time
        neighTime = NearestNeighbors(radius=thresholdTime)
        neighTime.fit(np.reshape(gdfEsub['tim'].values,(-1,1)))
        neighTimeList = neighTime.radius_neighbors(np.reshape(gdfEsub['tim'].values,(-1,1)),return_distance=False)
        # u-wind
        neighUwnd = NearestNeighbors(radius=thresholdUwnd)
        neighUwnd.fit(np.reshape(gdfEsub['uob'].values,(-1,1)))
        neighUwndList = neighUwnd.radius_neighbors(np.reshape(gdfEsub['uob'].values,(-1,1)),return_distance=False)
        # v-wind
        neighVwnd = NearestNeighbors(radius=thresholdVwnd)
        neighVwnd.fit(np.reshape(gdfEsub['vob'].values,(-1,1)))
        neighVwndList = neighVwnd.radius_neighbors(np.reshape(gdfEsub['vob'].values,(-1,1)),return_distance=False)
        # Reset index-values of gdfEsub
        gdfEsub = gdfEsub.reset_index()  # moves existing index-values to 'level_0'
        # Loop through subgroup
        for i in range(len(gdfEsub)):
            # Check if observation i is on-tile, if not, skip this loop
            # Check if observation i still needs to be assigned to a cluster, if not, skip this loop
            # NOTE: Including this if-check reduces the total number of clusters (i.e. generates larger
            #       clusters) and affects the standard-deviation statistics. It would be worth figuring
            #       out exactly *why* this change takes place.
            if (gdfEsub.iloc[i]['clusterIndex'] == -1) & (gdfEsub.iloc[i]['onTile'] == 1):
                # Define proximity, pressure, time, and u/v similarity neighbors among subgroup members
                # proxlist will never include observation i as a member
                proxlist = np.where(np.isin(gdfEsub['level_0'].values, w.neighbors[gdfEsub['level_0'].values[i]]))[0]
                # {pres,time,uwnd,vwnd}list will always include observation i as a member
                preslist = neighPresList[i]
                timelist = neighTimeList[i]
                uwndlist = neighUwndList[i]
                vwndlist = neighVwndList[i]
                # Define cluster members by intersection of all neighborhoods
                # since proxlist does not include observation i, it is dropped from the cluster here
                cluster = np.intersect1d(proxlist, preslist)
                cluster = np.intersect1d(cluster, timelist)
                cluster = np.intersect1d(cluster, uwndlist)
                cluster = np.intersect1d(cluster, vwndlist)
                # Add member i back into cluster
                cluster = np.append(i, cluster)
                # IF cluster contains more than one member (i.e. more than just observation i),
                # assign any member of cluster with a -1 (unassigned) clusterIndex to clusterID
                if np.size(cluster) > 1:
                    # increment clusterID
                    clusterID += 1
                    # assign clusterID to members of cluster
                    gdfEsub.iloc[cluster, gdfEsub.columns.get_loc('clusterIndex')] = np.where(gdfEsub.iloc[cluster, gdfEsub.columns.get_loc('clusterIndex')]==-1,
                                                                                              clusterID,
                                                                                              gdfEsub.iloc[cluster, gdfEsub.columns.get_loc('clusterIndex')])
        # Assign corresponding members of gdfE a clusterIndex value from gdfEsub, following clustering
        # on component-group 
        gdfE.set_index('index', inplace=True)
        gdfE.update(gdfEsub.set_index('index'))
        gdfE = gdfE.reset_index()  # to recover the initial structure
        # Reassert column types (integers tend to turn into floats after update)
        convert_dict = {'ob_idx': int,
                        'clusterIndex': int,
                        'compGroup': int
                       }
        gdfE = gdfE.astype(convert_dict)
        # Reassert gdfE crs as EPSG:4087 (tends to get dropped after update, further updates give warnings
        # about missing crs if this isn't done)
        gdfE = gdfE.set_crs("EPSG:4087")
    #########################################################################################################
    #
    # Assign clusterIndex values to single-member clusters in gdfE_islands and merge dataframes
    #
    # Assign each observation in gdfE_islands to its own cluster, incrementing from max(gdfE['clusterIndex'])
    gdfE_islands['clusterIndex'] = np.arange(max(gdfE['clusterIndex']) + 1, max(gdfE['clusterIndex']) + 1 + len(gdfE_islands))
    # Concatenate gdfE and gdfE_islands together to regenerate entire dataframe
    gdfE2 = gpd.GeoDataFrame(pd.concat([gdfE,gdfE_islands], ignore_index=True, verify_integrity=False, sort=False))
    # Sort gdfE2 by 'index' values and assert 'index' as dataframe index to put gdfE_islands back in-place in
    # the proper order to match input gdfE dataframe
    gdfE2 = gdfE2.sort_values('index')
    gdfE2.set_index('index', inplace=True)
    # Return gdfE2
    return gdfE2


#
# begin
#
if __name__ == "__main__":
    # define argparser for inputs
    parser = argparse.ArgumentParser(description='define full-path to data directory and name of ' +
                                                 'netCDF AMV file from process_AMVs_from_BUFR.py')
    parser.add_argument('anaDateTime', metavar='DTIME', type=str, help='YYYYMMDDHH of analysis')
    parser.add_argument('dataDir', metavar='DATADIR', type=str, help='full path to data directory')
    parser.add_argument('netcdfFileName', metavar='INFILE', type=str, help='netCDF AMV file')
    parser.add_argument('yamlFile', metavar='YAML', type=str, help='YAML file defining tiles')
    parser.add_argument('tileName', metavar='TILE', type=str, help='name of tile in YAML file to process')
    # parse arguments
    userInputs = parser.parse_args()
    # quality-control inputs: if userInputs.dataDir does not end in '/', append it
    dataDir = userInputs.dataDir + '/' if userInputs.dataDir[-1] != '/' else userInputs.dataDir
    netcdfFileName = userInputs.netcdfFileName
    hdl=Dataset(dataDir + netcdfFileName)
    # read raw (meta)data
    ob_pqc=np.asarray(hdl.variables['pqc']).squeeze()
    ob_typ=np.asarray(hdl.variables['typ']).squeeze()
    ob_pre=np.asarray(hdl.variables['pre']).squeeze()
    ob_lat=np.asarray(hdl.variables['lat']).squeeze()
    ob_lon=np.asarray(hdl.variables['lon']).squeeze()
    ob_tim=np.asarray(hdl.variables['tim']).squeeze()
    ob_uwd=np.asarray(hdl.variables['uwd']).squeeze()
    ob_vwd=np.asarray(hdl.variables['vwd']).squeeze()
    # fix longitudes to -180 to 180 format
    fix=np.where(ob_lon>180.)
    ob_lon[fix]=ob_lon[fix]-360.
    # create a vector to store clusterID values
    ob_cid = np.nan * np.ones(np.shape(ob_pre))
    # Pre-screening for pressure and time into discrete groups, then use DistanceBand grouping
    # and pressure/time NearestNeighbors against all obs to reach out and include obs outside
    # of initial screening. Perform similarity-based clustering on DistanceBand connected-groups,
    # which can include multiple proximity-based clusters but will contain no observations with
    # proximity-neighbors outside of the connected-group.
    #
    # define index of all qualifiable observations (meeting pre-QC and typ requirements)
    # try excluding type=240 (SWIR) and type=251 (VIS) which are not assimilated in GSI
    allidx=np.where((ob_pqc==1.)&(ob_typ>=240)&(ob_typ<=260))[0]#&(np.isin(ob_typ,[240,251])==False))[0]
    # define index of all searching observations (meeting pressure and time requirements in subset)
    thresDist = 100. * 1000.  # range of distance for clustering
    thresPres = 5000. # +/- range of pressure bin
    thresTime = 0.5  # +/- range of time bin
    thresUwnd = 5.0  # +/- range of u-wind differences for clustering
    thresVwnd = 5.0  # +/- range of v-wind differences for clustering
    # extract tile and halo data from yamlFile
    (tileValue, tilePresMin, tilePresMax, tileTimeMin, tileTimeMax,
     haloPresMin, haloPresMax, haloTimeMin, haloTimeMax) = parse_yaml(userInputs.yamlFile,
                                                                      userInputs.tileName)
    # generate a set of indexes for variables on-tile
    tileidx=np.intersect1d(allidx,np.where((ob_pre <= tilePresMax)&(ob_pre >= tilePresMin) &
                                           (ob_tim <= tileTimeMax)&(ob_tim >= tileTimeMin))[0])
    # expidx = index of total (tile+halo) "expanded search"
    expidx=np.intersect1d(allidx,np.where((ob_pre <= haloPresMax)&(ob_pre >= haloPresMin) &
                                          (ob_tim <= haloTimeMax)&(ob_tim >= haloTimeMin))[0])
    # halidx = index of halo, difference between expanded search and search
    halidx=np.setdiff1d(expidx,tileidx)
    print('{:d} observations in tile+halo'.format(np.size(expidx)))
    # construct a geopandas point dataset that contains all relevant ob-info
    point_list=[]
    tile_list=[]
    for i in expidx:
        point_list.append(Point(ob_lon[i],ob_lat[i]))
        if i in tileidx:
            tile_list.append(1)
        else:
            tile_list.append(-1)
    d = {'geometry': point_list, 
         'lat': list(ob_lat[expidx]),
         'lon': list(ob_lon[expidx]),
         'pre': list(ob_pre[expidx]),
         'tim': list(ob_tim[expidx]),
         'uob': list(ob_uwd[expidx]),
         'vob': list(ob_vwd[expidx]),
         'ob_idx': list(expidx),
         'clusterIndex': -1,  # placeholder, -1 == no assigned cluster
         'onTile': tile_list  # is observation on tile (True) or on halo (False)
        }
    gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    # transform gdf into cylindrical equidistant projection, where Point() units are in m
    gdfE = gdf.to_crs("EPSG:4087")
    # define clusters and assign clusterID values to each observation on-tile or within
    # halo, if the halo observation was assigned
    gdfE2 = define_clusters(gdfE, thresDist, thresPres, thresTime, thresUwnd, thresVwnd)
    # define gdfEonTile as only those gdfE2 rows that are on-tile
    gdfEonTile = gdfE2.loc[gdfE2['onTile']==1]
    gdfEonHalo = gdfE2.loc[gdfE2['onTile']==-1]
    # assign ob_cid values to all obs on tile+halo
    ob_cid[gdfEonTile['ob_idx'].values] = gdfEonTile['clusterIndex'].values
    ob_cid[gdfEonHalo['ob_idx'].values] = gdfEonHalo['clusterIndex'].values
    # write ob_cid to output file
    # output file-name is same as netcdfFileName, except tileName is included at the end prior to '.nc'
    ncOutFileName = dataDir + netcdfFileName[0:-3] + '_' + userInputs.tileName + '.nc'
    ncOut = Dataset( 
                      ncOutFileName  , # Dataset input: Output file name
                      'w'              , # Dataset input: Make file write-able
                      format='NETCDF4' , # Dataset input: Set output format to netCDF4
                    )
    # add dimensions
    ob  = ncOut.createDimension( 
                                 'ob' , # nc_out.createDimension input: Dimension name 
                                 None    # nc_out.createDimension input: Dimension size limit ("None" == unlimited)
                                 )
    # add variables
    lat = ncOut.createVariable(
                                  'lat'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    lon= ncOut.createVariable(
                                  'lon'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    pre = ncOut.createVariable(
                                  'pre'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    tim = ncOut.createVariable(
                                  'tim'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    uwd = ncOut.createVariable(
                                  'uwd'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    vwd = ncOut.createVariable(
                                  'vwd'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    typ = ncOut.createVariable(
                                  'typ'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    idx = ncOut.createVariable(
                                  'idx'       ,
                                  'i8'        ,
                                  ('ob')
                               )
    pqc = ncOut.createVariable(
                                  'pqc'       ,
                                  'f8'        ,
                                  ('ob')
                                )
    tid = ncOut.createVariable(   'tid'       ,
                                  'i8'        ,
                                  ('ob')
                              )
    cid = ncOut.createVariable(
                                  'cid'       ,
                                  'i8'        ,
                                  ('ob')
                                )
    # assign netCDF file attributes
    ncOut.tilePressureMinimum = tilePresMin
    ncOut.tilePressureMaximum = tilePresMax
    ncOut.tileTimeMinimum = tileTimeMin
    ncOut.tileTimeMaximum = tileTimeMax
    ncOut.haloPressureMinimum = haloPresMin
    ncOut.haloPressureMaximum = haloPresMax
    ncOut.haloTimeMinimum = haloTimeMin
    ncOut.haloTimeMaximum = haloTimeMax
    ncOut.analysisDateTime = userInputs.anaDateTime
    ncOut.tileName = userInputs.tileName
    ncOut.sourceFile = userInputs.netcdfFileName
    # fill netCDF file variables with only observations on-tile+halo (partial output)
    x = expidx
    lat[:]      = ob_lat[x]
    lon[:]      = ob_lon[x]
    pre[:]      = ob_pre[x]
    tim[:]      = ob_tim[x]
    uwd[:]      = ob_uwd[x]
    vwd[:]      = ob_vwd[x]
    typ[:]      = ob_typ[x]
    idx[:]      = x  # ob-index: retains order of observations from input netCDF file
    pqc[:]      = ob_pqc[x]
    tid[:]      = tileValue * np.ones((np.size(x),))  # tile-index: retains which tile observation was connected to for cluster assignment
    cid[:]      = ob_cid[x]
    # close ncOut
    ncOut.close()
#
# end
#
