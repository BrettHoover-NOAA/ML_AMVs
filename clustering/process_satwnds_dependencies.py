# bufr_query: make a stack of BUFR queries and return resultSet object containing data
#
# INPUTS:
#    bufrFile: full-path to BUFR file (string)
#    queryDict: dictionary with keys as query-strings and values as variable-names (dict)
#
# OUTPUTS:
#    r: resultSet object from BUFR query
#
# DEPENDENCIES:
#    bufr
def bufr_query(bufrFile, queryDict):
    import bufr
    # define a bufr.QuerySet() object
    q = bufr.QuerySet()
    # loop through queryDict key-value pairs and add them to QuerySet
    for key in list(queryDict.keys()):
       q.add(queryDict[key], key)
    # safe-open bufrFile and execute query
    with bufr.File(bufrFile) as f:
        r = f.execute(q)
    # return resultSet object
    return r


# process_NC005030: draws NC005030 observations (GOES LWIR AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005030(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # define internal functions
    #
    # pre_qc: perform pre-QC checks on input data, return indices of pass/fail obs
    #
    # INPUTS:
    #    pre: pressure, float(nobs,), hPa
    #    spd: wind speed, float(nobs,), m/s
    #    zen: zenith, angle float(nobs,), deg
    #    qin: quality indicator w/o forecast, int(nobs,), 0-100 index
    #    cov: coefficient of variation, float(nobs,), fractional coefficient
    #    exp: expected error, float(nobs,), m/s packed into 10. - 0.1*exp format
    #
    # OUTPUTS:
    #    idxPass: indices of observations passing all checks
    #    idxFail: indices of observations failing at least one check
    #
    # DEPENDENCIES:
    #    numpy
    def pre_qc(pre, spd, zen, qin, cov, exp):
        import numpy as np
        # generate vector of all indices and copy to idxPass
        idxAll = np.arange(np.size(pre))
        idxPass = np.copy(idxAll)
        # zenith angle check
        angMax = 68.
        checkPass = np.where(zen <= angMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail zenith angle check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # quality indicator check
        qiMin = 90
        qiMax = 100
        checkPass = np.where((qin >= qiMin) & (qin <= qiMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail quality indicator check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # pressure check
        preMin = 15000.
        checkPass = np.where(pre >= preMin)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail pressure check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # coefficient of variation check
        covMin = 0.04
        covMax = 0.50
        checkPass = np.where((cov >= covMin) & (cov <= covMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail coefficient of variation check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # exp-errnorm check
        expErrNorm = 100. * np.ones(np.size(exp,))
        speedExists = np.where(spd > 0.1)
        expErrNorm[speedExists] = np.divide(10. - 0.1*exp[speedExists], spd[speedExists])
        eeMax = 0.9
        checkPass = np.where(expErrNorm <= eeMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail exp-errnorm check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # define idxFail as all indices not in idxPass
        idxFail = np.setdiff1d(idxAll, idxPass)
        print('{:d} OBSERVATIONS FAIL ALL QC, {:d} PASS'.format(np.size(idxFail), np.size(idxPass)))
        # return
        return idxPass, idxFail
    
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    queryDict = {
                 'NC005030/PRLC[1]'     : 'pressure',               # (nobs,) dimension, there are multiple copies of PRLC but should all be identical
                 'NC005030/WSPD'        : 'windSpeed',              # (nobs,) dimension
                 'NC005030/SAZA'        : 'zenithAngle',            # (nobs,) dimension
                 'NC005030/AMVQIC/GNAPS': 'GNAPS',                  # (nobs,4) dimension, determines data stored in PCCF
                 'NC005030/AMVQIC/PCCF' : 'PCCF',                   # (nobs,4) dimension, QI w/o forecast at GNAPS==5, EE at GNAPS==7
                 'NC005030/AMVIVR{1}/CVWD' : 'coefficientOfVariation'  # (nobs,2) dimension, ioda-converter selects {1}, so I will do the same
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # initialize empty arrays for each pre-QC variable
    pressure               = np.asarray([])
    windSpeed              = np.asarray([])
    zenithAngle            = np.asarray([])
    qualityIndicator       = np.asarray([])
    expectedError          = np.asarray([])
    coefficientOfVariation = np.asarray([])
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate. This is done on a per-variable basis, because some
    # variables are packed together into multi-dimensional arrays and need to be split apart
    # to be sent to separate obs vectors. If you have a variable you want passed along to outputDict
    # that is one of these special cases, include it as a special case below.
    #
    # these are all handled as appends to an initially empty obs vector, since you could have multiple
    # individual queries point to the same output variable, e.g.: latitudes from multiple BUFR tanks
    # all pulled into a single 'latitude' obs vector.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'pressure':
            pressure = np.append(pressure, x)
            if 'pressure' in list(returnDict.values()):
                outputDict['pressure'] = np.append(outputDict['pressure'], x)
        elif mergedDict[key] == 'windSpeed':
            windSpeed = np.append(windSpeed, x)
            if 'windSpeed' in list(returnDict.values()):
                outputDict['windSpeed'] = np.append(outputDict['windSpeed'], x)
        elif mergedDict[key] == 'zenithAngle':
            zenithAngle = np.append(zenithAngle, x)
            if 'zenithAngle' in list(returnDict.values()):
                outputDict['zenithAngle'] = np.append(outputDict['zenithAngle'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAPS')  # GNAPS value determines which column contains which confidence value
            qi_idx = -9
            ee_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 5.:
                    qi_idx = j
                if np.mean(g[:,j]) == 7.:
                    ee_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
            if ee_idx >= 0:
                expectedError = np.append(expectedError, x[:,ee_idx].squeeze())
                #if 'expectedError' in list(returnDict.values()):
                #    outputDict['expectedError'] = np.append(outputDict['expectedError'], x[:,ee_idx].squeeze())
            else:
                print('EE index not found!')
                #if 'expectedError' in list(returnDict.values()):
                #    outputDict['expectedError'] = np.append(outputDict['expectedError'], -9. * x[:,0].squeeze())  # dummy-values
        elif mergedDict[key] == 'coefficientOfVariation':
            coefficientOfVariation = np.append(coefficientOfVariation, x.squeeze())
            if 'coefficientOfVariation' in list(returnDict.values()):
                outputDict['coefficientOfVariation'] = np.append(outputDict['coefficientOfVariation'], x.squeeze())
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here if they aren't already a pre-QC variable in queryDict
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # perform pre-QC checks
    idxPass, idxFail = pre_qc(pre=pressure,
                              spd=windSpeed,
                              zen=zenithAngle,
                              qin=qualityIndicator,
                              cov=coefficientOfVariation,
                              exp=expectedError)
    print(np.size(idxFail), np.size(idxPass))
    # create a preQC variable with 1==pass, -1==fail
    preQC = -1 * np.ones((np.size(idxPass) + np.size(idxFail),), dtype='int')
    preQC[idxPass] = 1
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign all values to 245
    obType = 245 * np.ones(np.shape(preQC), dtype='int')
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005031: draws NC005031 observations (GOES WVDL AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005031(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # define internal functions
    #
    # pre_qc: perform pre-QC checks on input data, return indices of pass/fail obs
    #
    # INPUTS:
    #    pre: pressure, float(nobs,), hPa
    #    spd: wind speed, float(nobs,), m/s
    #    zen: zenith, angle float(nobs,), deg
    #    qin: quality indicator w/o forecast, int(nobs,), 0-100 index
    #    exp: expected error, float(nobs,), m/s packed into 10. - 0.1*exp format
    #
    # OUTPUTS:
    #    idxPass: indices of observations passing all checks
    #    idxFail: indices of observations failing at least one check
    #
    # DEPENDENCIES:
    #    numpy
    def pre_qc(pre, spd, zen, qin, exp):
        import numpy as np
        # generate vector of all indices and copy to idxPass
        idxAll = np.arange(np.size(pre))
        idxPass = np.copy(idxAll)
        # zenith angle check
        angMax = 68.
        checkPass = np.where(zen <= angMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail zenith angle check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # quality indicator check
        qiMin = 90
        qiMax = 100
        checkPass = np.where((qin >= qiMin) & (qin <= qiMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail quality indicator check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # pressure check
        preMin = 15000.
        checkPass = np.where(pre >= preMin)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail pressure check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # exp-errnorm check
        expErrNorm = 100. * np.ones(np.size(exp,))
        speedExists = np.where(spd > 0.1)
        expErrNorm[speedExists] = np.divide(10. - 0.1*exp[speedExists], spd[speedExists])
        eeMax = 0.9
        checkPass = np.where(expErrNorm <= eeMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail exp-errnorm check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # define idxFail as all indices not in idxPass
        idxFail = np.setdiff1d(idxAll, idxPass)
        print('{:d} OBSERVATIONS FAIL ALL QC, {:d} PASS'.format(np.size(idxFail), np.size(idxPass)))
        # return
        return idxPass, idxFail
    
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    queryDict = {
                 'NC005031/PRLC[1]'     : 'pressure',               # (nobs,) dimension, there are multiple copies of PRLC but should all be identical
                 'NC005031/WSPD'        : 'windSpeed',              # (nobs,) dimension
                 'NC005031/SAZA'        : 'zenithAngle',            # (nobs,) dimension
                 'NC005031/AMVQIC/GNAPS': 'GNAPS',                  # (nobs,4) dimension, determines data stored in PCCF
                 'NC005031/AMVQIC/PCCF' : 'PCCF'                   # (nobs,4) dimension, QI w/o forecast at GNAPS==5, EE at GNAPS==7
               }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # initialize empty arrays for each pre-QC variable
    pressure               = np.asarray([])
    windSpeed              = np.asarray([])
    zenithAngle            = np.asarray([])
    qualityIndicator       = np.asarray([])
    expectedError          = np.asarray([])
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate. This is done on a per-variable basis, because some
    # variables are packed together into multi-dimensional arrays and need to be split apart
    # to be sent to separate obs vectors. If you have a variable you want passed along to outputDict
    # that is one of these special cases, include it as a special case below.
    #
    # these are all handled as appends to an initially empty obs vector, since you could have multiple
    # individual queries point to the same output variable, e.g.: latitudes from multiple BUFR tanks
    # all pulled into a single 'latitude' obs vector.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'pressure':
            pressure = np.append(pressure, x)
            if 'pressure' in list(returnDict.values()):
                outputDict['pressure'] = np.append(outputDict['pressure'], x)
        elif mergedDict[key] == 'windSpeed':
            windSpeed = np.append(windSpeed, x)
            if 'windSpeed' in list(returnDict.values()):
                outputDict['windSpeed'] = np.append(outputDict['windSpeed'], x)
        elif mergedDict[key] == 'zenithAngle':
            zenithAngle = np.append(zenithAngle, x)
            if 'zenithAngle' in list(returnDict.values()):
                outputDict['zenithAngle'] = np.append(outputDict['zenithAngle'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAPS')  # GNAPS value determines which column contains which confidence value
            qi_idx = -9
            ee_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 5.:
                    qi_idx = j
                if np.mean(g[:,j]) == 7.:
                    ee_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
            if ee_idx >= 0:
                expectedError = np.append(expectedError, x[:,ee_idx].squeeze())
                #if 'expectedError' in list(returnDict.values()):
                #    outputDict['expectedError'] = np.append(outputDict['expectedError'], x[:,ee_idx].squeeze())
            else:
                print('EE index not found!')
                #if 'expectedError' in list(returnDict.values()):
                #    outputDict['expectedError'] = np.append(outputDict['expectedError'], -9. * x[:,0].squeeze())  # dummy-values
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here if they aren't already a pre-QC variable in queryDict
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # perform pre-QC checks
    idxPass, idxFail = pre_qc(pre=pressure,
                              spd=windSpeed,
                              zen=zenithAngle,
                              qin=qualityIndicator,
                              exp=expectedError)
    print(np.size(idxFail), np.size(idxPass))
    # create a preQC variable with 1==pass, -1==fail
    preQC = -1 * np.ones((np.size(idxPass) + np.size(idxFail),), dtype='int')
    preQC[idxPass] = 1
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign all values to 247
    obType = 247 * np.ones(np.shape(preQC), dtype='int')
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005032: draws NC005032 observations (GOES VIS AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005032(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # define internal functions
    #
    # pre_qc: perform pre-QC checks on input data, return indices of pass/fail obs
    #
    # INPUTS:
    #    pre: pressure, float(nobs,), hPa
    #    spd: wind speed, float(nobs,), m/s
    #    zen: zenith, angle float(nobs,), deg
    #    qin: quality indicator w/o forecast, int(nobs,), 0-100 index
    #    cov: coefficient of variation, float(nobs,), fractional coefficient
    #    exp: expected error, float(nobs,), m/s packed into 10. - 0.1*exp format
    #
    # OUTPUTS:
    #    idxPass: indices of observations passing all checks
    #    idxFail: indices of observations failing at least one check
    #
    # DEPENDENCIES:
    #    numpy
    def pre_qc(pre, spd, zen, qin, cov, exp):
        import numpy as np
        # generate vector of all indices and copy to idxPass
        idxAll = np.arange(np.size(pre))
        idxPass = np.copy(idxAll)
        # zenith angle check
        angMax = 68.
        checkPass = np.where(zen <= angMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail zenith angle check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # quality indicator check
        qiMin = 90
        qiMax = 100
        checkPass = np.where((qin >= qiMin) & (qin <= qiMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail quality indicator check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # pressure check (preMin=70000. cutoff for VIS winds)
        preMin = 70000.
        checkPass = np.where(pre >= preMin)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail pressure check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # coefficient of variation check
        covMin = 0.04
        covMax = 0.50
        checkPass = np.where((cov >= covMin) & (cov <= covMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail coefficient of variation check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # exp-errnorm check
        expErrNorm = 100. * np.ones(np.size(exp,))
        speedExists = np.where(spd > 0.1)
        expErrNorm[speedExists] = np.divide(10. - 0.1*exp[speedExists], spd[speedExists])
        eeMax = 0.9
        checkPass = np.where(expErrNorm <= eeMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail exp-errnorm check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # define idxFail as all indices not in idxPass
        idxFail = np.setdiff1d(idxAll, idxPass)
        print('{:d} OBSERVATIONS FAIL ALL QC, {:d} PASS'.format(np.size(idxFail), np.size(idxPass)))
        # return
        return idxPass, idxFail
    
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    queryDict = {
                 'NC005032/PRLC[1]'     : 'pressure',               # (nobs,) dimension, there are multiple copies of PRLC but should all be identical
                 'NC005032/WSPD'        : 'windSpeed',              # (nobs,) dimension
                 'NC005032/SAZA'        : 'zenithAngle',            # (nobs,) dimension
                 'NC005032/AMVQIC/GNAPS': 'GNAPS',                  # (nobs,4) dimension, determines data stored in PCCF
                 'NC005032/AMVQIC/PCCF' : 'PCCF',                   # (nobs,4) dimension, QI w/o forecast at GNAPS==5, EE at GNAPS==7
                 'NC005032/AMVIVR{1}/CVWD' : 'coefficientOfVariation'  # (nobs,2) dimension, ioda-converter selects {1}, so I will do the same
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # initialize empty arrays for each pre-QC variable
    pressure               = np.asarray([])
    windSpeed              = np.asarray([])
    zenithAngle            = np.asarray([])
    qualityIndicator       = np.asarray([])
    expectedError          = np.asarray([])
    coefficientOfVariation = np.asarray([])
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate. This is done on a per-variable basis, because some
    # variables are packed together into multi-dimensional arrays and need to be split apart
    # to be sent to separate obs vectors. If you have a variable you want passed along to outputDict
    # that is one of these special cases, include it as a special case below.
    #
    # these are all handled as appends to an initially empty obs vector, since you could have multiple
    # individual queries point to the same output variable, e.g.: latitudes from multiple BUFR tanks
    # all pulled into a single 'latitude' obs vector.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'pressure':
            pressure = np.append(pressure, x)
            if 'pressure' in list(returnDict.values()):
                outputDict['pressure'] = np.append(outputDict['pressure'], x)
        elif mergedDict[key] == 'windSpeed':
            windSpeed = np.append(windSpeed, x)
            if 'windSpeed' in list(returnDict.values()):
                outputDict['windSpeed'] = np.append(outputDict['windSpeed'], x)
        elif mergedDict[key] == 'zenithAngle':
            zenithAngle = np.append(zenithAngle, x)
            if 'zenithAngle' in list(returnDict.values()):
                outputDict['zenithAngle'] = np.append(outputDict['zenithAngle'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAPS')  # GNAPS value determines which column contains which confidence value
            qi_idx = -9
            ee_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 5.:
                    qi_idx = j
                if np.mean(g[:,j]) == 7.:
                    ee_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
            if ee_idx >= 0:
                expectedError = np.append(expectedError, x[:,ee_idx].squeeze())
                #if 'expectedError' in list(returnDict.values()):
                #    outputDict['expectedError'] = np.append(outputDict['expectedError'], x[:,ee_idx].squeeze())
            else:
                print('EE index not found!')
                #if 'expectedError' in list(returnDict.values()):
                #    outputDict['expectedError'] = np.append(outputDict['expectedError'], -9. * x[:,0].squeeze())  # dummy-values
        elif mergedDict[key] == 'coefficientOfVariation':
            coefficientOfVariation = np.append(coefficientOfVariation, x.squeeze())
            if 'coefficientOfVariation' in list(returnDict.values()):
                outputDict['coefficientOfVariation'] = np.append(outputDict['coefficientOfVariation'], x.squeeze())
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here if they aren't already a pre-QC variable in queryDict
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # perform pre-QC checks
    idxPass, idxFail = pre_qc(pre=pressure,
                              spd=windSpeed,
                              zen=zenithAngle,
                              qin=qualityIndicator,
                              cov=coefficientOfVariation,
                              exp=expectedError)
    print(np.size(idxFail), np.size(idxPass))
    # create a preQC variable with 1==pass, -1==fail
    preQC = -1 * np.ones((np.size(idxPass) + np.size(idxFail),), dtype='int')
    preQC[idxPass] = 1
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign all values to 251
    obType = 251 * np.ones(np.shape(preQC), dtype='int')
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005034: draws NC005034 observations (GOES WVCT AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005034(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # define internal functions
    #
    # pre_qc: perform pre-QC checks on input data, return indices of pass/fail obs
    #
    # INPUTS:
    #    pre: pressure, float(nobs,), hPa
    #    spd: wind speed, float(nobs,), m/s
    #    zen: zenith, angle float(nobs,), deg
    #    qin: quality indicator w/o forecast, int(nobs,), 0-100 index
    #    cov: coefficient of variation, float(nobs,), fractional coefficient
    #    exp: expected error, float(nobs,), m/s packed into 10. - 0.1*exp format
    #
    # OUTPUTS:
    #    idxPass: indices of observations passing all checks
    #    idxFail: indices of observations failing at least one check
    #
    # DEPENDENCIES:
    #    numpy
    def pre_qc(pre, spd, zen, qin, cov, exp):
        import numpy as np
        # generate vector of all indices and copy to idxPass
        idxAll = np.arange(np.size(pre))
        idxPass = np.copy(idxAll)
        # zenith angle check
        angMax = 68.
        checkPass = np.where(zen <= angMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail zenith angle check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # quality indicator check
        qiMin = 90
        qiMax = 100
        checkPass = np.where((qin >= qiMin) & (qin <= qiMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail quality indicator check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # pressure check (preMin=15000., preMax=30000. cutoff for WVCT winds)
        preMin = 15000.
        preMax = 30000.
        checkPass = np.where((pre >= preMin) & (pre <= preMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail pressure check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # coefficient of variation check
        covMin = 0.04
        covMax = 0.50
        checkPass = np.where((cov >= covMin) & (cov <= covMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail coefficient of variation check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # exp-errnorm check
        expErrNorm = 100. * np.ones(np.size(exp,))
        speedExists = np.where(spd > 0.1)
        expErrNorm[speedExists] = np.divide(10. - 0.1*exp[speedExists], spd[speedExists])
        eeMax = 0.9
        checkPass = np.where(expErrNorm <= eeMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail exp-errnorm check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # define idxFail as all indices not in idxPass
        idxFail = np.setdiff1d(idxAll, idxPass)
        print('{:d} OBSERVATIONS FAIL ALL QC, {:d} PASS'.format(np.size(idxFail), np.size(idxPass)))
        # return
        return idxPass, idxFail
    
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    queryDict = {
                 'NC005034/PRLC[1]'     : 'pressure',               # (nobs,) dimension, there are multiple copies of PRLC but should all be identical
                 'NC005034/WSPD'        : 'windSpeed',              # (nobs,) dimension
                 'NC005034/SAZA'        : 'zenithAngle',            # (nobs,) dimension
                 'NC005034/AMVQIC/GNAPS': 'GNAPS',                  # (nobs,4) dimension, determines data stored in PCCF
                 'NC005034/AMVQIC/PCCF' : 'PCCF',                   # (nobs,4) dimension, QI w/o forecast at GNAPS==5, EE at GNAPS==7
                 'NC005034/AMVIVR{1}/CVWD' : 'coefficientOfVariation'  # (nobs,2) dimension, ioda-converter selects {1}, so I will do the same
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # initialize empty arrays for each pre-QC variable
    pressure               = np.asarray([])
    windSpeed              = np.asarray([])
    zenithAngle            = np.asarray([])
    qualityIndicator       = np.asarray([])
    expectedError          = np.asarray([])
    coefficientOfVariation = np.asarray([])
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate. This is done on a per-variable basis, because some
    # variables are packed together into multi-dimensional arrays and need to be split apart
    # to be sent to separate obs vectors. If you have a variable you want passed along to outputDict
    # that is one of these special cases, include it as a special case below.
    #
    # these are all handled as appends to an initially empty obs vector, since you could have multiple
    # individual queries point to the same output variable, e.g.: latitudes from multiple BUFR tanks
    # all pulled into a single 'latitude' obs vector.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'pressure':
            pressure = np.append(pressure, x)
            if 'pressure' in list(returnDict.values()):
                outputDict['pressure'] = np.append(outputDict['pressure'], x)
        elif mergedDict[key] == 'windSpeed':
            windSpeed = np.append(windSpeed, x)
            if 'windSpeed' in list(returnDict.values()):
                outputDict['windSpeed'] = np.append(outputDict['windSpeed'], x)
        elif mergedDict[key] == 'zenithAngle':
            zenithAngle = np.append(zenithAngle, x)
            if 'zenithAngle' in list(returnDict.values()):
                outputDict['zenithAngle'] = np.append(outputDict['zenithAngle'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAPS')  # GNAPS value determines which column contains which confidence value
            qi_idx = -9
            ee_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 5.:
                    qi_idx = j
                if np.mean(g[:,j]) == 7.:
                    ee_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
            if ee_idx >= 0:
                expectedError = np.append(expectedError, x[:,ee_idx].squeeze())
                #if 'expectedError' in list(returnDict.values()):
                #    outputDict['expectedError'] = np.append(outputDict['expectedError'], x[:,ee_idx].squeeze())
            else:
                print('EE index not found!')
                #if 'expectedError' in list(returnDict.values()):
                #    outputDict['expectedError'] = np.append(outputDict['expectedError'], -9. * x[:,0].squeeze())  # dummy-values
        elif mergedDict[key] == 'coefficientOfVariation':
            coefficientOfVariation = np.append(coefficientOfVariation, x.squeeze())
            if 'coefficientOfVariation' in list(returnDict.values()):
                outputDict['coefficientOfVariation'] = np.append(outputDict['coefficientOfVariation'], x.squeeze())
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here if they aren't already a pre-QC variable in queryDict
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # perform pre-QC checks
    idxPass, idxFail = pre_qc(pre=pressure,
                              spd=windSpeed,
                              zen=zenithAngle,
                              qin=qualityIndicator,
                              cov=coefficientOfVariation,
                              exp=expectedError)
    print(np.size(idxFail), np.size(idxPass))
    # create a preQC variable with 1==pass, -1==fail
    preQC = -1 * np.ones((np.size(idxPass) + np.size(idxFail),), dtype='int')
    preQC[idxPass] = 1
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign all values to 246
    obType = 246 * np.ones(np.shape(preQC), dtype='int')
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005039: draws NC005039 observations (GOES SWIR AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005039(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # define internal functions
    #
    # pre_qc: perform pre-QC checks on input data, return indices of pass/fail obs
    #
    # INPUTS:
    #    pre: pressure, float(nobs,), hPa
    #    spd: wind speed, float(nobs,), m/s
    #    zen: zenith, angle float(nobs,), deg
    #    qin: quality indicator w/o forecast, int(nobs,), 0-100 index
    #    cov: coefficient of variation, float(nobs,), fractional coefficient
    #    exp: expected error, float(nobs,), m/s packed into 10. - 0.1*exp format
    #
    # OUTPUTS:
    #    idxPass: indices of observations passing all checks
    #    idxFail: indices of observations failing at least one check
    #
    # DEPENDENCIES:
    #    numpy
    def pre_qc(pre, spd, zen, qin, cov, exp):
        import numpy as np
        # generate vector of all indices and copy to idxPass
        idxAll = np.arange(np.size(pre))
        idxPass = np.copy(idxAll)
        # zenith angle check
        angMax = 68.
        checkPass = np.where(zen <= angMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail zenith angle check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # quality indicator check
        qiMin = 90
        qiMax = 100
        checkPass = np.where((qin >= qiMin) & (qin <= qiMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail quality indicator check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # pressure check
        preMin = 15000.
        checkPass = np.where(pre >= preMin)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail pressure check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # coefficient of variation check
        covMin = 0.04
        covMax = 0.50
        checkPass = np.where((cov >= covMin) & (cov <= covMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail coefficient of variation check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # exp-errnorm check
        expErrNorm = 100. * np.ones(np.size(exp,))
        speedExists = np.where(spd > 0.1)
        expErrNorm[speedExists] = np.divide(10. - 0.1*exp[speedExists], spd[speedExists])
        eeMax = 0.9
        checkPass = np.where(expErrNorm <= eeMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail exp-errnorm check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # define idxFail as all indices not in idxPass
        idxFail = np.setdiff1d(idxAll, idxPass)
        print('{:d} OBSERVATIONS FAIL ALL QC, {:d} PASS'.format(np.size(idxFail), np.size(idxPass)))
        # return
        return idxPass, idxFail
    
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    queryDict = {
                 'NC005039/PRLC[1]'     : 'pressure',               # (nobs,) dimension, there are multiple copies of PRLC but should all be identical
                 'NC005039/WSPD'        : 'windSpeed',              # (nobs,) dimension
                 'NC005039/SAZA'        : 'zenithAngle',            # (nobs,) dimension
                 'NC005039/AMVQIC/GNAPS': 'GNAPS',                  # (nobs,4) dimension, determines data stored in PCCF
                 'NC005039/AMVQIC/PCCF' : 'PCCF',                   # (nobs,4) dimension, QI w/o forecast at GNAPS==5, EE at GNAPS==7
                 'NC005039/AMVIVR{1}/CVWD' : 'coefficientOfVariation'  # (nobs,2) dimension, ioda-converter selects {1}, so I will do the same
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # initialize empty arrays for each pre-QC variable
    pressure               = np.asarray([])
    windSpeed              = np.asarray([])
    zenithAngle            = np.asarray([])
    qualityIndicator       = np.asarray([])
    expectedError          = np.asarray([])
    coefficientOfVariation = np.asarray([])
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate. This is done on a per-variable basis, because some
    # variables are packed together into multi-dimensional arrays and need to be split apart
    # to be sent to separate obs vectors. If you have a variable you want passed along to outputDict
    # that is one of these special cases, include it as a special case below.
    #
    # these are all handled as appends to an initially empty obs vector, since you could have multiple
    # individual queries point to the same output variable, e.g.: latitudes from multiple BUFR tanks
    # all pulled into a single 'latitude' obs vector.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'pressure':
            pressure = np.append(pressure, x)
            if 'pressure' in list(returnDict.values()):
                outputDict['pressure'] = np.append(outputDict['pressure'], x)
        elif mergedDict[key] == 'windSpeed':
            windSpeed = np.append(windSpeed, x)
            if 'windSpeed' in list(returnDict.values()):
                outputDict['windSpeed'] = np.append(outputDict['windSpeed'], x)
        elif mergedDict[key] == 'zenithAngle':
            zenithAngle = np.append(zenithAngle, x)
            if 'zenithAngle' in list(returnDict.values()):
                outputDict['zenithAngle'] = np.append(outputDict['zenithAngle'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAPS')  # GNAPS value determines which column contains which confidence value
            qi_idx = -9
            ee_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 5.:
                    qi_idx = j
                if np.mean(g[:,j]) == 7.:
                    ee_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
            if ee_idx >= 0:
                expectedError = np.append(expectedError, x[:,ee_idx].squeeze())
                #if 'expectedError' in list(returnDict.values()):
                #    outputDict['expectedError'] = np.append(outputDict['expectedError'], x[:,ee_idx].squeeze())
            else:
                print('EE index not found!')
                #if 'expectedError' in list(returnDict.values()):
                #    outputDict['expectedError'] = np.append(outputDict['expectedError'], -9. * x[:,0].squeeze())  # dummy-values
        elif mergedDict[key] == 'coefficientOfVariation':
            coefficientOfVariation = np.append(coefficientOfVariation, x.squeeze())
            if 'coefficientOfVariation' in list(returnDict.values()):
                outputDict['coefficientOfVariation'] = np.append(outputDict['coefficientOfVariation'], x.squeeze())
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here if they aren't already a pre-QC variable in queryDict
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # perform pre-QC checks
    idxPass, idxFail = pre_qc(pre=pressure,
                              spd=windSpeed,
                              zen=zenithAngle,
                              qin=qualityIndicator,
                              cov=coefficientOfVariation,
                              exp=expectedError)
    print(np.size(idxFail), np.size(idxPass))
    # create a preQC variable with 1==pass, -1==fail
    preQC = -1 * np.ones((np.size(idxPass) + np.size(idxFail),), dtype='int')
    preQC[idxPass] = 1
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign all values to 240
    obType = 240 * np.ones(np.shape(preQC), dtype='int')
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005044: draws NC005044 observations (JMA IR/VIS/WV AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005044(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # define internal functions
    #
    # pre_qc: perform pre-QC checks on input data, return indices of pass/fail obs
    #
    # INPUTS:
    #    zen: zenith, angle float(nobs,), deg
    #    qin: quality indicator w/o forecast, int(nobs,), 0-100 index
    #    wcm: wind computation method, int(nobs,), categorical
    #
    # OUTPUTS:
    #    idxPass: indices of observations passing all checks
    #    idxFail: indices of observations failing at least one check
    #
    # DEPENDENCIES:
    #    numpy
    def pre_qc(zen, qin, wcm):
        import numpy as np
        # generate vector of all indices and copy to idxPass
        idxAll = np.arange(np.size(zen))
        idxPass = np.copy(idxAll)
        # zenith angle check
        angMax = 68.
        checkPass = np.where(zen <= angMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail zenith angle check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # quality indicator check
        qiMin = 85
        qiMax = 100
        checkPass = np.where((qin >= qiMin) & (qin <= qiMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail quality indicator check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # wind computation method check
        wcmExcludeList = [5]
        checkPass = np.where(np.isin(wcm, wcmExcludeList)==False)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail wind computation method check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # define idxFail as all indices not in idxPass
        idxFail = np.setdiff1d(idxAll, idxPass)
        print('{:d} OBSERVATIONS FAIL ALL QC, {:d} PASS'.format(np.size(idxFail), np.size(idxPass)))
        # return
        return idxPass, idxFail
    
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    queryDict = {
                 'NC005044/SAZA'        : 'zenithAngle',            # (nobs,) dimension
                 'NC005044/SWCM'        : 'windComputationMethod',  # (nobs,) dimension, as (int) type
                 'NC005044/QCPRMS[1]/PCCF' : 'qualityIndicator'     # (nobs,3) dimension
                                                                    # there are multiple copies of QCPRMS/PCCF but all should be identical
                                                                    # qifn is stored where QCPRMS/GNAP == 102
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # initialize empty arrays for each pre-QC variable
    zenithAngle            = np.asarray([])
    qualityIndicator       = np.asarray([])
    windComputationMethod  = np.asarray([])
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate. This is done on a per-variable basis, because some
    # variables are packed together into multi-dimensional arrays and need to be split apart
    # to be sent to separate obs vectors. If you have a variable you want passed along to outputDict
    # that is one of these special cases, include it as a special case below.
    #
    # these are all handled as appends to an initially empty obs vector, since you could have multiple
    # individual queries point to the same output variable, e.g.: latitudes from multiple BUFR tanks
    # all pulled into a single 'latitude' obs vector.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'zenithAngle':
            zenithAngle = np.append(zenithAngle, x)
            if 'zenithAngle' in list(returnDict.values()):
                outputDict['zenithAngle'] = np.append(outputDict['zenithAngle'], x)
        elif mergedDict[key] == 'qualityIndicator':
            # initialize output array as missing-values
            z = 1.0E+10 * np.ones((np.shape(x)[0],))
            # perform separate BUFR query for generatingApplication data
            querySubDict = {'NC005044/QCPRMS[1]/GNAP' : 'generatingApplication'}
            resultSubSet = bufr_query(bufrFileName, querySubDict)
            y = resultSubSet.get('generatingApplication')
            # find index where y[:,i] == 102 to extract qualityIndicator from x
            for i in range(3):
                if np.unique(y[:,i].squeeze()) == 102:
                    print('found qualityIndicator for i={:d}'.format(i))
                    z[:] = x[:,i].squeeze()
            # append z to qualityIndicator
            qualityIndicator = np.append(qualityIndicator, z)
            #if 'qualityIndicator' in list(returnDict.values()):
            #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], z)
        elif mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here if they aren't already a pre-QC variable in queryDict
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # perform pre-QC checks
    idxPass, idxFail = pre_qc(zen=zenithAngle,
                              qin=qualityIndicator,
                              wcm=windComputationMethod)
    print(np.size(idxFail), np.size(idxPass))
    # create a preQC variable with 1==pass, -1==fail
    preQC = -1 * np.ones((np.size(idxPass) + np.size(idxFail),), dtype='int')
    preQC[idxPass] = 1
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 252  # IR
    obType[np.where(windComputationMethod == 2)] = 242  # VIS
    obType[np.where(windComputationMethod == 3)] = 250  # WVCT
    obType[np.where(windComputationMethod >= 4)] = 250  # WVDL
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005045: draws NC005045 observations (JMA IR/VIS/WV AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005045(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # define internal functions
    #
    # pre_qc: perform pre-QC checks on input data, return indices of pass/fail obs
    #
    # INPUTS:
    #    zen: zenith, angle float(nobs,), deg
    #    qin: quality indicator w/o forecast, int(nobs,), 0-100 index
    #    wcm: wind computation method, int(nobs,), categorical
    #
    # OUTPUTS:
    #    idxPass: indices of observations passing all checks
    #    idxFail: indices of observations failing at least one check
    #
    # DEPENDENCIES:
    #    numpy
    def pre_qc(zen, qin, wcm):
        import numpy as np
        # generate vector of all indices and copy to idxPass
        idxAll = np.arange(np.size(zen))
        idxPass = np.copy(idxAll)
        # zenith angle check
        angMax = 68.
        checkPass = np.where(zen <= angMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail zenith angle check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # quality indicator check
        qiMin = 85
        qiMax = 100
        checkPass = np.where((qin >= qiMin) & (qin <= qiMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail quality indicator check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # wind computation method check
        wcmExcludeList = [5]
        checkPass = np.where(np.isin(wcm, wcmExcludeList)==False)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail wind computation method check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # define idxFail as all indices not in idxPass
        idxFail = np.setdiff1d(idxAll, idxPass)
        print('{:d} OBSERVATIONS FAIL ALL QC, {:d} PASS'.format(np.size(idxFail), np.size(idxPass)))
        # return
        return idxPass, idxFail
    
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    queryDict = {
                 'NC005045/SAZA'        : 'zenithAngle',            # (nobs,) dimension
                 'NC005045/SWCM'        : 'windComputationMethod',  # (nobs,) dimension, as (int) type
                 'NC005045/QCPRMS[1]/PCCF' : 'qualityIndicator'     # (nobs,3) dimension
                                                                    # there are multiple copies of QCPRMS/PCCF but all should be identical
                                                                    # qifn is stored where QCPRMS/GNAP == 102
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # initialize empty arrays for each pre-QC variable
    zenithAngle            = np.asarray([])
    qualityIndicator       = np.asarray([])
    windComputationMethod  = np.asarray([])
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate. This is done on a per-variable basis, because some
    # variables are packed together into multi-dimensional arrays and need to be split apart
    # to be sent to separate obs vectors. If you have a variable you want passed along to outputDict
    # that is one of these special cases, include it as a special case below.
    #
    # these are all handled as appends to an initially empty obs vector, since you could have multiple
    # individual queries point to the same output variable, e.g.: latitudes from multiple BUFR tanks
    # all pulled into a single 'latitude' obs vector.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'zenithAngle':
            zenithAngle = np.append(zenithAngle, x)
            if 'zenithAngle' in list(returnDict.values()):
                outputDict['zenithAngle'] = np.append(outputDict['zenithAngle'], x)
        elif mergedDict[key] == 'qualityIndicator':
            # initialize output array as missing-values
            z = 1.0E+10 * np.ones((np.shape(x)[0],))
            # perform separate BUFR query for generatingApplication data
            querySubDict = {'NC005045/QCPRMS[1]/GNAP' : 'generatingApplication'}
            resultSubSet = bufr_query(bufrFileName, querySubDict)
            y = resultSubSet.get('generatingApplication')
            # find index where y[:,i] == 102 to extract qualityIndicator from x
            for i in range(3):
                if np.unique(y[:,i].squeeze()) == 102:
                    print('found qualityIndicator for i={:d}'.format(i))
                    z[:] = x[:,i].squeeze()
            # append z to qualityIndicator
            qualityIndicator = np.append(qualityIndicator, z)
            #if 'qualityIndicator' in list(returnDict.values()):
            #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], z)
        elif mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here if they aren't already a pre-QC variable in queryDict
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # perform pre-QC checks
    idxPass, idxFail = pre_qc(zen=zenithAngle,
                              qin=qualityIndicator,
                              wcm=windComputationMethod)
    print(np.size(idxFail), np.size(idxPass))
    # create a preQC variable with 1==pass, -1==fail
    preQC = -1 * np.ones((np.size(idxPass) + np.size(idxFail),), dtype='int')
    preQC[idxPass] = 1
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 252  # IR
    obType[np.where(windComputationMethod == 2)] = 242  # VIS
    obType[np.where(windComputationMethod == 3)] = 250  # WVCT
    obType[np.where(windComputationMethod >= 4)] = 250  # WVDL
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005046: draws NC005046 observations (JMA IR/VIS/WV AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005046(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # define internal functions
    #
    # pre_qc: perform pre-QC checks on input data, return indices of pass/fail obs
    #
    # INPUTS:
    #    zen: zenith, angle float(nobs,), deg
    #    qin: quality indicator w/o forecast, int(nobs,), 0-100 index
    #    wcm: wind computation method, int(nobs,), categorical
    #
    # OUTPUTS:
    #    idxPass: indices of observations passing all checks
    #    idxFail: indices of observations failing at least one check
    #
    # DEPENDENCIES:
    #    numpy
    def pre_qc(zen, qin, wcm):
        import numpy as np
        # generate vector of all indices and copy to idxPass
        idxAll = np.arange(np.size(zen))
        idxPass = np.copy(idxAll)
        # zenith angle check
        angMax = 68.
        checkPass = np.where(zen <= angMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail zenith angle check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # quality indicator check
        qiMin = 85
        qiMax = 100
        checkPass = np.where((qin >= qiMin) & (qin <= qiMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail quality indicator check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # wind computation method check
        wcmExcludeList = [5]
        checkPass = np.where(np.isin(wcm, wcmExcludeList)==False)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail wind computation method check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # define idxFail as all indices not in idxPass
        idxFail = np.setdiff1d(idxAll, idxPass)
        print('{:d} OBSERVATIONS FAIL ALL QC, {:d} PASS'.format(np.size(idxFail), np.size(idxPass)))
        # return
        return idxPass, idxFail
    
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    queryDict = {
                 'NC005046/SAZA'        : 'zenithAngle',            # (nobs,) dimension
                 'NC005046/SWCM'        : 'windComputationMethod',  # (nobs,) dimension, as (int) type
                 'NC005046/QCPRMS[1]/PCCF' : 'qualityIndicator'     # (nobs,3) dimension
                                                                    # there are multiple copies of QCPRMS/PCCF but all should be identical
                                                                    # qifn is stored where QCPRMS/GNAP == 102
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # initialize empty arrays for each pre-QC variable
    zenithAngle            = np.asarray([])
    qualityIndicator       = np.asarray([])
    windComputationMethod  = np.asarray([])
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate. This is done on a per-variable basis, because some
    # variables are packed together into multi-dimensional arrays and need to be split apart
    # to be sent to separate obs vectors. If you have a variable you want passed along to outputDict
    # that is one of these special cases, include it as a special case below.
    #
    # these are all handled as appends to an initially empty obs vector, since you could have multiple
    # individual queries point to the same output variable, e.g.: latitudes from multiple BUFR tanks
    # all pulled into a single 'latitude' obs vector.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'zenithAngle':
            zenithAngle = np.append(zenithAngle, x)
            if 'zenithAngle' in list(returnDict.values()):
                outputDict['zenithAngle'] = np.append(outputDict['zenithAngle'], x)
        elif mergedDict[key] == 'qualityIndicator':
            # initialize output array as missing-values
            z = 1.0E+10 * np.ones((np.shape(x)[0],))
            # perform separate BUFR query for generatingApplication data
            querySubDict = {'NC005046/QCPRMS[1]/GNAP' : 'generatingApplication'}
            resultSubSet = bufr_query(bufrFileName, querySubDict)
            y = resultSubSet.get('generatingApplication')
            # find index where y[:,i] == 102 to extract qualityIndicator from x
            for i in range(3):
                if np.unique(y[:,i].squeeze()) == 102:
                    print('found qualityIndicator for i={:d}'.format(i))
                    z[:] = x[:,i].squeeze()
            # append z to qualityIndicator
            qualityIndicator = np.append(qualityIndicator, z)
            #if 'qualityIndicator' in list(returnDict.values()):
            #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], z)
        elif mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here if they aren't already a pre-QC variable in queryDict
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # perform pre-QC checks
    idxPass, idxFail = pre_qc(zen=zenithAngle,
                              qin=qualityIndicator,
                              wcm=windComputationMethod)
    print(np.size(idxFail), np.size(idxPass))
    # create a preQC variable with 1==pass, -1==fail
    preQC = -1 * np.ones((np.size(idxPass) + np.size(idxFail),), dtype='int')
    preQC[idxPass] = 1
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 252  # IR
    obType[np.where(windComputationMethod == 2)] = 242  # VIS
    obType[np.where(windComputationMethod == 3)] = 250  # WVCT
    obType[np.where(windComputationMethod >= 4)] = 250  # WVDL
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005067: draws NC005067 observations (EUMETSAT IR/VIS/WV AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005067(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # define internal functions
    #
    # pre_qc: perform pre-QC checks on input data, return indices of pass/fail obs
    #
    # INPUTS:
    #    zen: zenith, angle float(nobs,), deg
    #    qin: quality indicator w/o forecast, int(nobs,), 0-100 index
    #    wcm: wind computation method, int(nobs,), categorical
    #
    # OUTPUTS:
    #    idxPass: indices of observations passing all checks
    #    idxFail: indices of observations failing at least one check
    #
    # DEPENDENCIES:
    #    numpy
    def pre_qc(zen, qin, wcm):
        import numpy as np
        # generate vector of all indices and copy to idxPass
        idxAll = np.arange(np.size(zen))
        idxPass = np.copy(idxAll)
        # zenith angle check
        angMax = 68.
        checkPass = np.where(zen <= angMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail zenith angle check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # quality indicator check
        qiMin = 85
        qiMax = 100
        checkPass = np.where((qin >= qiMin) & (qin <= qiMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail quality indicator check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # wind computation method check
        wcmExcludeList = [5]
        checkPass = np.where(np.isin(wcm, wcmExcludeList)==False)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail wind computation method check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # define idxFail as all indices not in idxPass
        idxFail = np.setdiff1d(idxAll, idxPass)
        print('{:d} OBSERVATIONS FAIL ALL QC, {:d} PASS'.format(np.size(idxFail), np.size(idxPass)))
        # return
        return idxPass, idxFail
    
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    queryDict = {
                 'NC005067/SAZA'        : 'zenithAngle',            # (nobs,) dimension
                 'NC005067/AMVQIC/GNAPS': 'GNAPS',                  # (nobs,4) dimension, determines data stored in PCCF
                 'NC005067/AMVQIC/PCCF' : 'PCCF',                   # (nobs,4) dimension, QI w/o forecast at GNAPS==5, EE at GNAPS==7
                 'NC005067/SWCM' : 'windComputationMethod'          # (nobs,) dimension
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # initialize empty arrays for each pre-QC variable
    zenithAngle            = np.asarray([])
    qualityIndicator       = np.asarray([])
    windComputationMethod  = np.asarray([])
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate. This is done on a per-variable basis, because some
    # variables are packed together into multi-dimensional arrays and need to be split apart
    # to be sent to separate obs vectors. If you have a variable you want passed along to outputDict
    # that is one of these special cases, include it as a special case below.
    #
    # these are all handled as appends to an initially empty obs vector, since you could have multiple
    # individual queries point to the same output variable, e.g.: latitudes from multiple BUFR tanks
    # all pulled into a single 'latitude' obs vector.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'zenithAngle':
            zenithAngle = np.append(zenithAngle, x)
            if 'zenithAngle' in list(returnDict.values()):
                outputDict['zenithAngle'] = np.append(outputDict['zenithAngle'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAPS')  # GNAPS value determines which column contains which confidence value
            qi_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 5.:
                    qi_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
        elif mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here if they aren't already a pre-QC variable in queryDict
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # perform pre-QC checks
    idxPass, idxFail = pre_qc(zen=zenithAngle,
                              qin=qualityIndicator,
                              wcm=windComputationMethod)
    print(np.size(idxFail), np.size(idxPass))
    # create a preQC variable with 1==pass, -1==fail
    preQC = -1 * np.ones((np.size(idxPass) + np.size(idxFail),), dtype='int')
    preQC[idxPass] = 1
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 253  # IR
    obType[np.where(windComputationMethod == 2)] = 243  # VIS
    obType[np.where(windComputationMethod == 3)] = 254  # WVCT
    obType[np.where(windComputationMethod >= 4)] = 254  # WVDL
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #

# process_NC005068: draws NC005068 observations (EUMETSAT IR/VIS/WV AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005068(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # define internal functions
    #
    # pre_qc: perform pre-QC checks on input data, return indices of pass/fail obs
    #
    # INPUTS:
    #    zen: zenith, angle float(nobs,), deg
    #    qin: quality indicator w/o forecast, int(nobs,), 0-100 index
    #    wcm: wind computation method, int(nobs,), categorical
    #
    # OUTPUTS:
    #    idxPass: indices of observations passing all checks
    #    idxFail: indices of observations failing at least one check
    #
    # DEPENDENCIES:
    #    numpy
    def pre_qc(zen, qin, wcm):
        import numpy as np
        # generate vector of all indices and copy to idxPass
        idxAll = np.arange(np.size(zen))
        idxPass = np.copy(idxAll)
        # zenith angle check
        angMax = 68.
        checkPass = np.where(zen <= angMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail zenith angle check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # quality indicator check
        qiMin = 85
        qiMax = 100
        checkPass = np.where((qin >= qiMin) & (qin <= qiMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail quality indicator check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # wind computation method check
        wcmExcludeList = [5]
        checkPass = np.where(np.isin(wcm, wcmExcludeList)==False)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail wind computation method check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # define idxFail as all indices not in idxPass
        idxFail = np.setdiff1d(idxAll, idxPass)
        print('{:d} OBSERVATIONS FAIL ALL QC, {:d} PASS'.format(np.size(idxFail), np.size(idxPass)))
        # return
        return idxPass, idxFail
    
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    queryDict = {
                 'NC005068/SAZA'        : 'zenithAngle',            # (nobs,) dimension
                 'NC005068/AMVQIC/GNAPS': 'GNAPS',                  # (nobs,4) dimension, determines data stored in PCCF
                 'NC005068/AMVQIC/PCCF' : 'PCCF',                   # (nobs,4) dimension, QI w/o forecast at GNAPS==5, EE at GNAPS==7
                 'NC005068/SWCM' : 'windComputationMethod'          # (nobs,) dimension
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # initialize empty arrays for each pre-QC variable
    zenithAngle            = np.asarray([])
    qualityIndicator       = np.asarray([])
    windComputationMethod  = np.asarray([])
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate. This is done on a per-variable basis, because some
    # variables are packed together into multi-dimensional arrays and need to be split apart
    # to be sent to separate obs vectors. If you have a variable you want passed along to outputDict
    # that is one of these special cases, include it as a special case below.
    #
    # these are all handled as appends to an initially empty obs vector, since you could have multiple
    # individual queries point to the same output variable, e.g.: latitudes from multiple BUFR tanks
    # all pulled into a single 'latitude' obs vector.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'zenithAngle':
            zenithAngle = np.append(zenithAngle, x)
            if 'zenithAngle' in list(returnDict.values()):
                outputDict['zenithAngle'] = np.append(outputDict['zenithAngle'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAPS')  # GNAPS value determines which column contains which confidence value
            qi_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 5.:
                    qi_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
        elif mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here if they aren't already a pre-QC variable in queryDict
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # perform pre-QC checks
    idxPass, idxFail = pre_qc(zen=zenithAngle,
                              qin=qualityIndicator,
                              wcm=windComputationMethod)
    print(np.size(idxFail), np.size(idxPass))
    # create a preQC variable with 1==pass, -1==fail
    preQC = -1 * np.ones((np.size(idxPass) + np.size(idxFail),), dtype='int')
    preQC[idxPass] = 1
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 253  # IR
    obType[np.where(windComputationMethod == 2)] = 243  # VIS
    obType[np.where(windComputationMethod == 3)] = 254  # WVCT
    obType[np.where(windComputationMethod >= 4)] = 254  # WVDL
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #

# process_NC005069: draws NC005069 observations (EUMETSAT IR/VIS/WV AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005069(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # define internal functions
    #
    # pre_qc: perform pre-QC checks on input data, return indices of pass/fail obs
    #
    # INPUTS:
    #    zen: zenith, angle float(nobs,), deg
    #    qin: quality indicator w/o forecast, int(nobs,), 0-100 index
    #    wcm: wind computation method, int(nobs,), categorical
    #
    # OUTPUTS:
    #    idxPass: indices of observations passing all checks
    #    idxFail: indices of observations failing at least one check
    #
    # DEPENDENCIES:
    #    numpy
    def pre_qc(zen, qin, wcm):
        import numpy as np
        # generate vector of all indices and copy to idxPass
        idxAll = np.arange(np.size(zen))
        idxPass = np.copy(idxAll)
        # zenith angle check
        angMax = 68.
        checkPass = np.where(zen <= angMax)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail zenith angle check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # quality indicator check
        qiMin = 85
        qiMax = 100
        checkPass = np.where((qin >= qiMin) & (qin <= qiMax))
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail quality indicator check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # wind computation method check
        wcmExcludeList = [5]
        checkPass = np.where(np.isin(wcm, wcmExcludeList)==False)
        checkFail = np.setdiff1d(idxAll, checkPass)
        idxPass = np.setdiff1d(idxPass, checkFail)
        print('{:d} observations fail wind computation method check, {:d} pass'.format(np.size(checkFail), np.size(checkPass)))
        # define idxFail as all indices not in idxPass
        idxFail = np.setdiff1d(idxAll, idxPass)
        print('{:d} OBSERVATIONS FAIL ALL QC, {:d} PASS'.format(np.size(idxFail), np.size(idxPass)))
        # return
        return idxPass, idxFail
    
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    queryDict = {
                 'NC005069/SAZA'        : 'zenithAngle',            # (nobs,) dimension
                 'NC005069/AMVQIC/GNAPS': 'GNAPS',                  # (nobs,4) dimension, determines data stored in PCCF
                 'NC005069/AMVQIC/PCCF' : 'PCCF',                   # (nobs,4) dimension, QI w/o forecast at GNAPS==5, EE at GNAPS==7
                 'NC005069/SWCM' : 'windComputationMethod'          # (nobs,) dimension
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # initialize empty arrays for each pre-QC variable
    zenithAngle            = np.asarray([])
    qualityIndicator       = np.asarray([])
    windComputationMethod  = np.asarray([])
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate. This is done on a per-variable basis, because some
    # variables are packed together into multi-dimensional arrays and need to be split apart
    # to be sent to separate obs vectors. If you have a variable you want passed along to outputDict
    # that is one of these special cases, include it as a special case below.
    #
    # these are all handled as appends to an initially empty obs vector, since you could have multiple
    # individual queries point to the same output variable, e.g.: latitudes from multiple BUFR tanks
    # all pulled into a single 'latitude' obs vector.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'zenithAngle':
            zenithAngle = np.append(zenithAngle, x)
            if 'zenithAngle' in list(returnDict.values()):
                outputDict['zenithAngle'] = np.append(outputDict['zenithAngle'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAPS')  # GNAPS value determines which column contains which confidence value
            qi_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 5.:
                    qi_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
        elif mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here if they aren't already a pre-QC variable in queryDict
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # perform pre-QC checks
    idxPass, idxFail = pre_qc(zen=zenithAngle,
                              qin=qualityIndicator,
                              wcm=windComputationMethod)
    print(np.size(idxFail), np.size(idxPass))
    # create a preQC variable with 1==pass, -1==fail
    preQC = -1 * np.ones((np.size(idxPass) + np.size(idxFail),), dtype='int')
    preQC[idxPass] = 1
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 253  # IR
    obType[np.where(windComputationMethod == 2)] = 243  # VIS
    obType[np.where(windComputationMethod == 3)] = 254  # WVCT
    obType[np.where(windComputationMethod >= 4)] = 254  # WVDL
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005070: draws NC005070 observations (TERRA/AQUA IR/WV AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005070(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # No pre-QC checks on NC005070, return preQC as effectively all passed (=1) values
    #
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    # (NO pre-QC for this tank, but windComputationMethod required to determine observationType values)
    queryDict = {
                 'NC005070/GQCPRMS[1]/GNAP' : 'GNAP',                # (nobs,4) dimension, determines data stored in PCCF
                 'NC005070/GQCPRMS[1]/PCCF' : 'PCCF',                # (nobs,4) dimension, QI is stored where GNAP==1
                 'NC005070/SWCM'           : 'windComputationMethod' # (nobs,) dimension
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # initialize empty arrays for each pre-QC variable
    qualityIndicator = np.asarray([])
    windComputationMethod  = np.asarray([])
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAP')  # GNAP value determines which column contains which confidence value
            qi_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 1.:
                    qi_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # send "pre-QC" check indices as all-pass (=1)
    preQC = np.ones((np.size(windComputationMethod),), dtype='int')
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 257  # IR
    obType[np.where(windComputationMethod == 3)] = 258  # WVCT
    obType[np.where(windComputationMethod >= 4)] = 259  # WVDL
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005071: draws NC005071 observations (TERRA/AQUA IR/WV AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005071(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # No pre-QC checks on NC005071, return preQC as effectively all passed (=1) values
    #
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    # (NO pre-QC for this tank, but windComputationMethod required to determine observationType values)
    queryDict = {
                 'NC005071/GQCPRMS[1]/GNAP' : 'GNAP',               # (nobs,4) dimension, determines data stored in PCCF
                 'NC005071/GQCPRMS[1]/PCCF' : 'PCCF',               # (nobs,4) dimension, QI is stored where GNAP==1
                 'NC005071/SWCM' : 'windComputationMethod'          # (nobs,) dimension
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # initialize empty arrays for each pre-QC variable
    qualityIndicator = np.asarray([])
    windComputationMethod  = np.asarray([])
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAP')  # GNAP value determines which column contains which confidence value
            qi_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 1.:
                    qi_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # send "pre-QC" check indices as all-pass (=1)
    preQC = np.ones((np.size(windComputationMethod),), dtype='int')
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 257  # IR
    obType[np.where(windComputationMethod == 3)] = 258  # WVCT
    obType[np.where(windComputationMethod >= 4)] = 259  # WVDL
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #

# process_NC005072: draws NC005072 observations (LEO/GEO IR AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005072(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # No pre-QC checks on NC005072, return preQC as effectively all passed (=1) values
    #
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    # (NO pre-QC for this tank, but windComputationMethod required to determine observationType values)
    queryDict = {
                 'NC005072/GNAP[1]': 'GNAP',                        # (nobs,) dimension, there are 3 replications, GNAP[1]=1, GNAP[2]=2, GNAP[3]=3, ioda-converters picks GNAP[1]
                 'NC005072/LGRSQ4[1]/PCCF': 'PCCF',                 # (nobs,) dimension, corresponds to GNAP[1]=1, which is QI w/o forecast
                 'NC005072/SWCM' : 'windComputationMethod'          # (nobs,) dimension
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # initialize empty arrays for each pre-QC variable
    windComputationMethod  = np.asarray([])
    qualityIndicator = np.asarray([])
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        elif mergedDict[key] == 'PCCF':
            qualityIndicator = np.append(qualityIndicator, x)
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x)
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # send "pre-QC" check indices as all-pass (=1)
    preQC = np.ones((np.size(windComputationMethod),), dtype='int')
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 255  # IR
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005080: draws NC005080 observations (NOAA LEO IR AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005080(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # No pre-QC checks on NC005080, return preQC as effectively all passed (=1) values
    #
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    # (NO pre-QC for this tank, but windComputationMethod required to determine observationType values)
    queryDict = {
                 'NC005080/GQCPRMS[1]/GNAP' : 'GNAP',               # (nobs,4) dimension, determines data stored in PCCF
                 'NC005080/GQCPRMS[1]/PCCF' : 'PCCF',               # (nobs,4) dimension, QI is stored where GNAP==1
                 'NC005080/SWCM' : 'windComputationMethod'          # (nobs,) dimension
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # initialize empty arrays for each pre-QC variable
    qualityIndicator = np.asarray([])
    windComputationMethod  = np.asarray([])
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAP')  # GNAP value determines which column contains which confidence value
            qi_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 1.:
                    qi_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # send "pre-QC" check indices as all-pass (=1)
    preQC = np.ones((np.size(windComputationMethod),), dtype='int')
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 244  # IR
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005081: draws NC005081 observations (METOP LEO IR AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005081(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # No pre-QC checks on NC005081, return preQC as effectively all passed (=1) values
    #
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    # (NO pre-QC for this tank, but windComputationMethod required to determine observationType values)
    queryDict = {
                 'NC005081/AMVQIC/GNAPS' : 'GNAPS',             # (nobs,4) dimension, determines data stored in PCCF
                 'NC005081/AMVQIC/PCCF' : 'PCCF',               # (nobs,4) dimension, QI is stored where GNAPS==5
                 'NC005081/SWCM' : 'windComputationMethod'          # (nobs,) dimension
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # initialize empty arrays for each pre-QC variable
    qualityIndicator = np.asarray([])
    windComputationMethod  = np.asarray([])
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAPS')  # GNAPS value determines which column contains which confidence value
            qi_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 5.:
                    qi_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # send "pre-QC" check indices as all-pass (=1)
    preQC = np.ones((np.size(windComputationMethod),), dtype='int')
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 244  # IR
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005090: draws NC005090 observations (NOAA VIIRS IR AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005090(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # No pre-QC checks on NC005090, return preQC as effectively all passed (=1) values
    #
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    # (NO pre-QC for this tank, but windComputationMethod required to determine observationType values)
    queryDict = {
                 'NC005090/AMVQIC/GNAPS' : 'GNAPS',             # (nobs,4) dimension, determines data stored in PCCF
                 'NC005090/AMVQIC/PCCF' : 'PCCF',               # (nobs,4) dimension, QI is stored where GNAPS==5
                 'NC005090/SWCM' : 'windComputationMethod'          # (nobs,) dimension
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # initialize empty arrays for each pre-QC variable
    qualityIndicator = np.asarray([])
    windComputationMethod  = np.asarray([])
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAPS')  # GNAPS value determines which column contains which confidence value
            qi_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 5.:
                    qi_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # send "pre-QC" check indices as all-pass (=1)
    preQC = np.ones((np.size(windComputationMethod),), dtype='int')
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 260  # IR
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_NC005091: draws NC005091 observations (NOAA VIIRS IR AMVs) from BUFR file, and returns
#                   variables based on entries in returnDict.
#
# INPUTS:
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#    outputDict: dictionary with key/value pairs representing
#                    keys: variable name (string)
#                    values: vector of values (numpy vector)
#
# DEPENDENCIES:
#    numpy
#    bufr
#    bufr_query (above)
def process_NC005091(bufrFileName, returnDict):
    import numpy as np
    import bufr
    #
    # No pre-QC checks on NC005091, return preQC as effectively all passed (=1) values
    #
    #
    # begin
    #
    # define dictionary of query/variable key/value pairs needed for pre_qc()
    # (NO pre-QC for this tank, but windComputationMethod required to determine observationType values)
    queryDict = {
                 'NC005091/AMVQIC/GNAPS' : 'GNAPS',             # (nobs,4) dimension, determines data stored in PCCF
                 'NC005091/AMVQIC/PCCF' : 'PCCF',               # (nobs,4) dimension, QI is stored where GNAPS==5
                 'NC005091/SWCM' : 'windComputationMethod'          # (nobs,) dimension
                }
    # merge this dictionary with returnDict, defaulting to these values where appropriate
    mergedDict = returnDict.copy()
    mergedDict.update(queryDict)
    # obtain resultSet from bufr_query()
    resultSet = bufr_query(bufrFileName, mergedDict)
    # initialize empty arrays for each pre-QC variable
    qualityIndicator = np.asarray([])
    windComputationMethod  = np.asarray([])
    # loop through keys, extract array from resultSet and append to appropriate variable array
    # and/or outputDict as appropriate.
    outputDict = {}
    for varName in list(returnDict.values()):
        outputDict[varName] = np.asarray([])
    for key in list(mergedDict.keys()):
        print('processing '+ key + '...')
        x = resultSet.get(mergedDict[key])
        if mergedDict[key] == 'windComputationMethod':
            windComputationMethod = np.append(windComputationMethod, x)
            if 'windComputationMethod' in list(returnDict.values()):
                outputDict['windComputationMethod'] = np.append(outputDict['windComputationMethod'], x)
        elif mergedDict[key] == 'PCCF':
            g = resultSet.get('GNAPS')  # GNAPS value determines which column contains which confidence value
            qi_idx = -9
            for j in range(np.shape(g)[1]):
                if np.mean(g[:,j]) == 5.:
                    qi_idx = j
            if qi_idx >= 0:
                qualityIndicator = np.append(qualityIndicator, x[:,qi_idx].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], x[:,qi_idx].squeeze())
            else:
                print('QI index not found!')
                qualityIndicator = np.append(qualityIndicator, -9. * x[:,0].squeeze())
                #if 'qualityIndicator' in list(returnDict.values()):
                #    outputDict['qualityIndicator'] = np.append(outputDict['qualityIndicator'], -9. * x[:,0].squeeze())  # dummy-values
        else:
            # all variables in mergedDict not in queryDict, assumed to be simple variables with no
            # unpacking of multi-dimensional arrays necessary, but if any special cases exist feel free
            # to add them here
            print('key: ' + key + ' is NOT a pre-QC key')
            if mergedDict[key] in list(returnDict.values()):
                outputDict[mergedDict[key]] = np.append(outputDict[mergedDict[key]], x)
    # send "pre-QC" check indices as all-pass (=1)
    preQC = np.ones((np.size(windComputationMethod),), dtype='int')
    # append preQC to outputDict
    outputDict['preQC'] = preQC
    # create a obType variable and assign values based on windComputationMethod
    obType = -1 * np.ones(np.shape(preQC), dtype='int')
    obType[np.where(windComputationMethod == 1)] = 260  # IR
    # append obType to outputDict
    outputDict['observationType'] = obType
    # append qualityIndicator to outputDict
    outputDict['qualityIndicator'] = qualityIndicator
    # return outputDict
    return outputDict
    #
    # end
    #


# process_satwnd_tank: rout satwnd processing to proper process_<tank>() routine based on tank name, fix some common inputs that
#                      vary between tanks (e.g. specifying 'CLAT' vs 'CLATH' for some tanks when returning latitudes).
#
# INPUTS:
#    tankName: name of tank (string, NC005xxx)
#    bufrFileName: full-path to BUFR file (string)
#    returnDict: dictionary with key/value pairs representing
#                    keys: BUFR query (string)
#                    values: variable name (string)
#
# OUTPUTS:
#   outputDict: output dictionary from satwnd processing
#
# DEPENDENCIES:
#
# all process_<tank> funtions (above)
#
def process_satwnd_tank(tankName, bufrFileName, returnDict):
    # specify processing basedon tankName
    if tankName == 'NC005030':
        # processing GOES LWIR AMVs
        # some quirks:
        # 1) specify pressure key as PRLC[1], there are multiple repititions of PRLC but they should all be identical
        # 2) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005030/PRLC' in key:
                if key[-4:] == 'PRLC':
                    newKey = key + '[1]'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005030/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005030/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005030(bufrFileName, returnDict)
        return outputDict
    # specify processing based on tankName
    if tankName == 'NC005031':
        # processing GOES WVDL AMVs
        # some quirks:
        # 1) specify pressure key as PRLC[1], there are multiple repititions of PRLC but they should all be identical
        # 2) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005031/PRLC' in key:
                if key[-4:] == 'PRLC':
                    newKey = key + '[1]'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005031/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005031/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005031(bufrFileName, returnDict) 
        return outputDict
    if tankName == 'NC005032':
        # processing GOES VIS AMVs
        # some quirks:
        # 1) specify pressure key as PRLC[1], there are multiple repititions of PRLC but they should all be identical
        # 2) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005032/PRLC' in key:
                if key[-4:] == 'PRLC':
                    newKey = key + '[1]'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005032/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005032/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005032(bufrFileName, returnDict) 
        return outputDict
    if tankName == 'NC005034':
        # processing GOES WVCT AMVs
        # some quirks:
        # 1) specify pressure key as PRLC[1], there are multiple repititions of PRLC but they should all be identical
        # 2) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005034/PRLC' in key:
                if key[-4:] == 'PRLC':
                    newKey = key + '[1]'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005034/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005034/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005034(bufrFileName, returnDict) 
        return outputDict
    if tankName == 'NC005039':
        # processing GOES SWIR AMVs
        # some quirks:
        # 1) specify pressure key as PRLC[1], there are multiple repititions of PRLC but they should all be identical
        # 2) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005039/PRLC' in key:
                if key[-4:] == 'PRLC':
                    newKey = key + '[1]'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005039/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005039/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005039(bufrFileName, returnDict) 
        return outputDict
    if tankName == 'NC005044':
        # processing JMA IR/VIS/WV AMVs
        # no quirks reported
        # process and return outputDict
        outputDict = process_NC005044(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005045':
        # processing JMA IR/VIS/WV AMVs
        # no quirks reported
        # process and return outputDict
        outputDict = process_NC005045(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005046':
        # processing JMA IR/VIS/WV AMVs
        # no quirks reported
        # process and return outputDict
        outputDict = process_NC005046(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005067':
        # processing EUMETSAT IR/VIS/WV AMVs
        # some quirks:
        # 1) specify pressure key as PRLC[1], there are multiple repititions of PRLC but they should all be identical
        # 2) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005067/PRLC' in key:
                if key[-4:] == 'PRLC':
                    newKey = key + '[1]'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005067/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005067/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005067(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005068':
        # processing EUMETSAT IR/VIS/WV AMVs
        # some quirks:
        # 1) specify pressure key as PRLC[1], there are multiple repititions of PRLC but they should all be identical
        # 2) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005068/PRLC' in key:
                if key[-4:] == 'PRLC':
                    newKey = key + '[1]'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005068/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005068/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005068(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005069':
        # processing EUMETSAT IR/VIS/WV AMVs
        # some quirks:
        # 1) specify pressure key as PRLC[1], there are multiple repititions of PRLC but they should all be identical
        # 2) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005069/PRLC' in key:
                if key[-4:] == 'PRLC':
                    newKey = key + '[1]'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005069/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005069/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005069(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005070':
        # processing TERRA/AQUA IR/WV AMVs
        # no quirks reported
        # process and return outputDict
        outputDict = process_NC005070(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005071':
        # processing TERRA/AQUA IR/WV AMVs
        # no quirks reported
        # process and return outputDict
        outputDict = process_NC005071(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005072':
        # processing LEO/GEO IR AMVs
        # some quirks:
        # 1) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005072/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005072/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005072(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005080':
        # processing NOAA IR LEO AMVs (AVHRR)
        # no quirks reported
        # process and return outputDict
        outputDict = process_NC005080(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005081':
        # processing METOP IR LEO AMVs (AVHRR)
        # some quirks:
        # 1) specify pressure key as PRLC[1], there are multiple repititions of PRLC but they should all be identical
        # 2) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005081/PRLC' in key:
                if key[-4:] == 'PRLC':
                    newKey = key + '[1]'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005081/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005081/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005081(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005090':
        # processing NOAA VIIRS AMVs
        # some quirks:
        # 1) specify pressure key as PRLC[1], there are multiple repititions of PRLC but they should all be identical
        # 2) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005090/PRLC' in key:
                if key[-4:] == 'PRLC':
                    newKey = key + '[1]'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005090/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005090/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005090(bufrFileName, returnDict)
        return outputDict
    if tankName == 'NC005091':
        # processing NOAA VIIRS AMVs (new BUFR sequence)
        # some quirks:
        # 1) specify pressure key as PRLC[1], there are multiple repititions of PRLC but they should all be identical
        # 2) use 'CLATH' and 'CLONH' for lat/lon values (high-precision)
        for key in list(returnDict.keys()):
            if 'NC005091/PRLC' in key:
                if key[-4:] == 'PRLC':
                    newKey = key + '[1]'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005091/CLAT' in key:
                if key[-4:] == 'CLAT':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
            if 'NC005091/CLON' in key:
                if key[-4:] == 'CLON':
                    newKey = key + 'H'
                    returnDict[newKey] = returnDict.pop(key)
        # process and return outputDict
        outputDict = process_NC005091(bufrFileName, returnDict)
        return outputDict
