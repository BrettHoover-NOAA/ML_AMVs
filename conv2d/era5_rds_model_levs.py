#! /usr/bin/env python3
#
# python script to download selected files from rda.ucar.edu
# after you save the file, don't forget to make it executable
#   i.e. - "chmod 755 <name_of_script>"
#
import requests
#
files = [
#    "e5.oper.an.ml/202212/e5.oper.an.ml.0_5_0_0_0_t.regn320sc.2022123118_2022123123.nc",
    "e5.oper.an.ml/202212/e5.oper.an.ml.128_134_sp.regn320sc.2022123118_2022123123.nc",
#    "e5.oper.an.ml/202301/e5.oper.an.ml.0_5_0_0_0_t.regn320sc.2023010100_2023010105.nc",
    "e5.oper.an.ml/202301/e5.oper.an.ml.128_134_sp.regn320sc.2023010100_2023010105.nc",
#    "e5.oper.an.ml/202301/e5.oper.an.ml.0_5_0_0_0_t.regn320sc.2023010106_2023010111.nc",
    "e5.oper.an.ml/202301/e5.oper.an.ml.128_134_sp.regn320sc.2023010106_2023010111.nc",
]
#
# download the data file(s)
for file in files:
    idx = file.rfind("/")
    if (idx > 0):
        ofile = file[idx+1:]
    else:
        ofile = file

    response = requests.get("https://data.rda.ucar.edu/ds633.6/" + file)
    with open(ofile, "wb") as f:
        f.write(response.content)
