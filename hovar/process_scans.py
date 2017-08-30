#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import numpy as np
import utils as hu
from multiprocessing import Pool


# -- utilities
DATADIR = os.path.join("..","data")
VEGDIR  = os.path.join("..","output","veg_specs")
SKYDIR  = os.path.join("..","output","sky_specs")
SATDIR  = os.path.join("..","output","saturated")
BLDDIR  = os.path.join("..","output","bld_specs")
ALTDIR  = os.path.join("..","output","alt_specs")
NEWDIR  = os.path.join("..","output","new_specs")
satval  = 2**12 - 1 # the saturation value
nsrow   = 160 # the number of sky rows [10% of image]
dlabs   = np.load("../output/km_00055_ncl10_see314_labs_full.npy") # KM labels
trind   = (dlabs==2)|(dlabs==5) # vegetation spectra
sh      = [1600,1601] # shape of scan 00055 used to label


# -- define processing pipeline
def process_scan(fname):
    """
    Process a VNIR scan for vegetation.
    """

    # -- get the scan number
    snum = fname.split("_")[1].replace(".raw","")

    # -- set the output files
    vegfile = os.path.join(VEGDIR,"veg_specs_{0}.npy".format(snum))
    skyfile = os.path.join(SKYDIR,"sky_spec_{0}.npy".format(snum))
    satfile = os.path.join(SATDIR,"nsat_{0}.npy".format(snum))
    bldfile = os.path.join(BLDDIR,"bld_specs_{0}.npy".format(snum))
    altfile = os.path.join(ALTDIR,"alt_specs_{0}.npy".format(snum))
    newfile = os.path.join(NEWDIR,"new_specs_{0}.npy".format(snum))

    # -- check if they exist
    vegdone = os.path.isfile(vegfile)
    skydone = os.path.isfile(skyfile)
    satdone = os.path.isfile(satfile)
    blddone = os.path.isfile(bldfile)
    altdone = os.path.isfile(altfile)
    newdone = os.path.isfile(newfile)
    if vegdone and skydone and satdone and blddone and altdone and newdone:
        return

    # -- read data file and initialize time
    print("working on scan {0}...".format(snum))
    t0   = time.time()
    cube = hu.read_hyper(fname)

    # -- pull off vegetation pixels (remove last column if necessary)
    if not vegdone:
        print("pulling off vegetation pixels...")

        # -- write to file
        np.save(vegfile,cube.data[:,trind.reshape(sh)[:,:cube.data.shape[2]]])

    # -- calculate sky
    if not skydone:
        print("calculating sky...")

        # -- write to file
        np.save(skyfile,cube.data[:,:nsrow].mean(-1).mean(-1))

    # -- calculate number of saturated pixels
    if not satdone:
        print("calculating saturated pixels...")

        # -- write to file
        np.save(satfile,(cube.data==satval).sum(0))

    # -- get the building spectrum
    if not blddone:
        print("calculating building spectrum...")

        # -- write to file
        np.save(bldfile,cube.data[:,990:1034,799:956])
        np.save(bldfile.replace("specs_","specs_avg_"),
                cube.data[:,990:1034,799:956].mean(-1).mean(-1))

    # -- get the alternate building spectrum
    if not altdone:
        print("calculating alternate building spectrum...")

        # -- write to file
        # region 1
        # 933 344
        # 933 364
        # 970 344
        # 970 364
        # region 2
        # 970 352
        # 970 364
        # 1000 352
        # 1000 364
        # region 3
        # 931 455
        # 931 477
        # 962 455
        # 962 477
        r1r    = [933,970]
        r1c    = [344,364]
        r2r    = [970,1000]
        r2c    = [352,364]
        r3r    = [931,962]
        r3c    = [455,477]
        npixr1 = (r1r[1]-r1r[0])*(r1c[1]-r1c[0])
        npixr2 = (r2r[1]-r2r[0])*(r2c[1]-r2c[0])
        npixr3 = (r3r[1]-r3r[0])*(r3c[1]-r3c[0])
        aspecs = np.hstack([cube.data[:,r1r[0]:r1r[1],r1c[0]:r1c[1]] \
                                .reshape(cube.data.shape[0],npixr1),
                            cube.data[:,r2r[0]:r2r[1],r2c[0]:r2c[1]] \
                                .reshape(cube.data.shape[0],npixr2),
                            cube.data[:,r3r[0]:r3r[1],r3c[0]:r3c[1]] \
                                .reshape(cube.data.shape[0],npixr3)])
        np.save(altfile,aspecs.T)
        np.save(altfile.replace("specs_","specs_avg_"),aspecs.mean(-1))

    # -- get the new building spectrum
    if not newdone:
        print("calculating new building spectrum...")

        # -- write to file
        r1r    = [1125,1137]
        r1c    = [790,829]
        npixr1 = (r1r[1]-r1r[0])*(r1c[1]-r1c[0])
        nspecs = cube.data[:,r1r[0]:r1r[1],r1c[0]:r1c[1]] \
            .reshape(cube.data.shape[0],npixr1)
        np.save(newfile.replace("specs_","specs_avg_"),nspecs.mean(-1))

    # -- alert user
    dt = time.time() - t0
    print("processed cube in {0}m:{1:02}s".format(int(dt//60),int(dt%60)))

    return


if __name__=="__main__":

    # -- set number of processors
    nproc = 8

    # -- get the file list
    flist = sorted(glob.glob(os.path.join(DATADIR,"veg_*.raw")))

    # -- define processing runs for parallelization
    st = int(sys.argv[1])
    en = int(sys.argv[2])

    for fname in flist[st:en+1]:
        process_scan(fname)
