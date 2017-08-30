#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np
import utils as hu
from datetime import datetime
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from plotting import set_defaults

# -- plotting defaults
set_defaults()

# # -- plot in dark color scheme if available
# try:
#     dark_plot()
# except:
#     pass

# -- set the types
kind = "veg"
comp = "new"
runpca = False
runfan = False
runica = False
getndvi = True

#def analyze_specs(kind="veg",comp="bld"):

# -- get wavelengths
waves = hu.read_header("../data/veg_00000.hdr")["waves"]

# -- define good scans
good = np.array([int(i) for i in np.load("../output/good_scans.npy")]) 

# -- load the comparison set
print("getting {0} spectra...".format(comp))
blds = np.array([np.load(i) for i in
                 sorted(glob.glob("../output/{0}_specs/{0}_specs_avg*.npy"\
                                      .format(comp)))])
blds = blds[good]

# -- normalize spectra
ms, bs = [], []
for ii in range(blds.shape[0]):
    # m, b = np.polyfit(blds[ii,:100],blds[0,:100],1)
    # m, b = np.polyfit(blds[ii],blds[0],1)
    m = blds[0].mean()/blds[ii].mean()
    b = 0.0
    ms.append(m)
    bs.append(b)

ms   = np.array(ms)
bs   = np.array(bs)
norm = blds*ms[:,np.newaxis] + bs[:,np.newaxis]
rat  = norm/norm[0]

# -- get vegetation spectra
print("getting {0} spectra...".format(kind))
if kind=="veg":
    vegs = np.load("../output/veg_patch_specs.npy")
else:
    vegs = np.array([np.load(i) for i in
                     sorted(glob.glob("../output/" + 
                                      "{0}_specs/{0}_specs_avg*.npy" \
                                      .format(kind)))])
    vegs = vegs[good]

# -- normalize spectra
ss, os = [], []
for ii in range(vegs.shape[0]):
    # s, o = np.polyfit(vegs[ii,:100],vegs[0,:100],1)
    # s, o = np.polyfit(vegs[ii],vegs[0],1)
    s = vegs[0].mean()/vegs[ii].mean()
    o = 0.0
    ss.append(s)
    os.append(o)

ss    = np.array(ss)
os    = np.array(os)
vnorm = vegs*ss[:,np.newaxis] + os[:,np.newaxis]
vrat  = vnorm/vnorm[0]

# -- take the ratio of ratios
brat = vrat/rat

# -- get some ancillary data
sc     = pd.read_csv("../output/scan_conditions.csv")
sc_sub = sc[sc.filename.isin(["veg_{0:05}.raw".format(i) for i in good])]

temps = sc_sub.temperature.values
humid = sc_sub.humidity.values
pm25  = sc_sub.pm25.values
o3    = sc_sub.o3.values
secs  = []
for stime in sc_sub.time.values:
    yr, mo, dy, tm = stime.split()
    stime_fmt      = "{0} {1} {2:02} {3}:00".format(yr,mo,int(dy),tm)
    obs_dt         = datetime.strptime(stime_fmt,"%Y %b %d %H:%M:%S")
    secs.append(float(obs_dt.strftime("%s")))
secs = np.array(secs)

# -- PCA
if runpca:
    print("running PCA...")
    pca = PCA(n_components=6)
    pca.fit(vrat/rat)
    pamps = pca.transform(vrat/rat)

# -- Factor Analysis
if runfan:
    print("running Factor Analysis...")
    fan = FactorAnalysis(n_components=6)
    fan.fit(vrat/rat)
    famps = fan.transform(vrat/rat)

# -- ICA
if runica:
    print("running ICA...")
    ica = FastICA(n_components=6)
    ica.fit(vrat/rat)
    iamps = ica.transform(vrat/rat)

# -- get NDVI
if getndvi:
    print("calculating NDVI...")
    print("  getting sky spectra...")
    flist   = sorted(glob.glob("../output/sky_specs/*.npy"))
    skys    = np.array([np.load(i) for i in flist])[good]
    print("  getting reflectance...")
    ref     = (vegs - vegs.min(1,keepdims=True))/ \
        (skys-skys.min(1,keepdims=True))
    ind_ir  = np.argmin(np.abs(waves-860.))
    ind_vis = np.argmin(np.abs(waves-670.))
    print("  generating NDVI...")
    ndvi    = (ref[:,ind_ir]-ref[:,ind_vis]) / \
        (ref[:,ind_ir]+ref[:,ind_vis])


# -- multi-variate correlation
brightness = (brat).mean(1)
templates  = np.vstack([o3,pm25,temps,humid,np.ones_like(o3)]).T
#ind = brightness<2.0
ind  = np.arange(len(brightness))
sol  = np.linalg.lstsq(templates[ind],brightness[ind])
pred = np.dot(templates[ind],sol[0])
rsq  = 1.0-((brightness-pred)**2).sum() / \
    ((brightness-brightness.mean())**2).sum()

ind  = humid<99.
sol  = np.linalg.lstsq(templates[ind],brightness[ind])
pred2 = np.dot(templates[ind],sol[0])



# -- plots
plt.close("all")

# # -- example differential reflectance
# fig, ax = plt.subplots(figsize=[6.5,3])
# fig.subplots_adjust(0.125,0.18,0.95,0.9)
# ax.plot(waves*1e-3,brat[30],color="darkred")
# ax.set_xlabel("wavelength [micron]")
# ax.set_ylabel("$D(\lambda,t)$")
# xr, yr = ax.get_xlim(), ax.get_ylim()
# ax.text(xr[1],yr[1]+0.025*(yr[1]-yr[0]),r"$\Delta t = 10$ hrs",ha="right",
#         fontsize=14)
# fig.canvas.draw()
# fig.savefig("../output/diffref_ex.png", clobber=True)
# fig.savefig("../output/diffref_ex.eps", clobber=True)

# # -- all differential reflectance
# plt.close("all")
# fig, ax = plt.subplots(figsize=[6.5,5])
# fig.subplots_adjust(0.1,0.1,0.9,0.95)
# ax.xaxis.grid(0)
# ax.yaxis.grid(linestyle="-",color="darkorange",lw=0.5)
# ax.imshow(brat.T, clim=(0.7,1.5))
# ax.set_yticks([np.argmin(np.abs(waves-i)) for i in range(400,1100,100)])
# ax.set_yticklabels([str(i) for i in np.arange(0.4,1.1,0.1)])
# ax.set_xlabel("scan number")
# ax.set_ylabel("wavelength [micron]")
# xr, yr = ax.get_xlim(), ax.get_ylim()
# ax.text(xr[1],yr[1]+0.025*(yr[1]-yr[0]),r'$D_{\lambda,t}$',ha="right",
#         fontsize=14)
# cb = fig.add_axes((0.92,0.05,0.05,0.9))
# cb.grid("off")
# [cb.spines[i].set_linewidth(1) for i in cb.spines]
# cb.set_xticklabels("")
# cb.set_yticklabels("")
# cb.set_xticks([])
# cb.set_yticks([])
# cbvals = np.arange(1000).reshape(100,10)[::-1] // 10
# cb.text(5,-2,"0.7",ha="center",va="bottom")
# cb.text(5,102,"1.5",ha="center",va="top")
# cb.imshow(cbvals)
# fig.canvas.draw()
# fig.savefig("../output/diffref_scans.png", clobber=True)
# fig.savefig("../output/diffref_scans.eps", clobber=True)


# # -- example differential reflectance
# plt.close("all")
# fig, ax = plt.subplots(figsize=[6.5,3])
# fig.subplots_adjust(0.125,0.18,0.95,0.9)
# ax.plot(waves*1e-3,brat[::10].T,color="darkred",lw=0.1)
# ax.set_xlabel("wavelength [micron]")
# ax.set_ylabel("$D(\lambda,t)$")
# ax.set_ylim(0.6,1.8)
# xr, yr = ax.get_xlim(), ax.get_ylim()
# ax.text(xr[1],yr[1]+0.025*(yr[1]-yr[0]),"10% of scans",ha="right",
#         fontsize=14)
# fig.canvas.draw()
# fig.savefig("../output/diffref_10pct.png", clobber=True)
# fig.savefig("../output/diffref_10pct.eps", clobber=True)

# # -- pca
# plt.close("all")
# fig = plt.figure(figsize=[6.5,9])
# fig.subplots_adjust(0.15,0.1,0.95,0.95,0.15)
# ax = fig.add_subplot(3,1,1)
# ax.plot(waves*1e-3,pca.mean_,color=plt.rcParams["axes.color_cycle"][0])
# xr = ax.get_xlim()
# yr = ax.get_ylim()
# ax.text(xr[0],yr[1]+0.025*(yr[1]-yr[0]),"Mean spectrum",ha="left")
# ax.set_xlabel("wavelength [micron]")
# ax.set_ylabel('$D(\lambda, t)$')
# for ii in range(6):
#     ax = fig.add_subplot(5,2,4+ii+1)
#     ax.plot(waves*1e-3,pca.components_[ii],
#             color=plt.rcParams["axes.color_cycle"][ii+1])
#     ax.set_ylim(-0.1,0.1)
#     ax.set_xlim(0.4,1.0)
#     if (ii%2)!=0:
#         ax.set_yticklabels("")
#     if ii<4:
#         ax.set_xticklabels("")
#     else:
#         ax.set_xlabel("wavelength [micron]")
#     if ii==2:
#         ax.set_ylabel("PC amplitude [arb units]")
#     xr = ax.get_xlim()
#     yr = ax.get_ylim()
#     ax.text(xr[0],yr[1]+0.025*(yr[1]-yr[0]),"Component {0}".format(ii+1),
#             ha="left")
#     ax.text(xr[1],yr[1]+0.025*(yr[1]-yr[0]),"EV = {0:4.1f}%"\ 
#             .format(pca.explained_variance_ratio_[ii]*100),ha="right")
# fig.canvas.draw()
# fig.savefig("../output/pca_components.eps", clobber=True)
# fig.savefig("../output/pca_components.pdf", clobber=True)
# fig.savefig("../output/pca_components.png", clobber=True)

# # -- PCA vs environment
# plt.close("all")
# labs = ["NDVI [0.5, 1.0]","Humidity [20, 95] %","PM2.5 [0, 12] $\mu$g/m$^3$",
#         "O3 [0.015, 0.055] ppm"]
# fig, ax = plt.subplots(2,2,figsize=[6.5,6.5],sharex=True,sharey=True)
# fig.subplots_adjust(0.1,0.1,0.95,0.95,0.1,0.1)
# for ii,dat in enumerate([ndvi.clip(0.5,1.0),
#                          humid.values.clip(20,95),
#                          pm25.values.clip(0,12),
#                          o3.values.clip(0.015,0.055)]):
#     tax  = ax[ii//2,ii%2]
#     clrs = plt.cm.jet((dat-dat.min())/(dat-dat.min()).max())
#     tax.axes.set_axis_bgcolor("#AAAAAA")
#     tax.scatter(pamps[:,0],pamps[:,1],lw=0,c=clrs,s=10)
#     tax.set_xlim(-10,5)
#     tax.set_ylim(-5,10)
#     if ii%2==0:
#         tax.set_ylabel("Component 2 amplitude")
#     if ii//2==1:
#         tax.set_xlabel("Component 1 amplitude")
#     xr = tax.get_xlim()
#     yr = tax.get_ylim()
#     tax.text(xr[0],yr[1],labs[ii],va="top") \ 
#     .set_backgroundcolor("w")
# fig.canvas.draw()
# fig.savefig("../output/four_panel.eps", clobber=True)
# fig.savefig("../output/four_panel.png", clobber=True)
# fig.savefig("../output/four_panel.pdf", clobber=True)


# # -- daily fluctuation (get seconds, subtract first,  mod by day, etc)
# plt.close("all")
# secs0 = secs - secs[0]
# fold  = secs0 % (24.*3600.)

# fig, ax = plt.subplots(figsize=[6.5,4])
# fig.subplots_adjust(0.125,0.175,0.95,0.9)
# ax.plot(fold,ndvi,'.',color="darkblue")
# ax.set_xticks([i*3600 for i in range(-1,12)])
# ax.set_xticklabels(["{0:02}:00".format(i+8) for i in range(-1,12)],rotation=90)
# ax.set_ylabel("NDVI")
# xr = ax.set_xlim()
# yr = ax.set_ylim()
# ax.text(xr[1],yr[1]+0.025*(yr[1]-yr[0]),"Diurnal folding",fontsize=14,
#         ha="right")
# fig.canvas.draw()
# fig.savefig("../output/diurnal_folding.eps", clobber=True)
# fig.savefig("../output/diurnal_folding.pdf", clobber=True)
# fig.savefig("../output/diurnal_folding.png", clobber=True)


# # -- relationship to NDVI
# plt.close("all")
# fig, ax = plt.subplots(figsize=[6.5,4])
# ax.scatter(ndvi,brat[:,500]/brat[:,-1],lw=0,c='maroon',s=10)
# ax.plot(ndvi[ndvi<0],brat[ndvi<0,500]/brat[ndvi<0,-1],'o',color="none",ms=10,
#         mec="dodgerblue")
# ax.set_xlabel("NDVI")
# ax.set_ylabel(r'$D(\lambda=0.75\mu$m$)/D(\lambda=1.0\mu$m$)$')
# fig.canvas.draw()
# fig.savefig("../output/diffref_ndvi.eps", clobber=True)
# fig.savefig("../output/diffref_ndvi.pdf", clobber=True)
# fig.savefig("../output/diffref_ndvi.png", clobber=True)


# # -- all differential reflectance with O3 overlay
# plt.close("all")
# fig, ax = plt.subplots(figsize=[6.5,5])
# fig.subplots_adjust(0.1,0.1,0.9,0.95)
# ax.grid(0)
# ax.plot(o3*10000,color="darkorange",lw=1)
# ax.imshow(brat.T/brat[:,-100:].mean(1), clim=(0.7,1.5))
# lind = [np.argmin(np.abs(waves-i)) for i in range(400,1100,100)]
# ax.set_yticks(lind)
# ax.set_yticklabels([str(i) for i in np.arange(0.4,1.1,0.1)])
# ax.set_xlabel("scan number")
# ax.set_ylabel("wavelength [micron]")
# xr, yr = ax.get_xlim(), ax.get_ylim()
# ax.text(xr[1],yr[1]+0.025*(yr[1]-yr[0]),
#         r'$D_{\lambda,t}/\langle D_{\lambda_{0.9}^{1.0},t}\rangle$',ha="right",
#         fontsize=14)
# ax.text(xr[0]+0.025*(xr[1]-xr[0]),yr[0]+0.025*(yr[1]-yr[0]),"O3 [ppm]",
#         color="darkorange")
# fig.canvas.draw()
# fig.savefig("../output/diffref_scans_o3.eps", clobber=True)
# fig.savefig("../output/diffref_scans_o3.pdf", clobber=True)
# fig.savefig("../output/diffref_scans_o3.png", clobber=True)


# # -- plot prediction
# brightness = brat[:,500]/brat[:,-1]
# templates  = np.vstack([o3,pm25,temps,humid,np.ones_like(o3)]).T
# ind  = np.arange(len(brightness))
# sol  = np.linalg.lstsq(templates[ind],brightness[ind])
# pred = np.dot(templates[ind],sol[0])
# rsq  = 1.0-((brightness-pred)**2).sum() / \
#     ((brightness-brightness.mean())**2).sum()

# plt.close("all")
# fig, ax = plt.subplots(figsize=[6.5,3.5])
# fig.subplots_adjust(0.125,0.15,0.95,0.9)
# linb, = ax.plot(brightness,color="darkred",lw=1)
# linp, = ax.plot(pred,color="dodgerblue",lw=2)
# ax.set_ylim(0,2.5)
# ax.set_xlim(0,pred.size)
# ax.set_xlabel("scan number")
# #ax.set_ylabel(r'$\langle D(\lambda,t) \rangle_{\lambda}$')
# ax.set_ylabel(r'$D(\lambda=0.75\mu$m$)/D(\lambda=1.0\mu$m$)$')
# ax.legend([linb,linp],["data","model"],loc="upper left",fontsize=12)
# fig.canvas.draw()
# fig.savefig("../output/regress_vegbld.eps", clobber=True)
# fig.savefig("../output/regress_vegbld.pdf", clobber=True)
# fig.savefig("../output/regress_vegbld.png", clobber=True)


'''
figure()
plot(waves,brat[::10].T,lw=0.3,color="indianred")
xlabel("wavelength")
ylabel("brightness ratio")

figure()
lin0, = plot(brightness[ind],lw=1)
lin1, = plot(pred)
xlabel("scan number")
ylabel("\"brightness\"")
legend([lin0,lin1],["data", "model"],loc="lower left")
title("O3, PM2.5, T, Humid regression")
# savefig("../output/brightness_model_veg.png", facecolor="k", clobber=True)
'''

def plot4(amps, c0=0, c1=1):
    dark_plot()

    figure(figsize=(10,10))

    try:
        fndvi = ndvi.clip(0.4,0.8)
        clrs = plt.cm.jet((fndvi-fndvi.min())/(fndvi-fndvi.min()).max())
        gcf().add_subplot(221); scatter(amps[:,1],amps[:,3],c=clrs,s=10,
                                        linewidths=0)
        subplots_adjust(0.1)
        ylabel("PCA component {0}".format(c1))
        title("NDVI")
    except:
        pass

    clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
    gcf().add_subplot(222); scatter(amps[:,c0],amps[:,c1],c=clrs,s=10,
                                    linewidths=0)
    subplots_adjust(0.1)
    title("Humidity")

    clrs = plt.cm.jet((pm25-pm25.min())/(pm25-pm25.min()).max())
    gcf().add_subplot(223); scatter(amps[:,c0],amps[:,c1],c=clrs,s=10,
                                    linewidths=0)
    subplots_adjust(0.1)
    xlabel("PCA component {0}".format(c0))
    ylabel("PCA component {0}".format(c1))
    title("PM2.5")

    clrs = plt.cm.jet((o3-o3.min())/(o3-o3.min()).max())
    gcf().add_subplot(224); scatter(amps[:,c0],amps[:,c1],c=clrs,s=10,
                                    linewidths=0)
    subplots_adjust(0.1)
    xlabel("PCA component {0}".format(c0))
    title("O3")

# plot4(pamps,0,1)


# -- # -- # -- # -- # -- # -- # -- # -- # -- # -- # -- # -- 





# close("all")
# for ii in range (5):
#     clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
#     figure(figsize=[7,5]); scatter(o3,amps[:,ii],c=clrs,s=10,linewidths=0)
#     subplots_adjust(0.15)
#     xlabel("O3 [ppm]")
#     ylabel("PCA component {0}".format(ii))
#     title("Humidity")




# def ploto3(amps):
#     close("all")
#     figure(figsize=[10,10])
#     for ii in range (6):
#         clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
#         gcf().add_subplot(3,2,ii+1)
#         scatter(o3,amps[:,ii],c=clrs,s=10,linewidths=0)
#         subplots_adjust(0.15)
#         xlabel("O3 [ppm]")
#         ylabel("PCA component {0}".format(ii))
#         title("Humidity")
#     plt.subplots_adjust(0.05,0.05,0.95,0.95)


# def plotpm25(amps):
#     close("all")
#     figure(figsize=[10,10])
#     for ii in range (6):
#         clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
#         gcf().add_subplot(3,2,ii+1)
#         scatter(pm25,amps[:,ii],c=clrs,s=10,linewidths=0)
#         subplots_adjust(0.15)
#         xlabel("PM2.5 [ppm]")
#         ylabel("PCA component {0}".format(ii))
#         title("Humidity")
#     plt.subplots_adjust(0.05,0.05,0.95,0.95)


# def plottemp(amps):
#     close("all")
#     figure(figsize=[10,10])
#     for ii in range (6):
#         clrs = plt.cm.jet((humid-humid.min())/(humid-humid.min()).max())
#         gcf().add_subplot(3,2,ii+1)
#         scatter(temps,amps[:,ii],c=clrs,s=10,linewidths=0)
#         subplots_adjust(0.15)
#         xlabel("T [K]")
#         ylabel("PCA component {0}".format(ii))
#         title("Humidity")
#     plt.subplots_adjust(0.05,0.05,0.95,0.95)




# # -- let's try to find some weights that make a nice correlation
# data = vrat/rat
# data = data[1:]
# o3m  = o3[1:]

# data -= data.mean(0)
# data /= data.std(0)
# o3m  -= o3m.mean()
# o3m  /= o3m.std()

# dTd    = np.dot(data.T,data)
# o3mdT  = np.dot(data.T,o3m)
# dTdinv = np.linalg.pinv(dTd)
# wgto3  = np.dot(o3mdT,dTdinv)


# pm25m  = pm25[1:]

# pm25m  -= pm25m.mean()
# pm25m  /= pm25m.std()

# pm25mdT  = np.dot(data.T,pm25m)
# wgtpm25  = np.dot(pm25mdT,dTdinv)





# # -- more plots:
# dark_plot()
# figure()
# plot(o3*10000)
# imshow((vrat/rat).T,clim=[0.6,2.0]) # should be (0.99,1.01) for bld/bld
# grid(0)

# figure()
# plot(temps*18-800)
# imshow((vrat/rat).T,clim=[0.6,2.0]) # should be (0.99,1.01) for bld/bld

# figure()
# plot(humid*8)
# imshow((vrat/rat).T,clim=[0.6,2.0]) # should be (0.99,1.01) for bld/bld


# rrat = ((vrat/rat).T).clip(0.6,2.0)
# rgb  = rrat - rrat.min()
# rgb /= rgb.max()
# rgb  = np.dstack([255*rgb for i in range(3)])

# o3norm  = humid.values-humid.values.min()
# o3norm /= o3norm.max()
# o3norm *= 255

# o3rgb = np.zeros_like(rgb)

# for ii in range(rgb.shape[1]):
#     for jj in range(3):
#         o3rgb[:,ii,jj] = o3norm[ii]





# close("all")
# dark_plot()
# figure()
# plot(o3[ind],brightness[ind],'.')
# mo, bo = np.polyfit(o3[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mo*o3[ind]+bo))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(o3,mo*o3+bo)
# title(r"$R^{2}=%4.2f$" % R2)
# ylabel("brightness")
# xlabel("O3 [ppm]")
# savefig("../output/brightness_O3_veg.png", facecolor="k", clobber=True)

# figure()
# plot(pm25[ind],brightness[ind],'.')
# mp, bp = np.polyfit(pm25[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mp*pm25[ind]+bp))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(pm25,mp*pm25+bp)
# title(r"$R^{2}=%4.2f$" % R2)
# ylabel("brightness")
# xlabel("PM2.5 [ppm]")
# savefig("../output/brightness_PM25_veg.png", facecolor="k", clobber=True)

# figure()
# plot(temps[ind],brightness[ind],'.')
# mt, bt = np.polyfit(temps[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mt*temps[ind]+bt))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(temps,mt*temps+bt)
# title(r"$R^{2}=%4.2f$" % R2)
# ylabel("brightness")
# xlabel("T [F]")
# savefig("../output/brightness_T_veg.png", facecolor="k", clobber=True)

# figure()
# plot(humid[ind],brightness[ind],'.')
# mh, bh = np.polyfit(humid[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mh*humid[ind]+bh))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(humid,mh*humid+bh)
# title(r"$R^{2}=%4.2f$" % R2)
# ylabel("brightness")
# xlabel("Humidity [%]")
# savefig("../output/brightness_H_veg.png", facecolor="k", clobber=True)

# figure()
# lin0, = plot(brightness[ind],lw=1)
# lin1, = plot(pred)
# xlabel("scan number")
# ylabel("\"brightness\"")
# legend([lin0,lin1],["data", "model"],loc="lower left")
# title("O3, PM2.5, T, Humid regression")
# savefig("../output/brightness_model_veg.png", facecolor="k", clobber=True)

# figure()
# plot(o3*10000)
# imshow((vrat/rat).T,clim=[0.6,2.0])
# gca().set_yticks([np.argmin(np.abs(waves-i)) for i in range(400,1000,100)])
# gca().set_yticklabels([str(i) for i in np.arange(0.4,1.0,0.1)])
# gcf().canvas.draw()
# title("red line scales as O3 ppm")
# xlabel("scan number")
# ylabel("wavelength")
# savefig("../output/scaled_spectra_veg.png", facecolor="k", clobber=True)


# close("all")
# figure()
# plot(o3[ind],brightness[ind],'.')
# mo, bo = np.polyfit(o3[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mo*o3[ind]+bo))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(o3,mo*o3+bo)
# title(r"$R^{2}=%f$" % R2)
# ylabel("brightness")
# xlabel("O3 [ppm]")

# figure()
# plot(pm25[ind],brightness[ind],'.')
# mp, bp = np.polyfit(pm25[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mp*pm25[ind]+bp))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(pm25,mp*pm25+bp)
# title(r"$R^{2}=%f$" % R2)
# ylabel("brightness")
# xlabel("PM2.5 [ppm]")

# figure()
# plot(temps[ind],brightness[ind],'.')
# mt, bt = np.polyfit(temps[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mt*temps[ind]+bt))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(temps,mt*temps+bt)
# title(r"$R^{2}=%f$" % R2)
# ylabel("brightness")
# xlabel("T [F]")

# figure()
# plot(humid[ind],brightness[ind],'.')
# mh, bh = np.polyfit(humid[ind],brightness[ind],1)
# R2 = 1-((brightness[ind]-(mh*humid[ind]+bh))**2).sum() / \
#     ((brightness[ind]-brightness[ind].mean())**2).sum()
# plot(humid,mh*humid+bh)
# title(r"$R^{2}=%f$" % R2)
# ylabel("brightness")
# xlabel("Humidity [%]")

# figure()
# lin0, = plot(brightness[ind],lw=1)
# lin1, = plot(pred)
# xlabel("scan number")
# ylabel("\"brightness\"")
# legend([lin0,lin1],["data", "model"],loc="lower left")
# title("O3, PM2.5, T, Humid regression")

# A = 


# figure()
# plot(o3,(vrat/rat).sum(1))
# clf()
# plot(o3,(vrat/rat).sum(1),'o')
# figure()
# plot(humid,(vrat/rat).sum(1),'o')
# close("all")
