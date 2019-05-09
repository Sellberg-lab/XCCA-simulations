#!/usr/bin/env python2.7
#****************************************************************************************************************
# Add Gaussian and White Noise to each Detector Tile or ASIC
# 2019-05-09 v1 @ Caroline Dahlqvist cldah@kth.se
#			test_gnoise_asics.py
# Run directory must contain the Mask-folder
# Prometheus path: /Users/Lucia/Documents/KTH/Simulations_CsCl/
# lynch path: /Users/lynch/Documents/users/caroline/Simulations_CsCl/
# Davinci: /home/cldah/source/XCCA-simulations/CsCl
#*****************************************************************************************************************
import os
import numpy as np
from  matplotlib import pyplot as plt
from scipy import ndimage

this_dir = os.path.dirname(os.path.realpath(__file__))
mask = np.load("%s/masks/better_mask-assembled.npy" %str(this_dir))

def plot_label_nr(ax,r,c,l):
	for ri,ci,li in zip(r,c,range(1,l+1)):  
		ax.annotate( '# %i'%li, xy=(ci,ri ), xytext=(0,0), textcoords='offset points', ha='center', va='center', fontsize=10 )
	return

##def add_gasuss_noise(img=shots,   n_level=?1):
#
## Select and plot only segments of 1's in mask to find ASICs:
#

## ---- Fill in single pixel holes from 'bad' pixels: ----- ##
asics_test = ndimage.binary_fill_holes(mask).astype(int) ## if not tyecast => floats ##

## ---- For Tile  Selection (185x388, ca 27 pxl gap tile-tile): ---- ##
#asics_test=ndimage.binary_dilation(asics_test, structure=np.ones([12,18], dtype=int), iterations=1, border_value=0, origin=0, brute_force=False)

## ---- For ASIC Selection (185x194 pxl, ca 8pxl asic-asic) : ---- ##		!! OBS problem with sub-plots -> scaling joins segments!!
asics_test=ndimage.binary_dilation(asics_test, structure=np.ones([5,5], dtype=int), iterations=1, border_value=0, origin=0, brute_force=False)
##asics_test=ndimage.binary_dilation(asics_test, structure=np.ones([7,5], dtype=int), iterations=1, border_value=0, origin=0, brute_force=False)

img=asics_test.astype(int) 		## if not typecast => < type 'numpy.bool_'>##
del asics_test

## ---- label connected regions that are nonzero ---- ##
nzr_idx= img > 0.0
## labels !=0 groups, each location has the label number as its pixel value; 'nlabel'= number of labels ##
labels, nlabels = ndimage.label(nzr_idx)
## 'segment' = slices of pixels indicatint location of each labeled object ##
segments = ndimage.find_objects(labels) ## list of tulpes ##
#print "object-type: ", type(segments[0])
## center of mass weighted by pixels (floats) ##
r,c = np.vstack(ndimage.center_of_mass(img, labels, np.arange(nlabels)+1)).T
print "\n Dim of labels: ", labels.shape ## (1738, 1742), same as mask ##
print "\n nlabels: ", nlabels ## = 32 tiles, = 64 asics  ##
#print"\n r: ", r; print type(r[0]) ## = numpy.float ##


Gnoise_cmpl= [] # list obj # np.zeros_like(img)
Wnoise_cmpl= np.zeros_like(img)

#
## Generate Gaussian-shaped-noise for each labeled unit and summ complete image
#
std=0.1 ## std = sqrt(var)= FWHM:  1%. 5%, 10% ##
mean = img.astype(float).mean() ## ?MAX || MEAN (ca.200) of ring at q:(300,800)##
print"\n Mean of img: ", mean ## 0.7445379106063028 ##
for i in range(nlabels):
	j=i+1
	select=np.where( labels == j ) ## the pixels for label i as a (2,X) tuple. 2 = Rows,Columns  ##
	region=segments[i] ## type = tuple  length: 2##
	region_size = labels[region].shape
	#R,C =np.where( labels == i ) ## the pixels to make noise in Rows,Column ##
	print "label = %i : \n  "%i, select[0], "\n  ", select[1]
	print "\nNumber of coordinates: ", img[select].shape 	##  (35502,) ##
	#print "Shape of select = (label=N): ", np.array(select).shape 	## = (2, 35502) ##
	#print "Shape of labeles[ label = N ]: ", labels[ labels==i ].shape 	## = (35502,) ##
	#region = labels[select] # indes_where_l_is_i=labels[ labels==i ]
	print "Shape of selected region in 'labels': ", region_size ## (194, 183)
	mask_seg_g=np.zeros_like(img)
	mask_seg_w=np.zeros_like(img)

	## ---- GAUSSIAN noise =>  mean = mean(image) : ----- ##
	g_noise = np.random.normal(loc=mean, scale=std, size=region_size)
	## ---- WHITE noise =>  mean = 0.0 : ----- ##
	w_noise = np.random.normal(loc=0.0, scale=std, size=region_size)
	#w_noise *= mean ## alt. only fluctuation around 0 and add to image ##
	
	#for idx in range(img.shape[0]):
	#	## rand(n,m) yield nxm of numbers (0.0,1.0); raindint(low,high) ##
	#np.random.rand(cntr_int[1], cntr_int[0]) * ip[idx][:cntr_int[1]-250,:cntr_int[0]-250].max()*nlevel

	#print "Shape of 'mask_seg_G[region]': ", mask_seg_g[region].shape 	## (194, 183) ##
	#print "Shape of 'mask_seg_W[select]': ", mask_seg_w[select].shape 	## (35502,) ##
	mask_seg_g[region]=g_noise
	mask_seg_w[region]=w_noise 

	## mask_seg_g[select]=g_noise
	## ValueError: shape mismatch: value array of shape (194,183) could not be broadcast to indexing result of shape (35502,)##

	Gnoise_cmpl.append(mask_seg_g) 	## list obj append(Y,X) => (N,Y,X); extend(Z,Y,X)=>(Z*N,Y,X)##
	Wnoise_cmpl+= mask_seg_w 		## np.array (Y,X) ##
## Make lists to numpy arrays: ##
Gnoise_cmpl= np.asarray(Gnoise_cmpl) ## np.array (#labels,Y,X) ##
#Wnoise_cmpl= np.asarray(Wnoise_cmpl)
print "\nDim of Gaussian-noise, 1 per label: ", Gnoise_cmpl.shape
## Sum segments together to patch together complete image ##
Gnoise_cmpl= np.sum(Gnoise_cmpl,0)
#Wnoise_cmpl= np.sum(Wnoise_cmpl,0)
print "Dim of Gaussian-noise compilation: ", Gnoise_cmpl.shape

#
## Add Complete Noise-map to the signal (signal + noise): ##
#
img_g_noisy = Gnoise_cmpl
img_w_noisy = Wnoise_cmpl + img
## colorbar may need integer type for binary data (0 to 1) ##
#img_g_noisy, img_w_noisy  = img_g_noisy .astype(int), img_w_noisy.astype(int)

#R,C =np.where( labels >68 ) ## only observe labels larger than 68 ##
#print "R: ", R
#mask_seg[R,C]= labels[R,C]
#mask_seg[select]= labels[select] ## label# values l-64 ASICs alt 1-32 Tiles ##
#mask_seg[select]= img[select] ## binary 0-1 ##
#print "dim of segment: ", mask_seg.shape
#mask_seg = mask_seg.astype(int) ## colorbar need int for binary data ##

#fig = plt.figure(figsize=(13,5)) ## 15,6; better 13,6; works13,5
fig = plt.figure(figsize=(18,6)) ## 15,6; better 13,6; works13,5

fs_t = 14
cmap = plt.cm.get_cmap('rainbow') ## rainbow, coolwarm, bwr, RdYlBu, Wistia ##
ax_before = fig.add_subplot(131)
ax_g_noise = fig.add_subplot(132)
ax_w_noise = fig.add_subplot(133)

print type(img[0,0])	## <type 'numpy.bool_'> ##
## ALL ##
before=ax_before.imshow(np.ma.masked_array(img, ~nzr_idx), cmap=cmap, aspect='equal', interpolation='nearest',
							vmin=0, vmax=1)
plt.colorbar(before, ax=ax_before, fraction=0.046)
ax_before.set_title("Before :\n Detector Sections", fontsize=fs_t)
plot_label_nr(ax_before,r,c,nlabels)


#plot_label_nr(ax,r,c,nlabels)
#axs=[ax_before,ax_noise, ax_after]
#fig.colorbar(before, ax=axs.ravel().tolist(), orientation="horizontal",fraction=0.043)# shrink=0.75)

Gauss=ax_g_noise.imshow(np.ma.masked_array(img_g_noisy, ~nzr_idx), cmap=cmap, aspect='equal')
plt.colorbar(Gauss, ax=ax_g_noise, fraction=0.046)
ax_g_noise.set_title("Gaussian Noise :\n $\sigma =$ %.2f, $\mu =$ %.2f"%(std,mean), fontsize=fs_t)

WGNF=ax_w_noise.imshow(np.ma.masked_array(img_w_noisy, ~nzr_idx), cmap=cmap, aspect='equal')
plt.colorbar(WGNF, ax=ax_w_noise, fraction=0.046)
ax_w_noise.set_title("White Gaussian Noise (WGNF) :\n $\sigma =$ %.2f, $\mu =$ %.2f"%(std,mean),fontsize=fs_t)

#color_map= plt.cm.  image.get_cmap()
cmap.set_bad('black',1.) ## (color='k', alpha=None) Set color to be used for masked values. ##
plt.subplots_adjust(wspace=0.35, left=0.06, right=0.93, top=0.9, bottom=0.15) ## hspace=0.4 ##
#plt.suptitle('$\mu=$%.2f'%mean)
plt.show()