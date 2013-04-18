import numpy as np
import testImage as t
from pylab import *


def zpad(image):
    """Stupid old version of numpy doesn't have pad module in it """
    xsize,ysize=image.shape
    padded = np.zeros( (xsize*2,ysize*2))
    padded[xsize/2:xsize*1.5,ysize/2:ysize*1.5] = image
    return padded


def myplot(ack):
    figure()
    subplot(2,2,1)
    imshow(ack.image)
    title('Image')
    subplot(2,2,2)
    imshow(ack.fimage.real, vmin=-1,vmax=1)
    title('real FFT')
    colorbar()
    subplot(2,2,3)
    imshow(log(ack.psd2d), vmin=-10,vmax=10,origin='lower')
    title('log 2D Power Spectrum')
    colorbar()
    subplot(2,2,4)
    good = np.where(ack.psd1d != 0)
    semilogy(ack.rcenters[good],ack.psd1d[good])
    xlabel('Spatial Frequency')
    ylabel('1-D Power Spectrum')
    ylim([ack.psd1d[ack.rcenters.size-20], np.max(ack.psd1d)])


oldCLouds = np.load('oldclouds.npy')
newClouds = np.load('newclouds.npy')

ack = t.TestImage()
cloudim = zpad(oldCLouds.copy())

ack.setImage(cloudim)

ack.calcFft()
ack.calcPsd2d()
ack.calcPsd1d()

figure()
subplot(2,2,1)
imshow(ack.image)
title('Image')
subplot(2,2,2)
imshow(ack.fimage.real, vmin=-1,vmax=1)
title('real FFT')
colorbar()
subplot(2,2,3)
imshow(log(ack.psd2d), vmin=-10,vmax=10,origin='lower')
title('log 2D Power Spectrum')
colorbar()
subplot(2,2,4)
semilogy(ack.psd1d)
xlabel('Spatial Frequency')
ylabel('1-D Power Spectrum')

savefig('oldclouds.png',type='png')
clf()

cloudim = zpad(newClouds.copy())
ack.setImage(cloudim)
ack.calcFft()
ack.calcPsd2d()
ack.calcPsd1d()

figure()
subplot(2,2,1)
imshow(ack.image)
title('Image')
subplot(2,2,2)
imshow(ack.fimage.real, vmin=-1,vmax=1)
title('real FFT')
colorbar()
subplot(2,2,3)
imshow(log(ack.psd2d), vmin=-10,vmax=10,origin='lower')
title('log 2D Power Spectrum')
colorbar()
subplot(2,2,4)
semilogy(ack.psd1d)
xlabel('Spatial Frequency')
ylabel('1-D Power Spectrum')

savefig('newclouds.png',type='png')


xx = ack.xx.copy()
yy = ack.yy.copy()

im = t.TestImage()

scales = np.array([5.,10.,500.])

for i in np.arange(np.size(scales)):

    scale = scales[i]
    imsin = np.sin(2.*np.pi/scale*xx)#*im.xx**2.
    imss = imsin*np.sin(2.*np.pi/scale*yy)
    im.setImage(zpad(imss))
    im.calcFft()
    im.calcPsd2d()
    im.calcPsd1d()
    myplot(im)
    ylim([1e-5,1e7])
    savefig('sin_ps%d.png'%scale, type='png')
    clf()





#im.showFft()
#im.showPsd1d()
