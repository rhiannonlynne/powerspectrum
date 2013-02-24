import numpy as np
import testImage as t
from pylab import *


def zpad(image):
    """Stupid old version of numpy doesn't have pad module in it """
    xsize,ysize=image.shape
    padded = np.zeros( (xsize*2,ysize*2))
    padded[xsize/2:xsize*1.5,ysize/2:ysize*1.5] = image
    return padded


oldCLouds = np.load('oldclouds.npy')
newClouds = np.load('newclouds.npy')

ack = t.TestImage()
cloudim = zpad(oldCLouds.copy())

ack.setImage(cloudim)

ack.makeFft()
ack.makePsd()

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
ack.makeFft()
ack.makePsd()

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
