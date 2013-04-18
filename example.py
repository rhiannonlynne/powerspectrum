# Walk through constructing & inverting the Image/FFT/PSD/ACF & 1d PSD/1d ACF. 
# Comment and uncomment lines to do different things (this is really just a sort of convenient working script). 

import numpy
import pylab

from testImage import TestImage
# from pImage import PImage  # if you don't need to do any plotting
from pImagePlots import PImagePlots  # if you do need to use the plotting functions for PImage

# Use TestImage to set up the image
im = TestImage(shift=True, nx=1000, ny=1000)
#im.addLines(spacing=50, width=10, value=10, angle=0)
#im.addGaussian(xwidth=30, ywidth=30)
#im.addSin(scale=100)
#im.addCircle(radius=20)
im.addEllipseRandom(nEllipse=100, value=5)
im.addNoise(sigma=1.)
im.hanningFilter()
im.zeroPad()

# Calculate FFT/PSD2d/ACF/PSD1d in one go (can do these separately too).
# Use automatic binsize or user-defined binsize.
im.calcAll(min_dr=1.0, min_npix=2)

#im.showImage()
#im.showAcf2d(log=True, imag=False)
#im.showPsd1d()
#im.showAcf1d()
#im.showSf()

# Start at various points in inverting to reconstruct image (comment out stages above where you want to start).
# What I find is that not including the phases means a randomly lumpy reconstructed image - nothing like original
#  image. 
# However, if you start with the 1d ACF (even with no phase info), then you can reconstruct something which then
# will let you recalculate a pretty good copy of the original 1d ACF. 
# Similarly, if you start with the 1D PSD, you can reconstruct a pretty good copy of the 1d PSD.
# BUT, starting with the 1d PSD means that the 1d ACF will not be so good at large scales, and starting with the 
# 1d ACF means that the reconstructed 1d PSD will not be so good at small scales (although these scales seem to be so
# small that it may be meaningless .. but I'm still having some problems interpreting the pixel size in the 1d PSD, so
# it's probably on the order of a few pixel-scales ... the images do look grainier). 

# My conclusion from this is that the 1d ACF (because of how it's built) does better at preserving large scales, 
# while the 1d PSD may do better at preserving (very??) small scales after the reconstruction. 

# Start here to invert from 1d ACF (and use phase info or not)
#im.invertAcf1d()#phasespec=im.phasespec)
# Start here to invert from 2d ACF (useI = False then, and uses phase info - get perfect reconstruction)
im.invertAcf2d()
#im.invertAcf2d(useI=True)
# Start here to invert from 1d PSD (and use phase info or not)
#im.invertPsd1d(phasespec=im.phasespec)
# Start here to invert from 2d PSD (useI = False then, and uses phase info - get perfect reconstruction)
im.invertPsd2d(useI=True)
# Start here to invert from FFT (useI = False, and will get perfect reconstruction). 
im.invertFft(useI=True)

# Use im2 to recalculate 1d PSD/ACF starting from the reconstructed image, without altering the original. 
im2 = PImagePlots()
im2.setImage(im.imageI.real, copy=True)
im2.calcAll(min_dr=1.0, min_npix=2)


# Now start plotting things, in comparison. 
clims = im.showImage()
print clims
im2.showImage(clims=clims)
im2.showImage()
im.showFft(clims=clims)
im2.showFft(clims=clims)
im.showPsd2d()
im2.showPsd2d()
im.showPhases()
im2.showPhases()
im.showAcf2d()
im2.showAcf2d(imag=False)
im.showPsd1d(comparison=im2)
im.showAcf1d(comparison=im2)
im.showSf(comparison=im2)

pylab.show()
exit()

