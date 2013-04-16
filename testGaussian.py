# Run a test to evaluate parameters of 2d gaussian through FFT/PSD/ACF. 
# This is useful, because a Gaussian should be analytically predictable through each of these transformations. 
# Summary: 
# The FFT of a gaussian is a Gaussian, with sigma_fft (in frequency space) = 1/(2*pi*sigma_x). 
#  Note that a gaussian also translates to a real-only FFT (imaginary part = 0), if using analytic, perfect transform. 
# Then the 2d PSD is created by squaring the (real+imaginary) FFT - and a gaussian squared is a gaussian with sigma 
#   divided by sqrt 2. So the sigma_psd2d = sigma_fft / sqrt(2) (note this is still in frequency). 
# The 2d ACF is calculated by taking the FFT of the PSD, thus translating back to spatial coordinates .. so 
#  sigma_acf = 1/(2*pi*sigma_psd2d) = 1/(2*pi*sigma_FFT)*sqrt(2) = sigma_x*sqrt(2)
# Looking at the 1d version of the ACF, we should still find a gaussian with same sigma, and same for 1d PSD in frequency.
#  But, if we want to translate the PSD into 'spatial' scales related to pixels, we can look at the width of the gaussian
#  and try to translate that to pixel scales. The result is that sigma_psd(pix) = sqrt(nx*ny)/(sigma_psd(freq)*2*pi)

import numpy
import pylab
import testImage as ti
from scipy.optimize import curve_fit
from scipy import fftpack

# Gaussian function (for fitting).
def gaussian(x, *p):
    A, mu, sigma = p
    return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))
# Simple function to generate expected and fit data values.
def dofit(x, d, mean, sigma):
    p = (d.max().real, mean, sigma)
    coeff, var_matrix = curve_fit(gaussian, x, d, p0=p)
    fitval = gaussian(x, *coeff)
    expval = gaussian(x, *p)
    print '  Fit:', coeff,  ' Exp:', p
    return fitval, expval
def doplot(x, d, fitval, expval, title):
    pylab.figure()
    pylab.plot(x, d, 'b.', label='data')
    pylab.plot(x, fitval, 'r-', label='fit')
    pylab.plot(x, expval, 'k:', label='expected')
    pylab.legend(numpoints=1, fancybox=True, fontsize='small')
    pylab.title(title)
    return
    
# Step 1 - generate 2d gaussian image. 
# Generate 2d gaussian image of size 'size' and with sigma 'sigma_x'. 
size = 750
sigma_x = 5.0
gauss = ti.TestImage(shift=True, nx = size, ny=size)
gauss.addGaussian(xwidth=sigma_x, ywidth=sigma_x, value=1.0) 
gauss.zeroPad()
#gauss.hanningFilter()
# and calculate FFT/PSD2d/PSD1d/ACF2d/ACF1d.
gauss.calcAll(min_npix=2, min_dr=1.0)

print 'Using image size of %.0f (but with zeroPadding to get to %.0f) and sigma of %.2f' %(size, gauss.nx, sigma_x)

# Step 2 - analyze the width of the gaussians for each of the original image/FFT/PSD/ACF. 

# Look at original image, and slice through center. 
print 'Plotting Image'
gauss.showImage()
# Get x data for 1-d gaussian fit to data from the center of the image
x = numpy.arange(0, gauss.nx, 1.0)
# Get 'y' (image) data for 1-d gaussian fit
d = gauss.image[round(gauss.ny/2.0)][:]
# Set expected values for fit ...
mean = gauss.nx/2.0
sigma = sigma_x
fitval, expval = dofit(x, d, mean, sigma)
doplot(x, d, fitval, expval, 'Image slice')

# Move on to FFT
print 'Plotting FFT'
gauss.showFft(real=True, imag=True)
# Slice through FFT ... note that we're now dealing with FREQUENCY space
#   w (freq) = 1/(2*PI*x)
x = gauss.xfreq
d = fftpack.ifftshift(gauss.fimage)[0][:].real
d = numpy.abs(d)
idx = numpy.argsort(x)
d = d[idx]
x = x[idx]
# So expect gaussian to FT to another gaussian, but now with mu=0 and sigma=1/(2*PI*sigma_x)
mean = 0
sigma_fft = 1/(2.0*numpy.pi*sigma_x)
fitval, expval = dofit(x, d, mean, sigma_fft)
doplot(x, d, fitval, expval, 'FFT slice')
pylab.xlim(-.2, .2)

# Now look at PSD - in frequency space first.
print 'Plotting PSD frequency scale'
gauss.showPsd2d()
d = fftpack.ifftshift(gauss.psd2d)[0][:].real
d = d[idx]
# because PSD is square of FFT (which was a gaussian), expect mu = 0 but
#  sigma = sigma_fft / sqrt(2) == (1/sigma_x*2*PI)*1/sqrt(2)
mean = 0
sigma_psd_freq = sigma_fft/numpy.sqrt(2)
fitval, expval = dofit(x, d, mean, sigma_psd_freq)
doplot(x, d, fitval, expval, 'PSD2d slice')
pylab.xlim(-.2, .2)

# And now look at 2d psd in 'pixel' scales
print 'Plotting PSD2d spatial scale'
x = numpy.arange(-gauss.xcen, gauss.nx-gauss.xcen, 1.0)
d = gauss.psd2d[round(gauss.ny/2.0)][:].real
# Because of pixel 'scaling' (and is tied to total size of image) expect sigma = sqrt(nx*ny)/(sigma_x*sqrt(2)*2*PI)
mean = 0
sigma_psd_pix = 1/(sigma_x*numpy.sqrt(2))*numpy.sqrt(gauss.nx*gauss.ny)/(2.0*numpy.pi)
fitval, expval = dofit(x, d, mean, sigma_psd_pix)
doplot(x, d, fitval, expval, 'PSD2d slice, spatial scale')
pylab.xlim(-200, 200)

# Plot 1d PSD (checked this with frequency space, and it matches. However, in 'pixel space', it just doesn't really.
# This may be just a problem with my interpretation, but it does seem like there is something more fundamental here.
# For example, the distribution in pixel space is not really quite a gaussian (even fitting 1/x) - althouhg it is close.
"""
print 'plotting 1d PSD'
gauss.showPsd1d(linear=True)
x = gauss.psdx
d = gauss.psd1d.real
r = 1/x
sigma = 1/(sigma_psd_pix)
#r =x
#sigma = sigma_psd_freq
fitval, expval = dofit(r, d, 0., sigma)
doplot(x, d, fitval, expval, 'PSD1d')
"""

# Now on to the ACF
print 'Plotting ACF'
gauss.showAcf2d()
x = numpy.arange(-gauss.xcen, gauss.nx-gauss.xcen, 1.)
d = gauss.acf.real[round(gauss.ny/2.0)][:]
# Okay, ACF is actual inverse FFT of PSD, so in pixel scales, expect sigma_acf = 1/(sigma_psd_freq*2*PI)
#   ... or = sigma_x*sqrt(2) because of squaring the gaussian during the PSD creation. 
#sigma_acf = 1/(sigma_psd_freq*2.0*numpy.pi)
mean = 0
sigma_acf = sigma_x*numpy.sqrt(2)
fitval, expval = dofit(x, d, mean, sigma_acf)
doplot(x, d, fitval, expval, 'ACF2d slice')

# And look at the 1d ACF. 
print 'Plotting acf 1d.'
x = gauss.acfx
d = gauss.acf1d.real
sigma_acf = sigma_x*numpy.sqrt(2)
fitval, expval = dofit(x, d, mean, sigma_acf)
doplot(x, d, fitval, expval, 'ACF1d')

pylab.show()
