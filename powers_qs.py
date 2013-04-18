# Code to generate the figures in powers_qs.tex

import numpy
import pylab
from pImagePlots import PImagePlots 
from testImage import TestImage 

from scipy import fftpack

figformat='png'

do_example_lines =  True
do_example_lines =  False
do_gaussian_example = True
do_gaussian_example = False
do_compare1d_psd_acf = True
do_compare1d_psd_acf = False
do_inversion = True
do_inversion = False
do_elliptical = True
do_elliptical = False
do_clouds = True

# opening example
def example_lines():
    """Plot for example_lines, illustrating entire forward process both with and without hanning filter."""
    im = TestImage(shift=True, nx=1000, ny=1000)
    im.addLines(width=10, spacing=75, value=5, angle=45)
    im.zeroPad()
    im.calcAll()
    im.plotMore()
    pylab.savefig('example_lines_a.%s' %(figformat), format='%s' %(figformat))
    pylab.close()
    im = TestImage(shift=True, nx=1000, ny=1000)
    im.addLines(width=10, spacing=75, value=5, angle=45)
    im.hanningFilter()
    im.zeroPad()
    im.calcAll()
    im.plotMore()
    pylab.savefig('example_lines_b.%s' %(figformat), format='%s' %(figformat))
    pylab.close()
    return


# A collection of functions for the gaussian example
def gaussian(x, *p):
    A, mu, sigma = p
    return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))
def dofit(x, d, mean, sigma):
    from scipy.optimize import curve_fit
    p = (d.max().real, mean, sigma)
    coeff, var_matrix = curve_fit(gaussian, x, d, p0=p)
    fitval = gaussian(x, *coeff)
    expval = gaussian(x, *p)
    #print '  Fit:', coeff,  ' Exp:', p
    return fitval, expval
def doplot(x, d, fitval, expval, title, xlabel='Pixels'):
    pylab.figure()
    pylab.plot(x, d, 'b.', label='data')
    pylab.plot(x, fitval, 'r-', label='fit')
    pylab.plot(x, expval, 'k:', label='expected')
    pylab.legend(numpoints=1, fancybox=True, fontsize='small')
    pylab.xlabel(xlabel)
    pylab.title(title)
    return

# gaussian example
def gaussian_example():
    """Generate the plots for the example gaussian ... a more detailed version of this is walked through in the
    testGaussian.py code. """
    gauss = TestImage(shift=True, nx=1000, ny=1000)
    sigma_x = 10.
    gauss.addGaussian(xwidth=sigma_x, ywidth=sigma_x, value=1)
    gauss.zeroPad()
    gauss.calcAll(min_npix=2, min_dr=1)
    gauss.plotMore()
    pylab.savefig('gauss_all.%s' %(figformat), format='%s' %(figformat))
    # pull slice of image
    x = numpy.arange(0, gauss.nx, 1.0)
    d = gauss.image[round(gauss.ny/2.0)][:]
    mean = gauss.xcen
    sigma = sigma_x
    fitval, expval = dofit(x, d, mean, sigma)
    doplot(x, d, fitval, expval, 'Image slice', xlabel='Pixels')
    pylab.savefig('gauss_image.%s' %(figformat), format='%s' %(figformat))
    # pull slice of FFT
    x = gauss.xfreq
    d = fftpack.ifftshift(gauss.fimage)[0][:].real
    d = numpy.abs(d)
    idx = numpy.argsort(x)
    d = d[idx]
    x = x[idx]
    mean = 0
    sigma_fft = 1/(2.0*numpy.pi*sigma_x)
    fitval, expval = dofit(x, d, mean, sigma_fft)
    doplot(x, d, fitval, expval, 'FFT slice', xlabel='Frequency')
    pylab.xlim(-.2, .2)
    pylab.savefig('gauss_fft.%s' %(figformat), format='%s' %(figformat))
    # pull slice from PSD
    d = fftpack.ifftshift(gauss.psd2d)[0][:].real
    d = d[idx]
    mean = 0
    sigma_psd_freq = sigma_fft/numpy.sqrt(2)
    fitval, expval = dofit(x, d, mean, sigma_psd_freq)
    doplot(x, d, fitval, expval, 'PSD 2-d slice', xlabel='Frequency')
    pylab.xlim(-.2, .2)
    pylab.savefig('gauss_psd_freq.%s' %(figformat), format='%s' %(figformat))
    # and look at slice from PSD in spatial scale
    x = numpy.arange(-gauss.xcen, gauss.nx-gauss.xcen, 1.0)
    d = gauss.psd2d[round(gauss.ny/2.0)][:].real
    mean = 0
    sigma_psd_pix = 1/(sigma_x*numpy.sqrt(2))*numpy.sqrt(gauss.nx*gauss.ny)/(2.0*numpy.pi)
    fitval, expval = dofit(x, d, mean, sigma_psd_pix)
    doplot(x, d, fitval, expval, 'PSD 2-d slice, spatial scale', xlabel='"Pixels"')
    pylab.xlim(-200, 200)
    pylab.savefig('gauss_psd_x.%s' %(figformat), format='%s' %(figformat))
    # Show 1d PSD in both frequency and pixel space
    gauss.showPsd1d(linear=True)
    pylab.savefig('gauss_psd1d_all.%s' %(figformat), format='%s' %(figformat))
    # and check 1d PSD in frequency space (spatial space doesn't work ...)
    x = gauss.rfreq 
    d = gauss.psd1d.real
    sigma = sigma_psd_freq
    fitval, expval = dofit(x, d, 0., sigma) 
    doplot(x, d, fitval, expval, 'PSD 1-d', xlabel='Frequency')
    pylab.savefig('gauss_psd1d.%s' %(figformat), format='%s' %(figformat))
    # pull slice from ACF
    x = numpy.arange(-gauss.xcen, gauss.nx-gauss.xcen, 1.)
    d = gauss.acf.real[round(gauss.ny/2.0)][:]
    mean = 0
    sigma_acf = sigma_x*numpy.sqrt(2)
    fitval, expval = dofit(x, d, mean, sigma_acf)
    doplot(x, d, fitval, expval, 'ACF 2-d slice', xlabel='Pixels')
    pylab.xlim(-200, 200)
    pylab.savefig('gauss_acf.%s' %(figformat), format='%s' %(figformat))
    # and check 1d ACF
    x = gauss.acfx
    d = gauss.acf1d.real
    sigma_acf = sigma_x*numpy.sqrt(2)
    fitval, expval = dofit(x, d, mean, sigma_acf)
    doplot(x, d, fitval, expval, 'ACF 1-d', xlabel='Pixels')
    pylab.savefig('gauss_acf1d.%s' %(figformat), format='%s' %(figformat))
    pylab.close()
    return

def compare1d_psd_acf():
    """Compare 1d ACF in physical coordinates to 1d PSD in physical coordinates, for two similar but different images."""
    im = TestImage(shift=True, nx=1000, ny=1000)
    scale = 100
    im.addSin(scale=scale)
    im.hanningFilter()
    im.zeroPad()
    im.calcAll(min_npix=1, min_dr=1)
    im.showImage()
    pylab.grid()
    pylab.savefig('compare1d_image1.%s' %(figformat), format='%s' %(figformat))
    im.showPsd2d()
    pylab.savefig('compare1d_psd2d1.%s' %(figformat), format='%s' %(figformat))
    im.showPsd1d()
    pylab.savefig('compare1d_psd1.%s' %(figformat), format='%s' %(figformat))
    im.showAcf1d()
    pylab.savefig('compare1d_acf1.%s' %(figformat), format='%s' %(figformat))
    im = TestImage(shift=True, nx=1000, ny=1000)
    im.addSin(scale=scale*2)
    im.hanningFilter()
    im.zeroPad()
    im.calcAll(min_npix=1, min_dr=1)
    im.showImage()
    pylab.grid()
    pylab.savefig('compare1d_image2.%s' %(figformat), format='%s' %(figformat))
    im.showPsd2d()
    pylab.savefig('compare1d_psd2d2.%s' %(figformat), format='%s' %(figformat))
    im.showPsd1d()
    pylab.savefig('compare1d_psd2.%s' %(figformat), format='%s' %(figformat))
    im.showAcf1d()
    pylab.savefig('compare1d_acf2.%s' %(figformat), format='%s' %(figformat))
    pylab.close()
    return

def inversion():
    """Generate some example images & invert them to reconstruct the original image."""
    im = TestImage(shift=True, nx=1000, ny=1000)
    #im.addEllipseGrid(gridX=200, gridY=100, semiX=50, semiY=25, value=1)
    im.addLines(width=20, spacing=200, value=1, angle=45)
    im.addSin(scale=300)
    im.hanningFilter()
    im.zeroPad()
    #cmap = pylab.cm.gray_r
    cmap = None
    clims = im.showImage(cmap=cmap)
    pylab.savefig('invert_image.%s' %(figformat), format='%s' %(figformat))
    im.calcAll(min_npix=1, min_dr=1)
    # Invert from ACF and show perfect reconstruction.
    im.invertAcf2d()
    im.invertPsd2d(useI=True)
    im.invertFft(useI=True)
    im.showImageI(clims=clims, cmap=cmap)
    pylab.savefig('invert_acf2d_good.%s' %(figformat), format='%s' %(figformat))
    # Invert from ACF 2d without phases
    im.invertAcf2d(usePhasespec=False, seed=42)
    im.invertPsd2d(useI=True)
    im.invertFft(useI=True)    
    im.showImageI(clims=clims, cmap=cmap)
    pylab.savefig('invert_acf2d_nophases.%s' %(figformat), format='%s' %(figformat))
    # Invert from ACF 1d with phases
    im.invertAcf1d(phasespec=im.phasespec)
    im.invertAcf2d(useI=True)
    im.invertPsd2d(useI=True)
    im.invertFft(useI=True)
    im.showImageI(clims=clims, cmap=cmap)
    pylab.savefig('invert_acf1d_phases.%s' %(figformat), format='%s' %(figformat))
    # Invert from ACF 1d without phases
    im.invertAcf1d(seed=42)
    im.invertAcf2d(useI=True)
    im.invertPsd2d(useI=True)
    im.invertFft(useI=True)
    im.showImageI(clims=clims, cmap=cmap)
    pylab.savefig('invert_acf1d_nophases.%s' %(figformat), format='%s' %(figformat))
    # Recalculate 1-d PSD and ACF from this last reconstructed image (ACF1d no phases)
    im2 = PImagePlots()
    im2.setImage(im.imageI)
    im2.calcAll(min_npix=1, min_dr=1)
    legendlabels=['Reconstructed', 'Original']
    im2.showPsd1d(comparison=im, legendlabels=legendlabels)
    pylab.savefig('invert_recalc_ACF_Psd1d.%s' %(figformat), format='%s' %(figformat))
    im2.showAcf1d(comparison=im, legendlabels=legendlabels)
    pylab.savefig('invert_recalc_ACF_Acf1d.%s' %(figformat), format='%s' %(figformat))
    # Invert from PSD and show perfect reconstruction.                          
    im.invertPsd2d()
    im.invertFft(useI=True)
    im.showImageI(clims=clims, cmap=cmap)
    pylab.savefig('invert_psd2d_good.%s' %(figformat), format='%s' %(figformat))
    # Invert from PSD 2d without phases
    im.invertPsd2d(usePhasespec=False, seed=42)
    im.invertFft(useI=True)
    im.showImageI(clims=clims, cmap=cmap)
    pylab.savefig('invert_psd2d_nophases.%s' %(figformat), format='%s' %(figformat))
    # Invert from PSD 1d with phases                                   
    im.invertPsd1d(phasespec=im.phasespec)
    im.invertPsd2d(useI=True)
    im.invertFft(useI=True)
    im.showImageI(clims=clims, cmap=cmap)
    pylab.savefig('invert_psd1d_phases.%s' %(figformat), format='%s' %(figformat))
    # Invert from PSD 1d without phases                                             
    im.invertPsd1d(seed=42)
    im.invertPsd2d(useI=True)
    im.invertFft(useI=True)
    im.showImageI(clims=clims, cmap=cmap)
    pylab.savefig('invert_psd1d_nophases.%s' %(figformat), format='%s' %(figformat))
    # Recalculate 1-d PSD and ACF from this last reconstructed image (PSD1d no phases)
    im2 = PImagePlots()
    im2.setImage(im.imageI)
    im2.calcAll(min_npix=1, min_dr=1)
    im2.showPsd1d(comparison=im, legendlabels=legendlabels)
    pylab.savefig('invert_recalc_PSD_Psd1d.%s' %(figformat), format='%s' %(figformat))
    im2.showAcf1d(comparison=im, legendlabels=legendlabels)
    pylab.savefig('invert_recalc_PSD_Acf1d.%s' %(figformat), format='%s' %(figformat))
    pylab.close()
    return
    

def elliptical():
    """Generate a test image with random ellipses and background noise."""
    im = TestImage(shift=True, nx=1000, ny=1000)
    im.addEllipseRandom(nEllipse=100, value=5)
    im.addNoise(sigma=1)
    im.hanningFilter()
    #im.zeroPad()
    im.calcAll(min_npix=2, min_dr=1)
    im.plotMore()
    pylab.savefig('elliptical.%s' %(figformat), format='%s' %(figformat))
    # Invert from ACF 1d without phases                                             
    im.invertAcf1d()
    im.invertAcf2d(useI=True)
    im.invertPsd2d(useI=True)
    im.invertFft(useI=True)
    im.showImageAndImageI()
    pylab.savefig('elliptical_invert.%s' %(figformat), format='%s' %(figformat))
    pylab.close()
    return


def clouds():
    """Read an example of the french group's cloud generation."""
    # oldCloud.npy and newCloud.npy are images of size 240x240 that cover a fov of 4.0 deg 
    #  (if cloud generation code is understood correctly).
    # old clouds
    oldClouds = numpy.load('oldCloud.npy')
    fov = 4.0 #rad_fov = 2.0
    nx = len(oldClouds)
    pixscale = fov / float(nx)
    im = PImagePlots(shift=True)
    im.setImage(oldClouds)
    im.showImage()
    pylab.savefig('clouds_oldimage.%s' %(figformat), format='%s' %(figformat))
    im.hanningFilter()
    im.calcAll(min_npix=2, min_dr=1)
    im.plotMore()
    pylab.savefig('clouds_old.%s' %(figformat), format='%s' %(figformat))    
    # new clouds
    newClouds = numpy.load('newCloud.npy')
    im2 = PImagePlots(shift=True)
    im2.setImage(newClouds)
    im2.showImage()
    pylab.savefig('clouds_newimage.%s' %(figformat), format='%s' %(figformat))
    im2.hanningFilter()
    im2.calcAll(min_npix=2, min_dr=1)
    im2.plotMore()
    pylab.savefig('clouds_new.%s' %(figformat), format='%s' %(figformat))
    # compare structure functions
    legendlabels=['Old clouds', 'New clouds']
    # translate x axis from pixels to degrees .. 240 pix = 3.0 deg (?)
    im.sfx = im.sfx *pixscale
    im2.sfx = im2.sfx *pixscale
    im.showSf(comparison=im2, legendlabels=legendlabels, linear=True)
    pylab.xlabel('Degrees')
    pylab.savefig('clouds_sf.%s' %(figformat), format='%s' %(figformat))
    # look at phase spectrum
    pylab.figure()
    n, b, p = pylab.hist(im.phasespec.flatten(), bins=75, range=[-numpy.pi, numpy.pi], 
                         alpha=0.2, label='Old clouds phases')
    n, b, p = pylab.hist(im2.phasespec.flatten(), bins=b, range=[-numpy.pi, numpy.pi], 
                         alpha=0.2, label='New clouds phases')
    pylab.legend(fancybox=True, fontsize='smaller')
    pylab.savefig('clouds_phasehist.%s' %(figformat), format='%s' %(figformat))
    # the phase spectrum seems to be flatly distributed between -pi and pi
    pylab.figure()
    pylab.subplot(121)
    pylab.title('Old clouds')
    pylab.imshow(im.phasespec, origin='lower')
    pylab.colorbar(shrink=0.6)
    pylab.subplot(122)
    pylab.title('New clouds')
    pylab.imshow(im2.phasespec, origin='lower')
    pylab.colorbar(shrink=0.6)
    pylab.suptitle('Phase spectrum')
    pylab.savefig('clouds_phasespec.%s' %(figformat), format='%s' %(figformat))
    pylab.show()
    return


if __name__ == '__main__':
    if do_example_lines:
        example_lines()
    if do_gaussian_example:
        gaussian_example()
    if do_compare1d_psd_acf:
        compare1d_psd_acf()
    if do_inversion:
        inversion()
    if do_elliptical:
        elliptical()
    if do_clouds:
        clouds()
