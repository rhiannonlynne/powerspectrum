# Run a test to evaluate parameters of 2d gaussian through FFT/PSD/ACF
# In this case, looking at various image sizes and gaussian sigmas to evaluate what changes with these basic
# changes. (looking at the fit coefficients in particular). 

import numpy
import pylab
import testImage as ti
from scipy.optimize import curve_fit
from scipy import fftpack

# Gaussian function. 
def gaussian(x, *p):
    A, mu, sigma = p
    return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))

# Simple function to generate expected and fit data values.
def dofit(x, d, mean, sigma):
    p = (d.max().real, mean, sigma)
    coeff, var_matrix = curve_fit(gaussian, x, d, p0=p)
    fitval = gaussian(x, *coeff)
    expval = gaussian(x, *p)
    #print '  Fit:', coeff,  ' Exp:', p
    return fitval, expval, coeff
def doplot(x, d, fitval, expval, title):
    pylab.figure()
    pylab.plot(x, d, 'b.', label='data')
    pylab.plot(x, fitval, 'r-', label='fit')
    pylab.plot(x, expval, 'k:', label='expected')
    pylab.legend(numpoints=1, fancybox=True, fontsize='small')
    pylab.title(title)
    return

sigma = [2.5, 10.0, 25.0]
size = [500., 750., 1000.0, 1500.0]
ratio = []
pylab.figure()
pylab.title('1d PSD')
for s in sigma:
    sigma_psd_fit = []
    sigma_psd_exp = []
    for sx in size:
        factor = (sx)/(2.0*numpy.pi)
        print "sx, s", sx, s
        # Generate 2d gaussian image
        gauss = ti.TestImage(nx=sx, ny=sx)
        sigma_x = s
        gauss.addGaussian(xwidth=sigma_x, ywidth=sigma_x, value=1.0)        
        gauss.calcAll()
        # So expect gaussian to FT to another gaussian, but now with mu=0 and sigma=1/(2*PI*sigma_x)
        sigma_fft = 1/(2.0*numpy.pi*sigma_x)
        # Now look at PSD - in frequency space first.
        x = gauss.xfreq
        idx = numpy.argsort(x)
        x = x[idx]
        d = fftpack.ifftshift(gauss.psd2d)[0][:].real
        d = d[idx]
        # because PSD is square of FFT (which was a gaussian), expect mu = 0 but
        #  sigma = sigma_fft / sqrt(2) == (1/sigma_x*2*PI)*1/sqrt(2)
        sigma_psd_freq = sigma_fft/numpy.sqrt(2)
        fitval, expval, coeff = dofit(x, d, 0., sigma_psd_freq)
        # And now look at 2d psd in 'pixel' scales.
        x = numpy.arange(-gauss.xcen, gauss.nx-gauss.xcen, 1.0)
        d = gauss.psd2d[round(gauss.ny/2.0)][:].real
        # .. because of pixel 'scaling' expect sigma = sigma_x*sqrt(2)/2*PI  == 1/(sigma_psd_freq*4*PI^2)
        #sigma_psd_pix = 1/(sigma_x*numpy.sqrt(2))*factor
        sigma_psd_pix = 1/(sigma_x*numpy.sqrt(2))*numpy.sqrt(gauss.nx*gauss.ny)/(2.0*numpy.pi)        
        p = (gauss.psd2d.max().real, 0.0, sigma_psd_pix)
        fitval, expval, coeff = dofit(x, d, 0.0, sigma_psd_pix)
        sigma_psd_fit.append(coeff[2])
        sigma_psd_exp.append(p[2])
        print 'psd2d spatial', coeff, p
    
        # And now can look at 1d PSD
        x = gauss.psdx
        d = gauss.psd1d
        # because this is just 1d azimuthal average of 2d PSD in pixel space, expect sigma=sigma_psd_pix 
        # (although this doesn't quite work .. not sure why).
        #pylab.plot(x, d/d.max(), label='xsize %d sigma %.1f' %(sx, s))
        pylab.plot(x, d/sigma_x**4, label='xsize %d sigma %.1f' %(sx, s))
        #pylab.plot(x, fitval, 'r-', label='fit %d %.1f' % (sx, s))
        #pylab.plot(x, expval, 'k:')#, label='expected')

        # Now on to the ACF
        x = numpy.arange(-gauss.xcen, gauss.nx-gauss.xcen, 1.)
        d = gauss.acf.real[round(gauss.ny/2.0)][:]
        # Okay, ACF is actual inverse FFT of PSD, so in pixel scales, expect sigma_acf = 1/(sigma_psd_freq*2*PI)
        #   ... or = sigma_x*sqrt(2) because of squaring the gaussian during the PSD creation. 
        #sigma_acf = 1/(sigma_psd_freq*2.0*numpy.pi)
        sigma_acf = sigma_x*numpy.sqrt(2)
        fitval, expval, coeff = dofit(x, d, 0.0, sigma_acf)
        p = (d.max().real, 0.0, sigma_acf)
        print 'acf', coeff, p

    sigma_psd_exp = numpy.array(sigma_psd_exp)
    sigma_psd_fit = numpy.array(sigma_psd_fit)
    """
    pylab.plot(sigma_psd_fit, sigma_psd_fit/sigma_psd_exp, 'k.')
    pylab.plot(sigma_psd_fit, sigma_psd_fit/sigma_psd_exp, 'k-')
    """
    ratio.append((sigma_psd_fit/sigma_psd_exp).mean())
    print "RATIO", sigma_psd_fit/sigma_psd_exp, sigma, sx

pylab.legend(loc='lower right', fancybox=True, fontsize='small')

print "RATIO list", ratio
pylab.show()
exit()
pylab.figure()
ratio = numpy.array(ratio)
pylab.plot(size, ratio)
pylab.xlabel('size')
pylab.ylabel('ratio')
pylab.show()

