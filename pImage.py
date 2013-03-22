import numpy
import pylab
from scipy import fftpack

# For plotting an image in log scale
from matplotlib.colors import LogNorm

# For determining the best bin width for a histogram
try:
    from astroML.plotting import hist
    use_astroML = True
except ImportError:
    use_astroML = False

class PImage():
    def __init__(self):
        """Init. Does nothing."""
        # Note that numpy array translate to images in [y][x] order! 
        return

    def setImage(self, imdat, copy=False):
        """Set the image using an external numpy array. If 'copy' is true, makes a copy. """
        if copy:
            self.image = numpy.copy(imdat)
        else:
            self.image = imdat    
        self.ny, self.nx = self.image.shape
        self.yy, self.xx = numpy.indices(self.image.shape)
        self.padx = 0.0
        self.pady = 0.0
        self.fimage = None
        self.fimage2 = None
        return

    def zeroPad(self, width=None):
        """Add padding to the outside of the image data. Default width = 1/4 of image."""
        offsetx = numpy.floor(self.nx / 4.0)
        offsety = numpy.floor(self.ny / 4.0)
        self.padx = offsetx
        self.pady = offsety
        newy = self.ny + int(2*offsetx)
        newx = self.nx + int(2*offsety)
        newimage = numpy.zeros((newy, newx), 'float')
        newyy, newxx = numpy.indices(newimage.shape)
        condition = ((newxx >= offsetx) & (newxx < offsetx + self.nx) &
                     (newyy >= offsety) & (newyy < offsety + self.ny))
        newimage[condition] = self.image.flatten()
        self.nx = newx
        self.ny = newy
        self.xx = newxx
        self.yy = newyy
        self.image = newimage
        self.fimage = None
        self.fimage2 = None
        return
        
    def makeFft(self):
        """Take the 2d FFT of the image (self.fimage), adding a shift to move the small spatial scales to the
        center of the FFT image to self.fimage2. Also calculates the frequencies. """
        self.fimage = fftpack.fft2(self.image)
        self.fimage2 = fftpack.fftshift(self.fimage)
        self.xfreq = fftpack.fftfreq(self.nx, 1.0)
        self.yfreq = fftpack.fftfreq(self.ny, 1.0)
        return

    def makePsd(self, binsize=None):
        """Calculate the power spectrum - 2d and 1d.
        If binsize is defined, this will be used. Otherwise, the optimum 'knuth' binsize is determined
         using astroML (if available). If astroML is not available, and binsize is not defined, a default value
         of 5.5 pixels is used. """
        if self.fimage == None:
            print 'FFT needed first: calling makeFft'
            self.makeFft()
        # Calculate 2d power spectrum
        self.psd2d = numpy.absolute(self.fimage)**2.0
        # Use both shifted and non-shifted FFT because 1d PSD needs shifted.
        self.psd2d2 = numpy.absolute(self.fimage2)**2.0
        # Calculate 1d power spectrum                
        #  - use shifted FFT so that can create radial bins from center.        
        xcen = round(self.nx/2.0)
        ycen = round(self.ny/2.0)
        # Calculate all the radius values for all pixels
        rvals = numpy.hypot((self.xx-xcen), (self.yy-ycen))
        # Sort the PSD2d by the radius values and make flattened representations of these.
        idx = numpy.argsort(rvals.flatten())
        dvals = self.psd2d2.flatten()[idx]
        rvals = rvals.flatten()[idx]
        if binsize != None:
            # User-specified binsize.
            self.psd_binsize = binsize
            b = numpy.arange(0, rvals.max() + binsize, binsize)
        elif use_astroML:
            # Use astroML to determine best (equal-size) binsize for radial binning. 
            pylab.figure()            
            n, b, p = hist(rvals, bins='knuth')
            pylab.close()
            self.psd_binsize = (b[1] - b[0])
        else:
            # Use best guess. 
            self.psd_binsize = 5.5
            b = numpy.arange(0, rvals.max() + binsize, binsize)
        print 'Using binsize of %.3f' %(self.psd_binsize)
        # Calculate how many pixels are actually present in each radius bin (for weighting)
        nvals = numpy.histogram(rvals, bins=b)[0]
        # Calculate the value of the image in each radial bin (weighted by the # of pixels in each bin)
        rprof = numpy.histogram(rvals, bins=b, weights=dvals)[0] / nvals
        # Calculate the central radius values used in the histograms
        rcenters =  (b[1:] + b[:-1])/2.0
        # Set the value of the 1d psd, and interpolate over any nans (where there were no pixels)
        self.psd1d = numpy.interp(rcenters, rcenters[rprof==rprof], rprof[rprof==rprof])
        self.rcenters = rcenters
        # frequencies for radius 
        self.rfreq = (fftpack.fftfreq(len(self.rcenters), self.psd_binsize))
        return

    def makeAcf(self):
        """Calculate the auto correlation function. """
        self.acf = fftpack.fftshift(fftpack.ifft2(self.psd2d))
        return

    def showXSlice(self, y=None, source='image'):
        """Plot a 1d slice through the image (or fft or psd), at y (if None, defaults to center)."""
        if y == None:
            y = round(self.ny / 2.0)
        if source == 'image':
            x = numpy.arange(0, self.nx)
            pylab.plot(x, self.image[y][:])
        elif source == 'fft':
            # have to adjust y for the fact that fimage has 'shifted' axis versus fimage2
            #  (but xfreq goes with fimage)
            y = self.ny/2.0 - y
            pylab.plot(self.xfreq, self.fimage[y][:].real, 'k.')
        elif source == 'psd':
            y = self.ny/2.0 - y
            pylab.plot(self.xfreq, self.psd2d[y][:], 'k.')
        elif source == 'acf':
            x = numpy.arange(0, self.nx)
            pylab.plot(x, self.acf[y][:])
        else:
            raise Exception('Source must be one of image/fft/psd/acf')
        return

    def showYSlice(self, x=None):
        """Plot a 1d slice through the image, at x (if None, defaults to center)."""
        if x == None:
            x = round(self.nx/2.0)
        if source == 'image':
            y = numpy.arange(0, self.ny)
            pylab.plot(y, self.image[:][x])
        elif source == 'fft':
            pylab.plot(self.yfreq, self.fimage[:][x].real, 'k.')
        elif source == 'psd':
            pylab.plot(self.yfreq, self.psd2d[:][x], 'k.')
        elif source == 'acf':
            y = numpy.arange(0, self.ny)
            pylab.plot(y, self.acf[:][x])
        else:
            raise Exception('Source must be one of image/fft/psd/acf.')
        return

    def showImage(self, xlim=None, ylim=None, clims=None):
        pylab.figure()
        pylab.title('Image')
        if xlim == None:
            x0 = 0
            x1 = self.nx
        else:
            x0 = xlim[0]
            x1 = xlim[1]
        if ylim == None:
            y0 = 0
            y1 = self.ny
        else:
            y0 = ylim[0]
            y1 = ylim[1]
        if clims == None:
            pylab.imshow(self.image, origin='lower')
        else:
            pylab.imshow(self.image, origin='lower', vmin=clims[0], vmax=clims[1])
        pylab.xlabel('X')
        pylab.ylabel('Y')
        cb = pylab.colorbar()
        pylab.xlim(x0, x1)
        pylab.ylim(y0, y1)
        return

    def showFft(self, real=True, imag=False, clims=None):
        if ((real == True) & (imag==True)):
            p = 2
        else:
            p = 1            
        pylab.figure()        
        if real:
            pylab.subplot(1,p,1)
            pylab.title('Real FFT')
            if clims==None:
                pylab.imshow(self.fimage2.real, origin='lower',
                             extent=[self.xfreq.min(), self.xfreq.max(), self.yfreq.min(), self.yfreq.max()])
            else:
                pylab.imshow(self.fimage2.real, origin='lower', vmin=clims[0], vmax=clims[1],
                             extent=[self.xfreq.min(), self.xfreq.max(), self.yfreq.min(), self.yfreq.max()])
            pylab.xlabel('u')
            pylab.ylabel('v')
            cb = pylab.colorbar()
        if imag:
            pylab.subplot(1,p,p)
            pylab.title('Imaginary FFT')
            if clims == None:
                pylab.imshow(self.fimage2.imag, origin='lower',
                             extent=[self.xfreq.min(), self.xfreq.max(), self.yfreq.min(), self.yfreq.max()])
            else:
                pylab.imshow(self.fimage2.imag, origin='lower', vmin=clims[0], vmax=clims[1],
                             extent=[self.xfreq.min(), self.xfreq.max(), self.yfreq.min(), self.yfreq.max()])
            pylab.xlabel('u')
            pylab.ylabel('v')
            cb = pylab.colorbar()
        return

    def showPsd2d(self, log=True):
        pylab.figure()
        pylab.title('2d Power Spectrum')
        if log==True:
            from matplotlib.colors import LogNorm
            norml = LogNorm()
            pylab.imshow(self.psd2d2, origin='lower', extent=[self.xfreq.min(), self.xfreq.max(),
                                                              self.yfreq.min(), self.yfreq.max()], norm=norml)
        else:
            pylab.imshow(self.psd2d2, origin='lower', extent=[self.xfreq.min(), self.xfreq.max(),
                                                              self.yfreq.min(), self.yfreq.max()])
        cb = pylab.colorbar()
        pylab.xlabel('u')
        pylab.ylabel('v')
        return

    def showPsd1d(self):
        pylab.figure()
        pylab.subplot(121)
        pylab.semilogy(self.rfreq, self.psd1d, 'k.')
        pylab.xlabel('Frequency')
        pylab.ylabel('1-D Power Spectrum')
        pylab.subplot(122)
        pylab.semilogy(self.rcenters, self.psd1d, 'k.')
        pylab.xlabel('Spatial scale (pix)')
        pylab.title('1-d Power Spectrum, Radial binsize=%.3f' %(self.psd_binsize))
        return

    def showAcf(self):
        pylab.figure()
        pylab.title('ACF')
        pylab.imshow(self.acf.real, origin='lower')
        pylab.xlabel('X')
        pylab.ylabel('Y')
        cb = pylab.colorbar()
        return

    def makeAll(self, binsize=None):
        self.makeFft()
        self.makePsd(binsize=binsize)
        self.makeAcf()
        return

    def plotAll(self, title=None):
        pylab.figure()        
        pylab.subplots_adjust(left=0.1, right=0.97, wspace=0.45, hspace=0.2)
        ax1 = pylab.subplot2grid((2,3),(0,0))
        pylab.imshow(self.image, origin='lower')
        pylab.xticks(rotation=45)
        pylab.xlabel('X')
        pylab.ylabel('Y')
        cb = pylab.colorbar(shrink=0.7)
        clims = cb.get_clim()
        pylab.title('Image', fontsize=12)
        ax2 = pylab.subplot2grid((2,3), (0,1))
        pylab.imshow(self.fimage2.real, origin='lower', vmin=clims[0], vmax=clims[1],
                     extent=[self.xfreq.min(), self.xfreq.max(), self.yfreq.min(), self.yfreq.max()])
        pylab.xticks(rotation=45)
        pylab.xlabel('u')
        pylab.ylabel('v')
        cb = pylab.colorbar(shrink=0.7)
        pylab.title('Real FFT', fontsize=12)
        ax3 = pylab.subplot2grid((2,3), (1,0))
        norml = LogNorm()
        pylab.imshow(self.psd2d2, origin='lower', extent=[self.xfreq.min(), self.xfreq.max(),
                                                          self.yfreq.min(), self.yfreq.max()], norm=norml)
        cb = pylab.colorbar(shrink=0.7)
        pylab.xticks(rotation=45)
        pylab.xlabel('u')
        pylab.ylabel('v')
        pylab.title('2d-PS', fontsize=12)
        ax4 = pylab.subplot2grid((2,3), (1,1))
        pylab.imshow(self.acf.real, origin='lower')
        pylab.xticks(rotation=45)
        pylab.xlabel('X')
        pylab.ylabel('Y')
        cb = pylab.colorbar(shrink=0.7)
        pylab.title('ACF', fontsize=12)
        ax5 = pylab.subplot2grid((2,3), (0,2), rowspan=2)
        maxradius_image = numpy.sqrt((self.nx/2.0-self.padx)**2 + (self.ny/2.0-self.pady)**2)
        condition = (self.rcenters <= maxradius_image)
        pylab.semilogy(self.rcenters[condition], self.psd1d[condition], 'k-')
        pylab.xticks(rotation=45)
        pylab.xlabel('Spatial Scale (pix)')
        pylab.title('1-D PSD\n Radial binsize=%.3f' %(self.psd_binsize), fontsize=12)
        pos1 = ax2.get_position().bounds
        pos2 = ax4.get_position().bounds
        ax5.set_position([0.77, 0.2, pos2[2]*1.2, pos2[3]*1.5]) 
        if title!=None:
            pylab.suptitle(title, fontsize=14)
        return
