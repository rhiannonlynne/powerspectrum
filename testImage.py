import numpy
import pylab
from scipy import fftpack

# Comments on additional useful resources: 
# book chapter 10
# book appendix (see fft calculations in appendix)
# Peter Coles (Nature) papers about SF phases
# see also Szalay & Landry(?) data-data / data-random / random-random FFT analysis (FFT in presence of gaps)
# 'compressive sensing' (Emanuel Condes, also David Donaho, @ Stanford Math, Terry Towel)

# TODO
# Investigate adding filter to FFT of image (to smooth edge effects)


class TestImage():
    def __init__(self, nx=500, ny=500):
        """Initialize the test image, with an nx/ny size zero image."""
        # Note that numpy array translate to images in [y][x] order! 
        self.nx = nx
        self.ny = ny        
        self.image = numpy.zeros((self.ny, self.nx), 'float')
        self.yy, self.xx = numpy.indices(self.image.shape)
        self.fimage = None
        return

    def setImage(self, imdat):
        """Set the image using an external numpy array"""
        self.image = imdat    
        self.ny, self.nx = self.image.shape
        self.yy, self.xx = numpy.indices(self.image.shape)
        self.fimage = None
        return

    def zeroPad(self, width=None):
        """Add padding to the outside of the image data. """
        offsetx = numpy.floor(self.nx / 4.0)
        offsety = numpy.floor(self.ny / 4.0)
        newy = self.ny + int(2*offsetx)
        newx = self.nx + int(2*offsety)
        newimage = numpy.zeros((newy, newx), 'float')
        newyy, newxx = numpy.indices(newimage.shape)
        condition = ((newxx >= offsetx) & (newxx < offsetx + self.nx) & (newyy >= offsety) & (newyy < offsety + self.ny))
        newimage[condition] = self.image.flatten()
        self.nx = newx
        self.ny = newy
        self.xx = newxx
        self.yy = newyy
        self.image = newimage
        self.fimage = None
        return
        
    def addNoise(self, sigma=1.0):
        """Add gaussian noise to the image, mean=0, sigma=noiseSigma."""
        noise = numpy.random.normal(loc=0, scale=sigma, size=(self.nx, self.ny))
        self.image += noise
        return

    def setFlatImage(self, value=1.0):
        """Set the image to a flat value of 'value'."""
        self.fimage = None
        self.image = numpy.zeros((self.ny, self.nx), 'float') + value
        return

    def addGaussian(self, xwidth=100., ywidth=100., xcen=None, ycen=None, value=1.0):
        """Add a 2-d gaussian to the image with widths xwidth/ywidth.
        Can specify xcen/ycen (default=Center of image) and peak value of gaussian."""
        if xcen == None:
            xcen = self.nx/2.0
        if ycen == None:
            ycen = self.ny/2.0
        self.fimage = None
        gaussian = numpy.exp(-(self.xx-xcen)**2/(2.0*xwidth**2) - (self.yy-ycen)**2/(2.0*ywidth**2))
        self.image += gaussian * value / gaussian.max()
        return

    def addLines(self, spacing=10, angle=0.0, value=1.0, width=1):
        """Add lines to the image at an angle of angle (wrt x axis),
        separated by spacing (having width of width) and with a value of value."""
        self.fimage = None
        # Create an array with the lines.
        angle = angle * numpy.pi / 180.0
        tmp = numpy.round(self.xx*numpy.cos(angle) - self.yy*numpy.sin(angle)) % spacing
        lines = numpy.where(((tmp<width/2.0) | (spacing-tmp<width/2.0)), value, 0)
        # Add to image.
        self.image += lines
        return

    def addCircle(self, radius=5.0, value=1.0, cx=None, cy=None):
        """Add a circle to cx/cy (if None, use center) of the image, with 'radius' and value of value."""
        self.fimage = None
        # Create a circle at the center.
        if cx == None:
            cx = self.nx/2.0
        if cy == None:
            cy = self.ny/2.0
        tmp = (self.xx - cx)**2 + (self.yy - cy)**2
        circle = numpy.where(tmp<radius, value, 0)
        self.image += circle
        return

    def addEllipseGrid(self, gridX=50, gridY=50, angle=None, semiX=5.0, semiY=2.0, value=1.0):
        """Add ellipses to the image on a regular grid with gridX/gridY spacing, at
        either random position angles (angle=None), or at a consistent angle.
        SemiX and SemiY describe the 'x' and 'y' length of the ellipse. """
        self.fimage = None
        gridx = numpy.arange(0, self.nx+gridX, gridX)
        gridy = numpy.arange(0, self.ny+gridY, gridY)
        if angle != None:            
            angles = numpy.zeros(len(gridx)*len(gridy), 'float') + angle * numpy.pi / 180.0
        if angle == None:
            angles = numpy.random.uniform(0, 2*numpy.pi, size=len(gridx)*len(gridy))
        count = 0
        for j in gridy:
            for i in gridx:
                angle = angles[count]
                count += 1
                xx = (self.xx - i) * numpy.cos(angle) + (self.yy - j) * numpy.sin(angle) 
                yy = -1*(self.xx - i) * numpy.sin(angle) + (self.yy - j) * numpy.cos(angle)
                tmp = ((xx)**2/float(semiX)**2 + (yy)**2/float(semiY)**2)
                ellipses = numpy.where(tmp<=1.0, value, 0)
                # Add to image.
                self.image += ellipses
        return
    
    def addEllipseRandom(self, nEllipse=50, angle=None, semiX=None, semiY=None, value=1.0):
        """Add nEllipse ellipses to the image on a random grid, at
        either random position angles (angle=None), or at a consistent angle.
        SemiX and SemiY describe the 'x' and 'y' length of the ellipse, and can here be specified as 'None',
        which will make these random values (between 5-30) as well. """
        self.fimage = None
        gridx = numpy.random.uniform(low=0, high=self.nx, size=nEllipse)
        gridy = numpy.random.uniform(low=0, high=self.ny, size=nEllipse)
        if angle != None:            
            angles = numpy.zeros(nEllipse, 'float') + angle * numpy.pi / 180.0
        if angle == None:
            angles = numpy.random.uniform(0, 2*numpy.pi, size=nEllipse)
        if semiX != None:
            semiXs = numpy.zeros(nEllipse, 'float') + semiX
        else:
            semiXs = numpy.random.uniform(low=5., high=30., size=nEllipse)
        if semiY != None:
            semiYs = numpy.zeros(nEllipse, 'float') + semiY
        else:
            semiYs = numpy.random.uniform(low=5., high=30., size=nEllipse)
        count = 0
        for j, i in zip(gridy, gridx):
            angle = angles[count]
            semiX = semiXs[count]
            semiY = semiYs[count]
            count += 1
            xx = (self.xx - i) * numpy.cos(angle) + (self.yy - j) * numpy.sin(angle) 
            yy = -1*(self.xx - i) * numpy.sin(angle) + (self.yy - j) * numpy.cos(angle)
            tmp = ((xx)**2/float(semiX)**2 + (yy)**2/float(semiY)**2)
            ellipses = numpy.where(tmp<=1.0, value, 0)
            # Add to image.
            self.image += ellipses
        return
 
    def makeFft(self):
        """Take the 2d FFT of the image (self.fimage), adding a shift to move the small spatial scales to the
        center of the FFT image to self.fimage2. Also calculates the frequencies. """
        self.fimage = fftpack.fft2(self.image)
        self.fimage2 = fftpack.fftshift(self.fimage)
        self.xfreq = fftpack.fftfreq(self.nx, 1.0)
        self.yfreq = fftpack.fftfreq(self.ny, 1.0)
        return


    def makePsd(self):
        """Calculate the power spectrum - 2d and 1d."""
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
        # Calculate the unique radius values (the bins we want to use for psd1d)
        rbinsize = 1.0
        r = numpy.arange(0, rvals.max()+rbinsize, rbinsize)
        rcenters = (r[1:] + r[:-1])/2.0
        # Sort the PSD2d by the radius values
        idx = numpy.argsort(rvals.flatten())
        dvals = self.psd2d2.flatten()[idx]
        rvals = rvals.flatten()[idx]
        # Calculate how many pixels are actually present in each radius bin (for weighting)
        # Number of pixels in each radius bin
        nvals = numpy.histogram(rvals, r)[0]
        # Value of image in each radial bin (sum)
        rprof = numpy.histogram(rvals, r, weights=dvals)[0] / nvals
        # Set the value of the 1d psd, and interpolate over any nans (where there were no pixels)
        self.psd1d = numpy.interp(rcenters, rcenters[rprof==rprof], rprof[rprof==rprof])
        self.rcenters = rcenters
        # frequencies for radius 
        self.rfreq = (fftpack.fftfreq(len(self.rcenters), rbinsize))
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
            # have to adjust y for the fact that fimage has 'shifted' axis versus fimage2 (but xfreq goes with fimage)
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

    def showImage(self, xlim=None, ylim=None):
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
        pylab.imshow(self.image, origin='lower')
        pylab.xlabel('X')
        pylab.ylabel('Y')
        cb = pylab.colorbar()
        pylab.xlim(x0, x1)
        pylab.ylim(y0, y1)
        return

    def showFft(self, real=True, imag=False):
        if ((real == True) & (imag==True)):
            p = 2
        else:
            p = 1            
        pylab.figure()        
        pylab.title('FFT')
        if real:
            pylab.subplot(1,p,1)
            pylab.title('Real')
            pylab.imshow(self.fimage2.real, origin='lower', extent=[self.xfreq.min(), self.xfreq.max(), self.yfreq.min(), self.yfreq.max()])
            pylab.xlabel('u')
            pylab.ylabel('v')
            cb = pylab.colorbar()
        if imag:
            pylab.subplot(1,p,p)
            pylab.title('Imaginary')
            pylab.imshow(self.fimage2.imag, origin='lower', extent=[self.xfreq.min(), self.xfreq.max(), self.yfreq.min(), self.yfreq.max()])
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
            pylab.imshow(self.psd2d2, origin='lower', extent=[self.xfreq.min(), self.xfreq.max(), self.yfreq.min(), self.yfreq.max()], norm=norml)
        else:
            pylab.imshow(self.psd2d2, origin='lower', extent=[self.xfreq.min(), self.xfreq.max(), self.yfreq.min(), self.yfreq.max()])
        cb = pylab.colorbar()
        return

    def showPsd1d(self):
        pylab.figure()
        pylab.subplot(121)
        pylab.semilogy(self.rfreq, self.psd1d)
        pylab.xlabel('Frequency')
        pylab.ylabel('1-D Power Spectrum')
        pylab.subplot(122)
        pylab.semilogy(self.rcenters, self.psd1d)
        pylab.xlabel('Spatial scale (pix)')
        return

    def showAcf(self):
        pylab.figure()
        pylab.title('ACF')
        pylab.imshow(self.acf.real, origin='lower')
        cb = pylab.colorbar()
        return
