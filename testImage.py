import numpy
import pylab
from scipy import fftpack
import radialProfile 

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
        self.yy, self.yy = numpy.indices(self.image.shape)
        self.fimage = None
        return

    def addNoise(self, sigma=1.0):
        """Add gaussian noise to the image, mean=0, sigma=noiseSigma."""
        noise = numpy.random.normal(loc=0, scale=noiseSigma, size=(self.nx, self.ny))
        self.image += noise
        return

    def setFlatImage(self, value=1.0):
        """Set the image to a flat value of 'value'."""
        self.fft = None
        self.image = numpy.zeros((self.nx, self.ny), 'float') + value
        return

    def addLines(self, spacing=10, angle=0.0, value=1.0, width=1):
        """Add lines to the image at an angle of angle (wrt x axis),
        separated by spacing (having width of width) and with a value of value."""
        self.fft = None
        # Create an array with the lines.
        angle = angle * numpy.pi / 180.0
        tmp = numpy.round(self.xx*numpy.cos(angle) - self.yy*numpy.sin(angle)) % spacing
        lines = numpy.where(((tmp<width/2.0) | (spacing-tmp<width/2.0)), value, 0)
        # Add to image.
        self.image += lines
        return

    def addCircle(self, radius=5.0, value=1.0, cx=None, cy=None):
        """Add a circle to cx/cy (if None, use center) of the image, with 'radius' and value of value."""
        self.fft = None
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
        either random position angles (angle=None), or at a consistent angle. """
        self.fft = None
        gridx = numpy.arange(0, self.nx+gridX, gridX)
        gridy = numpy.arange(0, self.ny+gridY, gridY)
        # Try for zero position angle first ..
        if angle != None:            
            angle = angle * numpy.pi / 180.0
        for i in gridx:
            for j in gridy:
                # If angle was not specified, pick a random rotation angle. 
                if angle == None:
                    angle = numpy.random.uniform(0, 2*numpy.pi, size=1)
                # Ignore angle for now ... (TODO)
                tmp = ((self.xx-i)**2/float(semiX)**2 + (self.yy-j)**2/float(semiY)**2)
                ellipses = numpy.where(tmp<=1.0, value, 0)
                # Add to image.
                self.image += ellipses
        return
 
    def makeFft(self, shift=False):
        self.fimage = fftpack.fft(self.image)
        if shift:
            self.fimage = fftpack.fftshift(self.fimage)
        return

    def makePsd(self):
        """Calculate the power spectrum - 2d and 1d."""
        if self.fimage == None:
            print 'FFT needed first: calling makeFft (with no shift)'
            self.makeFft()
        # Calculate 2d power spectrum
        self.psd2d = numpy.abs(self.fimage)**2.0
        # Calculate 2d power spectrum
        self.psd1d = radialProfile.azimuthalAverage(self.psd2d)
        return

    def makeSF(self):
        """Calculate the structure function. """        
        pass
    
    def showImage(self, xlim=None, ylim=None):
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
        pylab.show()

    def showFft(self):
        pylab.imshow(self.fimage.real, origin='lower')
        pylab.xlabel('X')
        pylab.ylabel('Y')
        cb = pylab.colorbar()
        pylab.show()

    def showPsd2d(self):
        pylab.imshow(self.psd2d, origin='lower')
        cb = pylab.colorbar()
        pylab.show()

    def showPsd1d(self):
        pylab.semilogy(self.psd1d)
        pylab.xlabel('Spatial Frequency')
        pylab.ylabel('1-D Power Spectrum')
        pylab.show()
