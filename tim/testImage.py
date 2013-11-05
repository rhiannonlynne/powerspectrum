import numpy
import pylab
#from pImage import PImage
from pImagePlots import PImagePlots

# Comments on additional useful resources: 
# book chapter 10
# book appendix (see fft calculations in appendix)

# further information (to be read) about structure functions & more complicated analysis
# Peter Coles (Nature) papers about SF phases
# see also Szalay & Landry(?) data-data / data-random / random-random FFT analysis (FFT in presence of gaps)
# 'compressive sensing' (Emanuel Condes, also David Donaho, @ Stanford Math, Terry Towel)


class TestImage(PImagePlots):
    def __init__(self, nx=1000, ny=1000, shift=True):
        """Initialize the test image, with an nx/ny size zero image."""
        # Note that numpy array translate to images in [y][x] order! 
        self.nx = int(nx)
        self.ny = int(ny)
        self.image = numpy.zeros((self.ny, self.nx), 'float')
        self.yy, self.xx = numpy.indices(self.image.shape)
        self.padx = 0.0
        self.pady = 0.0
        self.xcen = round(self.nx/2.0)
        self.ycen = round(self.ny/2.0)
        self.fimage = None
        self.psd2d = None
        self.phasespec = None
        self.psd1d = None
        self.acf2d = None
        self.acf1d = None
        self.shift = shift
        return
        
    def addNoise(self, sigma=1.0):
        """Add gaussian noise to the image, mean=0, sigma=noiseSigma."""
        noise = numpy.random.normal(loc=0, scale=sigma, size=(self.ny, self.nx))
        self.image += noise
        return

    def setFlatImage(self, value=1.0):
        """Set the image to a flat value of 'value'."""
        self.fimage = None
        self.image = numpy.zeros((self.ny, self.nx), 'float') + value
        return

    def addSin(self, scale=(2.*numpy.pi), value=1.0):
        """Add a sin variation to the images,
        z = sin(2pi*x/scale) * sin(2pi*y/scale), with peak of 'value'. """
        self.fimage = None
        z = numpy.sin(2.0*numpy.pi*self.xx/float(scale)) * numpy.sin(2.0*numpy.pi*self.yy/float(scale))
        self.image += z * value
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

    def addRectangle(self, widthX=50., widthY=20., value=1.0, cx=None, cy=None):
        """Add a rectangle with width widthX/widthY centered at cx/cy (if None, use center). """
        self.fimage = None
        if cx == None:
            cx = self.nx/2.0
        if cy == None:
            cy = self.ny/2.0
        tmpx = numpy.abs(self.xx - cx)/2.0
        tmpy = numpy.abs(self.yy - cy)/2.0
        rectangle = numpy.where((tmpx<=widthX/4.0) & (tmpy<=widthY/4.0), value, 0)
        self.image += rectangle
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
        circle = numpy.where(tmp<=radius**2, value, 0)
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
 
