import pylab
from testImage import TestImage

im = TestImage()
#im.makeFft()
#im.makePsd()
#im.makeAcf()
#im.showImage()
#im.showFft()
#im.showPsd2d()
#im.showPsd1d()
#im.showAcf()

# Plain image, with noise
im = TestImage()
im.addNoise()
im.zeroPad()
im.makeAll()
im.plotAll(title='Gaussian Noise')

im = TestImage()
im.addNoise()
im.zeroPad()
im.makeAll(binsize=8)
im.plotAll(title='Gaussian Noise')

# Gaussian image
im = TestImage()
im.addGaussian(xwidth=20, ywidth=20)
im.zeroPad()
im.makeAll()
im.plotAll(title='Gaussian')

# Sin, s=500
im = TestImage()
im.addSin(scale=500)
im.zeroPad()
im.makeAll()
im.plotAll(title='Sin, scale=500')
#im.showPsd1d()

# Sin, s=10
im = TestImage()
im.addSin(scale=10)
im.zeroPad()
im.makeAll()
im.plotAll(title='Sin, scale=10')

# Sin, s=5
im = TestImage()
im.addSin(scale=5)
im.zeroPad()
im.makeAll()
im.plotAll(title='Sin, scale=5')

# Rectangle
im = TestImage()
im.addRectangle(widthX=200., widthY=100., value=1.0)
im.addRectangle(widthX=100., widthY=50., value=-1.0)
im.zeroPad()
im.makeAll()
im.plotAll(title='Rectangle')

# Ring
im = TestImage()
im.addCircle(radius=200., value=1.)
im.addCircle(radius=100, value=-1.)
im.zeroPad()
im.makeAll()
im.plotAll(title='Ring')

# ellipses, grid
im = TestImage()
im.addEllipseGrid(gridX=80, gridY=80, semiX=20, semiY=40)
im.zeroPad()
im.makeAll()
im.plotAll(title='Ellipse, grid')

# ellipses, random locations
im = TestImage()
im.addEllipseRandom()
im.zeroPad()
im.makeAll()
im.plotAll(title='Ellipse, random')

# ellipses, random locations, with noise
im = TestImage()
im.addEllipseRandom(value=5)
im.addNoise()
im.zeroPad()
im.makeAll()
im.plotAll(title='Ellipse, random, with noise')


pylab.show()
