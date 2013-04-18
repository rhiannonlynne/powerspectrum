import pylab
from testImage import TestImage

# Plain image, with noise
im = TestImage()
im.addLines(width=10, spacing=50, value=5)
#im.addNoise()
im.hanningFilter()
im.zeroPad()
im.calcAll()
#im.plotAll(title='Gaussian Noise')
im.plotMore(title='Gaussian Noise')
pylab.show()
exit()

# Gaussian image
im = TestImage()
im.addGaussian(xwidth=20, ywidth=20)
im.hanningFilter()
im.zeroPad()
im.calcAll()
im.plotAll(title='Gaussian')

# Sin, s=100
im = TestImage()
im.addSin(scale=100)
im.hanningFilter()
im.zeroPad()
im.calcAll()
im.plotAll(title='Sin, scale=100')

# Sin, s=50
im = TestImage()
im.addSin(scale=50)
im.hanningFilter()
im.zeroPad()
im.calcAll()
im.plotAll(title='Sin, scale=50')

# Sin, s=10
im = TestImage()
im.addSin(scale=10)
im.hanningFilter()
im.zeroPad()
im.calcAll()
im.plotAll(title='Sin, scale=10')

# Lines, 45 degrees.
im = TestImage()
im.addLines(spacing=20, width=5, value=2, angle=45)
im.hanningFilter()
im.zeroPad()
im.calcAll()
im.plotAll(title='Lines, 45 degrees')

# Lines, 90 degrees.
im = TestImage()
im.addLines(spacing=20, width=5, value=2, angle=90)
im.hanningFilter()
im.zeroPad()
im.calcAll()
im.plotAll(title='Lines, 90 degrees')

# Rectangle
im = TestImage()
im.addRectangle(widthX=200., widthY=100., value=1.0)
im.addRectangle(widthX=100., widthY=50., value=-1.0)
im.hanningFilter()
im.zeroPad()
im.calcAll()
im.plotAll(title='Rectangle')

# Ring
im = TestImage()
im.addCircle(radius=200., value=1.)
im.addCircle(radius=100, value=-1.)
im.hanningFilter()
im.zeroPad()
im.calcAll()
im.plotAll(title='Ring')

# ellipses, grid
im = TestImage()
im.addEllipseGrid(gridX=80, gridY=80, semiX=20., semiY=40., angle=45.)
im.hanningFilter()
im.zeroPad()
im.calcAll()
im.plotAll(title='Ellipse, grid')

# ellipses, random locations
im = TestImage()
im.addEllipseRandom(value=5)
im.hanningFilter()
im.zeroPad()
im.calcAll()
im.plotAll(title='Ellipse, random')

# ellipses, random locations, with noise
#im = TestImage()
#im.addEllipseRandom(value=5)
im.addNoise()
#im.zeroPad()
im.calcAll()
im.plotAll(title='Ellipse, random, with noise')

pylab.show()
