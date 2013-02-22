import pylab
from testImage import TestImage

im = TestImage()
im.addGaussian(xwidth=100, ywidth=100)
im.showImage()

im.makeFft()
im.showFft()
im.makePsd()
im.showPsd1d()

im.makeFft(shift=True)
im.showFft()
im.makePsd()
im.showPsd1d()

pylab.show()
