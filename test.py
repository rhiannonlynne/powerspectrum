import pylab
from testImage import TestImage

im = TestImage()
im.addGaussian(xwidth=5, ywidth=5)
im.showImage()

im.makeFft(shift=True)
im.showFft()
im.makePsd()
im.showPsd1d()

pylab.show()
