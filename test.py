import pylab
from testImage import TestImage

im = TestImage()
im.addGaussian(xwidth=20, ywidth=20)
im.makeFft()
im.makePsd()
im.makeAcf()

im.showImage()
im.showFft()
im.showPsd2d()
im.showPsd1d()
im.showAcf()

pylab.show()
