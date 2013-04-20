# This snippet of code was to play around with the reflectEdges method & hanning filter, looking at the effect on 
# the 2-d transforms (look for artifacts). 

# The wider widths for the reflect edges (which bring edges into the final image) definitely make more artifacts. 
# The hanning filter also seems to introduce its own artifacts (but which perhaps could be removed?). 

import numpy
import pylab
from testImage import TestImage

for i in range(1, 5):
    im =TestImage(shift=True, nx=500, ny=500)
    im.addSin(scale=100)
    #im.addLines(angle=30, width=20, spacing=50)
    #im.image += 10.0
    width=500/float(i)
    print width
    im.reflectEdges(width=width)
    im.hanningFilter()
    #im.showImage()
    im.calcAll()
    im.plotMore()
    pylab.suptitle('Reflect edges (width %f) and hanning filter' %(width))

width= 0
for i in (0, 1, 2, 3, 4):
    im = TestImage(shift=True, nx=500, ny=500)
    im.addSin(scale=100)
    #im.addLines(angle=30, width=20, spacing=50)
    #im.image += 10.0
    if i == 0:
        title = 'Plain image only'
    if i == 1:
        im.hanningFilter()
        title = 'Hanning filter only'
    if i == 2:
        im.zeroPad()
        title = 'Zero padding only' 
    if i == 3:
        im.zeroPad()
        im.hanningFilter()
        title = 'Zero pad and hanning filter'
    if i == 4:
        im.reflectEdges(width=None)
        im.hanningFilter()
        title = 'Reflect edges (default) and hanning filter'
    #im.showImage()
    im.calcAll()
    im.plotMore()    
    if i == 0:
        im.plotMore(useClims=False)
    pylab.suptitle(title)


pylab.show()
