import numpy as np
import pImage as p
import pylab

oldCLouds = np.load('oldclouds.npy')
newClouds = np.load('newclouds.npy')

ack = p.PImage()
cloudim = oldCLouds.copy()
ack.setImage(cloudim)
ack.zeroPad()

ack.makeAll()
ack.plotAll(title='Old Clouds')

cloudim = newClouds.copy()
ack.setImage(cloudim)
ack.zeroPad()

ack.makeAll()
ack.plotAll(title='New Clouds')

pylab.show()
