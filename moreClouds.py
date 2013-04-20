import numpy
import pylab
from pImagePlots import PImagePlots

def ComputeStructureFunction(x0=0.0, x1=1.75, y1=0.04, ymm=0.08, xmax=10.):
    """Create the structure function based on an analytical model.
    
    This analytical model is determined by fitting data (done elsewhere).
    The variables x0, x1, y1, and ymm describe the structure function.
    x0=0, x1=1.75,y1=0.04, ymm=0.08 are defaults fit to the SDSS (?) structure function.
    ymm is related to the long-range grayscale extinction, while x1 and y2 are related to
    the inflection points in the structure function. """
    # define the x range
    xsf = numpy.arange(0, xmax, 0.1)
    # Calculate variable for the structure function.
    al = -(1/(x1-x0))*numpy.log(1-(y1/ymm))
    # Calculate the actual structure function.
    SF = ymm*(1.-numpy.exp(-al*xsf))   
    return xsf, SF

    
if __name__ == "__main__":
    xsf, SF = ComputeStructureFunction()
    # Plot result - looks like X in degrees. 
    pylab.figure()
    pylab.plot(xsf, SF)
    pylab.xlabel('Degrees')
    pylab.ylabel('Structure Function')
    pylab.title('SDSS structure function')
    pylab.grid()
    pylab.savefig('clouds_sf_SDSS.png', format='png')

    # Okay, let's try to translate this to an image we can use.     
    imsize = 1500 # pixels desired in final image
    rad_fov = 2.0 # degrees
    pixscale = 2*rad_fov / float(imsize)
    print 'Rad_fov', rad_fov, 'Imsize', imsize, 'Pixscale', pixscale, 'deg/pix', '(', pixscale*60.*60., 'arcsec/pix)'
    # set up 1d SF with desired scale and pixel scale
    condition = (xsf <= rad_fov)
    xr = xsf[condition] / pixscale # xr = in pixels, over range that want to simulate
    xrpix = numpy.arange(0, imsize, 1.0)
    sfpix = numpy.interp(xrpix, xr, SF[condition])

    # try making image
    im = PImagePlots(shift=True, nx= imsize, ny=imsize)
    im.invertSf(sfx=xrpix, sf=sfpix)
    im.invertAcovf1d()    
    im.invertAcovf2d(useI=True)
    im.invertPsd2d(useI=True)
    im.invertFft(useI=True)    
    im.showImageI()
    pylab.savefig('clouds_1.png', format='png')

    im.image = im.imageI.real
    #im.hanningFilter()
    im.calcAll(min_npix=2, min_dr=1)
    im.plotMore()
    pylab.savefig('clouds_1_dat.png', format='png')
    # Rescale sfx to be in physical (degrees)
    im.sfx = im.sfx * pixscale

    # make another image, as should see difference due to different random phases
    im2 = PImagePlots(shift=True, nx= imsize, ny=imsize)
    im2.invertSf(sfx=xrpix, sf=sfpix)
    im2.invertAcovf1d()
    im2.invertAcovf2d(useI=True)
    im2.invertPsd2d(useI=True)
    im2.invertFft(useI=True)
    im2.showImageI()
    pylab.savefig('clouds_2.png', format='png')

    # Compare structure functions, but scaled (due to loss of amplitude info with random phases)
    im2.image = im2.imageI.real
    im2.calcAll(min_npix=2, min_dr=1)
    im2.plotMore()
    pylab.savefig('clouds_2_dat.png', format='png')
    im2.sfx = im2.sfx * pixscale
    im2.sf = im2.sf/im2.sf.max()
    im.sf = im.sf/im.sf.max()
    im.showSf(linear=True, comparison=im2, legendlabels=['Clouds 1 (scaled)', 'Clouds 2 (scaled)'])
    rm = rad_fov
    condition = (numpy.abs(xsf - rm) < 0.2)
    SF2 = SF - SF.min()
    SF2 = SF2 / SF[condition].mean()
    pylab.plot(xsf, SF2, 'k:', label='Original/SDSS (scaled)')
    pylab.xlim(0, rm)
    pylab.ylim(0, 1.2)
    pylab.legend(numpoints=1, fancybox=True, fontsize='smaller')
    pylab.title('Structure Function')
    pylab.xlabel('Degrees')
    pylab.savefig('clouds_sf_new.png', format='png')
    


    pylab.show()

