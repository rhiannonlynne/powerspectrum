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

def readSF(file):
    import useful_input as ui 
    data = ui.readDatafile(file, ('theta', 'sf'))
    # Convert arcminutes to degrees
    data['theta'] = data['theta'] / 60.0
    # Convert Tim's SF**2 to SF (mag**2 -> mag)
    data['sf'] = numpy.sqrt(data['sf'])
    return data['theta'], data['sf']

    
def rescaleImage(image, sigma_goal, kappa):
    sigma = numpy.std(image)
    mean = numpy.mean(image)
    print 'Before: mean/sigma/min/max:', mean, sigma, image.min(), image.max()
    image = image - mean
    image *= sigma_goal / sigma
    image += kappa
    # make sure no 'negative' clouds                                                                                 
    image = numpy.where(image<0, 0, image)
    sigma = numpy.std(image)
    mean = numpy.mean(image)
    print 'After: mean/sigma/min/max:', mean, sigma, image.min(), image.max()
    return image

def imageStats(image):
    print 'Image: mean/sigma/min/max:', numpy.std(image), numpy.mean(image), image.min(), image.max()
    return

if __name__ == "__main__":
    #xsf, SF = ComputeStructureFunction()
    xsf, SF = readSF('sf_arma.dat')
    # Plot result - looks like X in degrees. 
    pylab.figure()
    pylab.plot(xsf, SF)
    pylab.xlabel('Degrees')
    pylab.ylabel('Structure Function (mag)')
    pylab.title('ARMA structure function')
    pylab.grid()
    pylab.savefig('clouds_sf_ARMA.png', format='png')

    goal_rad_fov = 1.75 * numpy.sqrt(2)  # radius to furthest corner of fov for LSST
    sigma_goal = SF[numpy.abs(xsf-goal_rad_fov*numpy.sqrt(2))<0.1].mean()
    print 'sigma_goal', sigma_goal
    kappa = 1.

    # And also try to make an image with the same (final) pixel scale, but that will be created (first)
    #  larger and then scaled down, to avoid introducing artifacts from the ACovF1d being truncated. 
    final_imsize = 1500 # pixels desired in final image
    final_rad_fov = 1.75*numpy.sqrt(2) # degrees
    final_pixscale = 2*final_rad_fov / float(final_imsize)
    # Start with larger image 
    pixscale = final_pixscale
    rad_fov = final_rad_fov
    imsize = int(2*rad_fov / pixscale)
    if (imsize%2 != 0):
        imsize = imsize + 1
    xr = xsf / pixscale # xr = in pixels, over range that want to simulate
    print 'Image: Rad_fov', rad_fov, 'Imsize', imsize, 'Pixscale', pixscale, 'deg/pix', '(', pixscale*60.*60., 'arcsec/pix)'    
    im = PImagePlots(shift=True, nx=imsize, ny=imsize)
    im.makeImageFromSf(sfx=xr, sf=SF)
    im.showAcovf1d()
    im.showPsd2dI()
    # Trim image to desired final size
    trim = round((imsize - final_imsize)/2.0)
    print 'Trimming about %d pixels from each side' %(trim)
    image = im.imageI[trim:trim+final_imsize, trim:trim+final_imsize]
    image = rescaleImage(image.real, sigma_goal, kappa)
    imsize = len(image)
    pixscale = pixscale
    rad_fov = imsize/2.0*pixscale
    print 'Image After Trimming: Rad_fov', rad_fov, 'Imsize', imsize, 'Pixscale', pixscale, 'deg/pix', '(', pixscale*60.*60., 'arcsec/pix)'    
    im.setImage(image.real)
    imageStats(im.image)
    im.showImage(copy=True)
    im.hanningFilter()
    im.calcAll()
    im.showPsd2d()
    im.showAcovf2d()
    im.showAcovf1d(comparison=im)

    # And also try to make an image with the same (final) pixel scale, but that will be created (first)
    #  larger and then scaled down, to avoid introducing artifacts from the ACovF1d being truncated. 
    final_imsize = 1500 # pixels desired in final image
    final_rad_fov = 1.75*numpy.sqrt(2) # degrees
    final_pixscale = 2*final_rad_fov / float(final_imsize)
    # Start with larger image 
    pixscale = final_pixscale
    rad_fov = final_rad_fov*2. 
    imsize = int(2*rad_fov / pixscale)
    if (imsize%2 != 0):
        imsize = imsize + 1
    xr = xsf / pixscale # xr = in pixels, over range that want to simulate
    print 'Image: Rad_fov', rad_fov, 'Imsize', imsize, 'Pixscale', pixscale, 'deg/pix', '(', pixscale*60.*60., 'arcsec/pix)'    
    im2 = PImagePlots(shift=True, nx=imsize, ny=imsize)
    im2.makeImageFromSf(sfx=xr, sf=SF)
    im2.showAcovf1d()
    im2.showPsd2dI()
    # Trim image to desired final size
    trim = round((imsize - final_imsize)/2.0)
    print 'Trimming about %d pixels from each side' %(trim)
    image = im2.imageI[trim:trim+final_imsize, trim:trim+final_imsize]
    image = rescaleImage(image.real, sigma_goal, kappa)
    imsize = len(image)
    pixscale = pixscale
    rad_fov = imsize/2.0*pixscale
    print 'Image After Trimming: Rad_fov', rad_fov, 'Imsize', imsize, 'Pixscale', pixscale, 'deg/pix', '(', pixscale*60.*60., 'arcsec/pix)'    
    im2.setImage(image.real)
    imageStats(im2.image)
    im2.showImage(copy=True)
    im2.hanningFilter()
    im2.calcAll()
    im2.showPsd2d()
    im2.showAcovf2d()
    im2.showAcovf1d(comparison=im)

    # Compare structure functions, but scaled (due to loss of amplitude info with random phases)     
    im.sfx = im.sfx * pixscale
    im2.sfx = im2.sfx * pixscale
    scaleval = im.sf[numpy.abs(im.sfx - goal_rad_fov)<0.05].mean()
    print 'Cloud 1 scaleval', scaleval
    im.sf = im.sf/scaleval
    scaleval = im2.sf[numpy.abs(im2.sfx - goal_rad_fov)<0.05].mean()
    print 'Cloud 2 scaleval', scaleval
    im2.sf = im2.sf/scaleval
    im.showSf(linear=True, comparison=im2, legendlabels=['Clouds 1', 'Clouds 2'])
    scaleval = SF[numpy.where(numpy.abs(xsf - goal_rad_fov)<0.2)].mean()
    print 'ARMA scaleval', scaleval
    SF2 = SF / scaleval
    pylab.plot(xsf, SF2, 'k:', label='Original/ARMA')
    pylab.xlim(0, goal_rad_fov*1.2)
    pylab.ylim(0, 1.2)
    pylab.legend(numpoints=1, fancybox=True, fontsize='smaller')
    pylab.title('Structure Function (mag)')
    pylab.xlabel('Degrees')

    
    pylab.show()
    exit()
    
