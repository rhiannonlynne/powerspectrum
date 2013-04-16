import numpy
from scipy import fftpack

class PImage():
    def __init__(self, shift=True):
        """Init. Does nothing."""
        # Note that numpy array translate to images in [y][x] order!
        self.shift = shift
        return

    def setImage(self, imdat, copy=False):
        """Set the image using an external numpy array. If 'copy' is true, makes a copy. """
        if copy:
            self.image = numpy.copy(imdat)
        else:
            self.image = imdat    
        self.ny, self.nx = self.image.shape
        self.yy, self.xx = numpy.indices(self.image.shape)
        self.padx = 0.0
        self.pady = 0.0
        self.xcen = round(self.nx/2.0)
        self.ycen = round(self.ny/2.0)
        return

    def zeroPad(self, width=None):
        """Add padding to the outside of the image data. Default width = 1/4 of image."""
        if width != None:
            self.padx = numpy.floor(width)
            self.pady = numpy.floor(width)
        else:
            self.padx = numpy.floor(self.nx / 4.0)
            self.pady = numpy.floor(self.ny / 4.0)
        newy = self.ny + int(2*self.padx)
        newx = self.nx + int(2*self.pady)
        newimage = numpy.zeros((newy, newx), 'float')
        newyy, newxx = numpy.indices(newimage.shape)
        condition = ((newxx >= self.padx) & (newxx < self.padx + self.nx) &
                     (newyy >= self.pady) & (newyy < self.pady + self.ny))
        newimage[condition] = self.image.flatten()
        self.nx = newx
        self.ny = newy
        self.xx = newxx
        self.yy = newyy
        self.image = newimage
        self.xcen = round(self.nx/2.0)
        self.ycen = round(self.ny/2.0)
        return

    def hanningFilter(self, rmax=None):
        """Apply a radial hanning filter to the image data.
        This removes noise in the FFT resulting from discontinuities in the edges of the images."""
        if rmax==None:
            rmax = min(self.nx, self.ny) / 2.0
        rvals = numpy.hypot((self.xx-self.xcen), (self.yy-self.ycen)) 
        hanning = numpy.where(rvals<=rmax, (0.5-0.5*numpy.cos(numpy.pi*(1-rvals/rmax))), 0.0)
        self.image *= hanning
        return
        
    def calcFft(self):
        """Calculate the 2d FFT of the image (self.fimage).
        If 'shift', adds a shift to move the small spatial scales to the
        center of the FFT image to self.fimage. Also calculates the frequencies. """
        # Generate the FFT (note, scipy places 0 - largest spatial scale frequencies - at corners)
        self.fimage = fftpack.fft2(self.image)
        if self.shift:
            # Shift the FFT to put the largest spatial scale frequencies at center
            self.fimage = fftpack.fftshift(self.fimage)
        # Note, these frequencies follow unshifted order (0= first, largest spatial scale (with positive freq)). 
        self.xfreq = fftpack.fftfreq(self.nx, 1.0)
        self.yfreq = fftpack.fftfreq(self.ny, 1.0)
        self.xfreqscale = self.xfreq[1] - self.xfreq[0]
        self.yfreqscale = self.yfreq[1] - self.yfreq[0]
        return

    def calcPsd2d(self):
        """Calculate the 2d power and phase spectrum of the image.
        If 'shift', shifts small frequencies to the edges
        (just carries through for PSD, but changes calculation of phase)."""
        try:
            self.fimage
        except AttributeError:
            self.calcFft()
        # Calculate 2d power spectrum.
        # psd = <| R(u,v)^2 + I(u,v)^2| >
        self.psd2d = numpy.absolute(self.fimage)**2.0
        # phase spectrum
        # phase = arctan(I(u,v) / R(u,v))
        if self.shift:
            self.phasespec = numpy.arctan2(fftpack.ifftshift(self.fimage).imag,
                                           fftpack.ifftshift(self.fimage).real)
        else:
            self.phasespec = numpy.arctan2(self.fimage.imag, self.fimage.real)
        return

    def calcPsd1d(self, min_npix=3., min_dr=1.0):
        """Calculate the 1-D power spectrum. The 'tricky' part here is determining spatial scaling.
        At least npix pixels will be included in each radial bin, in frequency space, and the minimum
        spacing between bins will be minthresh pixels. This means the 'central' bins (large pixel scales)
        will have larger steps in the 1dPSD. """
        # Calculate 1d power spectrum                
        #  - uses shifted PSD so that can create radial bins from center, with largest scales at center.
        # Calculate all the radius values for all pixels. These are still in frequency space. 
        rvals = numpy.hypot((self.yy-self.ycen), (self.xx-self.xcen)) + 0.5
        # Sort the PSD2d by the radius values and make flattened representations of these.
        idx = numpy.argsort(rvals.flatten())
        if self.shift:
            dvals = self.psd2d.flatten()[idx]
        else:
            dvals = (fftpack.fftshift(self.psd2d)).flatten()[idx]
        rvals = rvals.flatten()[idx]
        # Set up bins uniform in min_dr pix per bin, but don't want to subdivide center with too
        #  few pixels, so rebin if needed if fall below min_npix. 
        self.min_dr = min_dr
        self.min_npix = min_npix
        rbins = numpy.arange(0, rvals.max()+min_dr*2., min_dr)
        pixperbin = 2.0*numpy.pi*(rbins[1:]**2 - rbins[:-1]**2)
        update = [rbins[0],]
        r0 = rbins[0]
        while r0 < rbins[numpy.where(pixperbin>min_npix)].min():
            r1 = min_npix / 2.0 / numpy.pi + r0**2
            update.append(r1)        
            r0 = r1
        update = numpy.array(update, 'float')
        rbins = numpy.concatenate((update, rbins[numpy.where(rbins>update.max())]))
        # Calculate how many data points are actually present in each radius bin (for weighting)
        nvals = numpy.histogram(rvals, bins=rbins)[0]
        # Calculate the value of the image in each radial bin (weighted by the # of pixels in each bin)
        rprof = numpy.histogram(rvals, bins=rbins, weights=dvals)[0]
        rprof = numpy.where(nvals>0, rprof/nvals, numpy.nan)
        # Calculate the central radius values used in the histograms (note this is still in frequency). 
        rcenters =  (rbins[1:] + rbins[:-1])/2.0
        # And calculate the relevant frequencies (using self.xfreq/yfreq & the corresponding pixel values).
        self.rfreq = rcenters * self.xfreqscale
        # Set the value of the 1d psd, and interpolate over any nans (where there were no pixels).
        self.psd1d = numpy.interp(rcenters, rcenters[rprof==rprof], rprof[rprof==rprof])
        # Scale rcenters to 'original pixels' scale (ie. in the FFT space, pixels are scaled x_fft = 1/(x_pix*2pi)
        #   but must also account for overall size of image 
        self.psdx = 1/(rcenters*2.0*numpy.pi) * numpy.sqrt(self.nx*self.ny)
        return

    def calcAcf2d(self):
        """Calculate the 2d auto correlation function. """
        # See Wiener-Kinchine theorem
        if self.shift:
            # Note, the ACF needs the unshifted 2d PSD for inverse FFT, so unshift.
            #  Then shift back again. 
            self.acf = fftpack.fftshift(fftpack.ifft2(fftpack.ifftshift(self.psd2d)))
        else:
            self.acf = fftpack.ifft2(self.psd2d)
        return

    def calcAcf1d(self, min_npix=3, min_dr=1.0):
        """Calculate the 1d average of the ACF. This is probably more intuitive than the PSD."""
        # Calculate all the radius values for all pixels. These are actually in 'pixel' space
        #    (as ACF is FFT of PSD). 
        rvals = numpy.hypot((self.xx-self.xcen), (self.yy-self.ycen)) + 0.5
        # Sort the ACF2d by the radius values and make flattened representations of these.
        idx = numpy.argsort(rvals.flatten())
        if self.shift:
            dvals = self.acf.flatten()[idx].real
        else:
            dvals = (fftpack.fftshift(self.acf)).flatten()[idx].real
        rvals = rvals.flatten()[idx]
        # Set up bins uniform in npix per bin, for the 1d ACF calculation (like with PSD case). 
        #  but want to subdivide outer parts of image too much, so use a minimum of min_dr pix
        self.min_dr = min_dr
        self.min_npix = min_npix
        rbins = numpy.arange(0, rvals.max()+min_dr, min_dr)
        pixperbin = 2.0*numpy.pi*(rbins[1:]**2 - rbins[:-1]**2)
        update = [rbins[0],]
        r0 = rbins[0]
        while r0 < rbins[numpy.where(pixperbin>min_npix)].min():
            r1 = npix / 2.0 / numpy.pi + r0**2
            update.append(r1)        
            r0 = r1
        update = numpy.array(update, 'float')
        rbins = numpy.concatenate((update, rbins[numpy.where(rbins>update.max())]))
        # Calculate how many data points are actually present in each radius bin (for weighting)
        nvals = numpy.histogram(rvals, bins=rbins)[0]
        # Calculate the value of the image in each radial bin (weighted by the # of pixels in each bin)
        rprof = numpy.histogram(rvals, bins=rbins, weights=dvals)[0] / nvals
        # Calculate the central radius values used in the histograms
        rcenters =  (rbins[1:] + rbins[:-1])/2.0
        # Set the value of the 1d ACF, and interpolate over any nans (where there were no pixels)
        self.acf1d = numpy.interp(rcenters, rcenters[rprof==rprof], rprof[rprof==rprof])
        self.acfx = rcenters
        return

    def calcSf(self):
        """Calculate the structure function from the 1d ACF, discounting zeroPadding region. Result is scaled 0-1."""
        self.sfx = numpy.arange(0, (numpy.sqrt((self.nx/2.0 - self.padx)**2 + (self.ny/2.0 - self.pady)**2)), 1.0)
        self.sf = numpy.interp(self.sfx, self.acfx, (1.0 - self.acf1d))        
        # SF should be > 0, so scale .. might as well scale to be between 0 and 1.
        self.sf = (self.sf - self.sf.min()) / (self.sf.max() - self.sf.min())
        return

    def calcAll(self, min_npix=3, min_dr=3.):
        self.calcFft()
        self.calcPsd2d()
        self.calcPsd1d(min_npix=min_npix, min_dr=min_dr)
        self.calcAcf2d()
        self.calcAcf1d(min_npix=min_npix, min_dr=min_dr)
        self.calcSf()
        return

    def _makeRandomPhases(self):
        # Generate random phases (uniform -360 to 360)
        self.phasespecI = numpy.random.uniform(low=-numpy.pi, high=numpy.pi, size=[self.ny, self.nx])
        # Generate random phases with gaussian distribution around 0
        #self.phasespecI = numpy.random.normal(loc=0, scale=(numpy.pi*2.0 / 2.0), size=[self.ny, self.nx])
        # Wrap into -180 to 180 range
        self.phasespecI = (self.phasespecI-self.phasespecI.min()) % (numpy.pi*2.0) - (numpy.pi)
        return

    def invertFft(self, useI=False):
        """Convert the 2d FFT into an image (imageI)."""
        # Checking this process with a simple (non-noisy) image shows that it will result in errors on the
        #  level of 1e-15 counts (in an original image with min/max scale of 1.0).
        if useI:
            fimage = self.fimageI
        else:
            fimage = self.fimage
        if self.shift:
            self.imageI = fftpack.ifft2(fftpack.ifftshift(fimage))
        else:
            self.imageI = fftpack.ifft2(fimage)
        if self.imageI.imag.max() < 1e-14:
            print "Inverse FFT created only small imaginary portion - discarding."
            self.imageI = self.imageI.real
        return

    def invertPsd2d(self, useI=False):
        """Convert the 2d PSD and phase spec into an FFT image (FftI). """
        # The PHASEs of the FFT are encoded in the phasespec ('where things are')
        # The AMPLITUDE of the FFT is encoded in the PSD ('how bright things are' .. also radial scale)
        # amp = sqrt(| R(uv)^2 + I(u,v)^2|) == length of 'z' (z = x + iy, in fourier image)
        # phase = arctan(y/x)
        if useI:
            psd2d = self.psd2dI
            phasespec = self.phasespecI
        else:
            psd2d = self.psd2d
            phasespec = self.phasespec
        if self.shift:
            amp = numpy.sqrt(fftpack.ifftshift(psd2d))
        else:
            amp = numpy.sqrt(psd2d)
        # Shift doesn't matter for phases, because 'unshifted' it above, before calculating phase.
        x = numpy.cos(phasespec) * amp
        y = numpy.sin(phasespec) * amp
        self.fimageI = x + 1j*y
        if self.shift:
            self.fimageI = fftpack.fftshift(self.fimageI)
        return
    
    def invertPsd1d(self, amp1d=None, phasespec=None):
        """Convert a 1d PSD, generate a phase spectrum (or use user-supplied values) into a 2d PSD (psd2dI)."""
        # Converting the 2d PSD into a 1d PSD is a lossy process, and then there is additional randomness
        #  added when the phase spectrum is not the same as the original phase spectrum, so this may or may not
        #  look that much like the original image (unless you keep the phases, doesn't look like image). 
        # 'Swing' the 1d PSD across the whole fov.         
        if amp1d == None:
            amp1d = self.psd1d
            xr = self.rfreq / self.xfreqscale
        else:
            xr = numpy.arange(0, len(amp1d), 1.0)
        # Resample into even bins (definitely necessarily if using psd1d).
        xrange = numpy.arange(0, max(self.xcen, self.ycen), 1.0)
        amp1d = numpy.interp(xrange, xr, amp1d)
        # Calculate radii - distance from center.
        rad = numpy.hypot((self.yy-self.ycen), (self.xx-self.xcen))
        # Calculate the PSD2D from the 1d value.
        self.psd2dI = numpy.interp(rad.flatten(), xrange, amp1d)
        self.psd2dI = self.psd2dI.reshape(self.ny, self.nx)
        if phasespec == None:
            self._makeRandomPhases()
        else:
            self.phasespecI = phasespec
        if not(self.shift):
            # The 1d PSD is centered, so the 2d PSD will be 'shifted' here. 
            self.psd2dI = fftpack.ifftshift(self.psd2dI)
        return

    def invertAcf2d(self, useI=False):
        """Convert the 2d ACF into a 2d PSD (psd2dI). """
        if useI:
            acf = self.acfI
        else:
            acf = self.acf
            self.phasespecI = self.phasespec
        # Calculate the 2dPSD from the ACF. 
        if self.shift:
            self.psd2dI = fftpack.ifftshift(fftpack.fft2(fftpack.fftshift(acf)))
        else:
            self.psd2dI = fftpack.fft2(acf)
        # PSD2d should be entirely real and positive (PSD2d = |R(uv,)**2 + I(u,v)**2|
        #print 'PSD real limits', self.psd2dI.real.min(), self.psd2dI.real.max()
        #print 'PSD imaginary limits', self.psd2dI.imag.min(), self.psd2dI.imag.max()
        # Okay, I admit - this next line is a bit of a hack, but it does seem to work. 
        self.psd2dI = numpy.sqrt(numpy.abs(self.psd2dI)**2)
        return

    def invertAcf1d(self, amp1d=None, phasespec=None):
        """Convert a 1d ACF into a 2d ACF (acfI). """
        # 'Swing' the 1d ACF across the whole fov.         
        if amp1d == None:
            amp1d = self.acf1d
            xr = self.acfx
        else:
            xr = numpy.arange(0, len(amp1d), 1.0)
        # Resample into even bins (definitely necessarily if using acf1d).
        xrange = numpy.arange(0, max(self.xcen, self.ycen), 1.0)
        amp1d = numpy.interp(xrange, xr, amp1d)
        # Calculate radii - distance from center.
        rad = numpy.hypot((self.yy-self.ycen), (self.xx-self.xcen))
        # Calculate the PSD2D from the 1d value.
        self.acfI = numpy.interp(rad.flatten(), xrange, amp1d)
        self.acfI = self.acfI.reshape(self.ny, self.nx)
        if phasespec == None:
            self._makeRandomPhases()
        else:
            self.phasespecI = phasespec
        if not(self.shift):
            self.acfI = fftpack.fftshift(self.acfI)
        return

    
