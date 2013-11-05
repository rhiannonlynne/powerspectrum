#
# lambda - physical length scale of AR process
# lambda_s - physical sampling interval of AR process
# lambda_avg - physical length over which AR process is integrated
# x_max - max physical distance over which to calculate the ACF
#
ACF_ARMA <- function(lambda, lambda_s, lambda_avg, x_max) {
	#
	nma <- round(lambda_avg/lambda_s)
	ma_coeff <- seq(length=nma, from=1.0, by=0)
	lag_max <- round(x_max/lambda_s)
	#
	# calculate the AR a1 coeff from lambda and lambda_s
	#
	a1 <- exp(-lambda_s / lambda)
	#
	acf <- ARMAacf(ar=a1, ma=ma_coeff, lag.max=lag_max)
	sf <- sqrt(1 - acf)
	x <- seq(length = length(acf), from=0.0, by=lambda_s)
	sfinterp <- approxfun(x, sf, method="linear", 0, 1)
	#
	list(x=x, acf=acf, sf=sf, sffun=sfinterp)
}

#
# write acf at fixed lambda, lambda_avg, lambda_s as a function of 
# angular distance theta, in arcmin, at 10km.   The native x coordinate of the acf
# is the sampling distance lambda_s.
#
ACF_Write <- function(filename, lambda, lambda_avg, lambda_s, x_max) {
	#
	acf <- ACF_ARMA(lambda, lambda_s, lambda_avg, x_max)
	#
	for (i in seq(length(acf$x))) {
	    write(c(acf$x[i], acf$acf[i]), filename, append=TRUE)
	}

}

