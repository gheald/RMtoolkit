# RMtoolkit
Simple 1-D RM Synthesis &amp; RMCLEAN toolkit

This provides simple functionality to perform 1-dimensional
RM Synthesis and RMCLEAN.

## References

- RM Synthesis: [Brentjens & de Bruyn (2005)](http://adsabs.harvard.edu/abs/2005A%26A...441.1217B)
- RMCLEAN: [Heald, Braun & Edmonds (2009)](http://adsabs.harvard.edu/abs/2009A%26A...503..409H)

## How to use it

Obtain the `rm.py` script, and use it like this (as a simple example):

```
>>> import rm
>>> import numpy as np
>>> freq_hz,idata,ierr,qdata,qerr,udata,uerr = np.loadtxt('mockdata.txt',unpack=True)
>>> p = rm.PolObservation(freq_hz,(idata,qdata,udata),IQUerr=(ierr,qerr,uerr))
Fitted spectral index is -0.719883

>>> phi_axis = np.arange(-2000.,2000.1,10.)
>>> p.rmsynthesis(phi_axis)
Using 300 of 300 channels
Using uniform weights
Normalisation value = 0.003333
FWHM of RMSF is 192.128413 rad/m2
Max RM scale is 89.368310 rad/m2
Max RM value is -21193.286828 rad/m2
Calculating RMSF...
Calculating FDF...

>>> p.rmclean(cutoff=1.)
CLEAN will proceed down to 0.213212
First component found at 130.000000 rad/m2
Iteration 1: max residual = 1.225228
Iteration 2: max residual = 1.102705
Iteration 3: max residual = 0.992814
Iteration 4: max residual = 0.893690
Iteration 5: max residual = 0.804646
Iteration 6: max residual = 0.724291
Iteration 7: max residual = 0.652145
Iteration 8: max residual = 0.587001
Iteration 9: max residual = 0.528548
Iteration 10: max residual = 0.475733
Convolving clean components...
Restoring convolved clean components...

>>> p.plot_fdf()
```

This example contained within `mockdata.txt` contains two sources,
- A 60% polarized source with I=2Jy and alpha=-0.7, and RM=135rad/m2; and
- A 33% polarized source with I=700mJy and alpha=-0.7, and RM=-625rad/m2.

Both sources are in the same IQU spectrum, observed from 1300-1600 MHz with 1 MHz channels. The noise in I is ~20 mJy, and in Q and U is ~10 mJy.

