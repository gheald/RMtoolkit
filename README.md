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
>>> p = rm.PolObservation(freq_hz,(idata,qdata,udata),IQUerr=(ierrors,qerrors,uerrors))
>>> phi_axis = np.arange(-2000.,2000.1,10.)
>>> p.rmsynthesis(phi_axis)
>>> p.rmclean()
>>> p.plot_fdf()
```

