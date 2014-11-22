# -*- coding: utf-8 -*-

from numba import vectorize
from timeit import repeat

import numpy as np
from numba import jit, void, double

from multi_thread_vectorizer import mvectorise

if __name__=='__main__':
    """ Example Usage """
    
    def looped_floor_closest_valid_odds( x ):
    
        r = np.zeros_like( x )
    
        for i in range( len( r ) ):
            xi = x[i]
            if xi<=1.0:
                r[i] = np.nan
            elif xi<=2.0:            
                r[i] = 0.01 * np.floor( xi / 0.01 )            
            elif xi<=3.0:                        
                r[i] = 0.02 * np.floor( xi / 0.02 )            
            elif xi<=4.0:            
                r[i] = 0.05 * np.floor( xi / 0.05 )
            elif xi<=6.0:            
                r[i] = 0.1 * np.floor( xi / 0.1 )
            elif xi<=10.0:            
                r[i] = 0.5 * np.floor( xi / 0.5 )
            elif xi<=20.0:            
                r[i] = 1.0 * np.floor( xi / 1.0 )
            elif xi<=30.0:            
                r[i] = 2.0 * np.floor( xi / 2.0 )
            elif xi<=50.0:            
                r[i] = 2.0 * np.floor( xi / 2.0 )
            elif xi<=100.0:            
                r[i] = 5.0 * np.floor( xi / 5.0 )
            elif xi<=1000.0:            
                r[i] = 10.0 * np.floor( xi / 10.0 )
            else:            
                r[i] = 1000.0
        return r
    
    """
    This was the original fastest function
    Cannot use jit( ..., nopython=True ) as np.zeros_like is currently not compatible in no python mode
    """
    signature = double[:](double[:])
    lf        = jit( signature )( looped_floor_closest_valid_odds )
    
    def floor_closest_valid_odds( xi ):

        if xi<=1.0:
            return 1.0
        elif xi<=2.0:            
            return 0.01 * np.floor( xi / 0.01 )            
        elif xi<=3.0:                        
            return 0.02 * np.floor( xi / 0.02 )            
        elif xi<=4.0:            
            return 0.05 * np.floor( xi / 0.05 )
        elif xi<=6.0:            
            return 0.1 * np.floor( xi / 0.1 )
        elif xi<=10.0:            
            return 0.5 * np.floor( xi / 0.5 )
        elif xi<=20.0:            
            return 1.0 * np.floor( xi / 1.0 )
        elif xi<=30.0:            
            return 2.0 * np.floor( xi / 2.0 )
        elif xi<=50.0:            
            return 2.0 * np.floor( xi / 2.0 )
        elif xi<=100.0:            
            return 5.0 * np.floor( xi / 5.0 )
        elif xi<=1000.0:            
            return 10.0 * np.floor( xi / 10.0 )
        else:            
            return 1000.0
        return 0.0

    signature = double(double,)
    nb_floor_closest_valid_odds = jit(signature, nopython=True)(floor_closest_valid_odds)
    
    mf = mvectorise( nb_floor_closest_valid_odds, double[:], double[:] )
    
    signature = double[:](double[:],)
    vf = vectorize(['float64(float64)'], nopython=True)(floor_closest_valid_odds)
    
    uf = np.vectorize( floor_closest_valid_odds )
        
    def timefunc(correct, s, func, *args, **kwargs):
        print(s.ljust(20), end=" ")
        # Make sure the function is compiled before we start the benchmark
        res = func(*args, **kwargs)
        if correct is not None:
            assert np.allclose(res, correct)
        # time it
        print('{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs),
                                              number=5, repeat=2)) * 1000))
        return res
    
    x = np.random.uniform( 1.0, 1000.0, 1e2)
    
    correct = vf( x )
    
    timefunc(correct, "numba (looped)", lf,x)
    timefunc(correct, "numba (vectorised)", vf,x)
    timefunc(correct, "numba (multi-threaded vectorised)", mf,x)
    
    import timeit
    ls = np.logspace(1,6,10)
    
    mf_results = []
    vf_results = []
    uf_results = []
    lf_results = []
    
    for i, xsize in enumerate( ls ):
        print( 'Generating stats for xsize %s, index %i'%( xsize, i ) )
        x = np.random.uniform( 1.0, 1000.0, int(xsize))
        
        lf_results.append( timeit.timeit( 'lf(x)', "from __main__ import lf,x", number=3) )
        #uf_results.append( timeit.timeit( 'uf(x)', "from __main__ import uf,x", number=3) )
        mf_results.append( timeit.timeit( 'mf(x)', "from __main__ import mf,x", number=3) )
        vf_results.append( timeit.timeit( 'vf(x)', "from __main__ import vf,x", number=3) )
        
    import pandas as pd
    df = pd.DataFrame( {'multi-threaded':mf_results, 'looped':lf_results, 'vectorized':vf_results }, index=ls )
    import matplotlib.pyplot as plt
    df.plot()
    plt.show()

