import numpy as np

def yield_err( n_eff, isSignal):
    lumiscale, delta_scale = (0.0618, 0.00027) if isSignal else (552.33, 2.875)
    n_real = n_eff / lumiscale
    delta_n_real = np.sqrt(n_real)
    delta_n_eff = n_eff  * np.sqrt( (delta_n_real/n_real)**2 + (delta_scale/lumiscale)**2 ) 
    print("lumiscale = {} +/- {}, n_real = {} +/- {}, total_yield = {} +/- {}".format(lumiscale, delta_scale, n_real, delta_n_real, n_eff, delta_n_eff))

    return delta_n_eff
