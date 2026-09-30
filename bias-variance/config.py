import numpy as np
import plotnine as gg

gg.theme_set(gg.theme_bw());

CONFIG = {
    "num_data" : 500,
    "noise_sigma" : 0.1,
    "lamda_cands" : np.arange(0,3, 0.05),
    "num_features" : [5,10,15,20,25],
    "feature_cands": ["polynomial", "fourier", "gaussian", "identical"],

}
