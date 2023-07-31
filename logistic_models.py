from glob import glob
from tqdm import tqdm

from inspect import signature

import datetime

import numpy as np
import pandas as pd
from scipy import ndimage, spatial, optimize, signal, stats

from skimage import io,morphology, exposure
from skvideo import io as vio
import cv2

from joblib import Parallel, delayed

#### functions
class models:

    def __init__(self):
        self.names = [model for model in dir(self)
                      if (model.startswith('__') is False) and (model.endswith('const') is False)]

    def trf1(self, x, x0, A, b, d, e):
        return A/(1.+b*np.exp(-d*(x-x0)))**(1/e)

    def trf2(self, x, x0, A, b, d):
        return A/(1.+b*np.exp(-d*(x-x0)))**(1/b)

    def trf3(self, x, x0, A, b, d):
        return A/(1.+b*np.exp(-d*(x-x0)))

    def gomperz(self, x, x0, A, b, d):
        return A*np.exp(np.log(b/A)*np.exp(-d*(x-x0)))

    ### inverted functions

    def itrf1(self, x, x0, A, b, d, e):
        return A-A/(1.+b*np.exp(d*(x-x0)))**(1/e)

    def itrf2(self, x, x0, A, b, d):
        return A-A/(1.+b*np.exp(d*(x-x0)))**(1/b)

    def itrf3(self, x, x0, A, b, d):
        return A/(1.+b*np.exp(-d*(x-x0)))

    def igomperz(self, x, x0, A, b, d):
        return A-A*np.exp(np.log(b/A)*np.exp(-d*(x0-x)))

    ### constant inverted functions
    def itrf2_const(self, x, x0, A):
        return A-A/(1.+5.0*np.exp(-0.5*(x0-x)))**(1./5.)





def fit(ff, X, Y, p0=None,bounds=None):

    smask = np.ones_like(X).astype(bool)
    return optimize.curve_fit(ff, X[smask].astype(float), Y[smask].astype(float),
                                # sigma = Z[smask],#np.ones_like(Z)*np.max(Z),
                                # bounds= (0.00000000000000000001,np.inf),
                                # bounds=([10,0.001,0,0],[np.inf,np.inf,np.inf,np.inf]),
                                bounds= bounds,
                                p0    = p0,
                                maxfev= 1000000000,
                                )

def get_fits(X, Y, ff, res=200, p1=None, XMAX=None, bounds=None):

    pn = len(signature(ff).parameters)-1

    if p1 == None:
        #p0 = [X[3], Y[-1], 10, 10, 10][:pn]
        #print(X[3])
        p0 = [12, 30, 0.1, 0.1, 0.1][:pn]
    else:
        p0 = p1

    if bounds==None:

       #b0 = ([0, 1, 1, 0.1, 1][:pn], [100, 100, 100, 100, 100][:pn])
       b0 =  ([0., 0, 0, 0,0][:pn], [30, 100, 50, 10,10][:pn])
       #b0 = ([0, 10,0,0,0][:pn], [20, 100, 10, 10, 1][:pn])
       #if len(p0) == 5:
       #b0 = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
       #      [np.inf, np.inf, np.inf, np.inf, np.inf])
    else:
       b0 = bounds

    if len(p0) == 5:
        p0 = None
    p, q = fit(ff, X, Y, p0, b0)

    if XMAX == None:
        XMAX = X[-1]+1

    XF = np.linspace(X[0], XMAX, res)
    YF = ff(XF, *p)

    return XF, YF, p, q


def get_fit_error(X,Y,ff,P):
    YF = ff(np.asarray(X,dtype="float"),*list(P))
    L1 = np.abs(Y   -YF)
    L2 = np.abs(Y**2-YF**2)
    return L1,L2


def sfit(func, X, Y, p0=None):
    return optimize.curve_fit(func, X, Y,
                              p0=p0,
                              maxfev=5000000,
                              )


def get_2_lim_fits(X, Y, res=200, ff=None, p1=None, XF=None):

    if ff == None:
        ff = models.trf2

    if p1 == None:
        p0 = None
    else:
        p0 = p1

    if XF == None:
        XF = np.linspace(X[0], 60, res)

    p, q = optimize.curve_fit(ff, X, Y,
                              # sigma = Z[smask],#np.ones_like(Z)*np.max(Z),
                              # bounds= (0.00000000000000000001,np.inf),
                              # bounds=([10,0.001,0,0],[np.inf,np.inf,np.inf,np.inf]),
                              bounds=bounds,
                              p0=p0,
                              maxfev=50000,
                              )

    YF = ff(XF, *p)

    return XF, YF, p, q
