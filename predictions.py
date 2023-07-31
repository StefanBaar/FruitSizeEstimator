from glob import glob
from tqdm import tqdm

import datetime

import numpy as np
import pandas as pd
from scipy import ndimage, spatial, optimize, signal, stats

from lmfit import Model

from skimage import io, morphology, exposure
from skvideo import io as vio
import cv2

from joblib import Parallel, delayed
import logistic_models as LM


class models:

    def __init__(self):
        self.names = [model for model in dir(self)
                      if (model.startswith('__') is False) and (model.endswith('const') is False)]

    def trf1( x, x0, A, b, d, e):
        return A/(1.+b*np.exp(-d*(x-x0)))**(1/e)

    def trf2( x, x0, A, b, d):
        return A/(1.+b*np.exp(-d*(x-x0)))**(1/b)

    def trf3( x, x0, A, b, d):
        return A/(1.+b*np.exp(-d*(x-x0)))

    def gomperz( x, x0, A, b, d):
        return A*np.exp(np.log(b/A)*np.exp(-d*(x-x0)))

    # inverted functions

    def itrf1( x, x0, A, b, d, e):
        return A-A/(1.+b*np.exp(-d*(x-x0)))**(1/e)

    def itrf2(x, x0, A, b, d):
        return A-A/(1.+b*np.exp(-d*(x-x0)))**(1/b)

    def itrf3(x, x0, A, b, d):
        return A/(1.+b*np.exp(-d*(x-x0)))

    def igomperz(x, x0, A, b, d):
        return A-A*np.exp(np.log(b/A)*np.exp(-d*(x0-x)))

    # constant inverted functions

    def trf2_const(x, x0, A):
        return A/(1.+5.*np.exp(-0.5*(x-x0)))**(1/5.)

    def itrf2_const( x, x0, A):
        return A-A/(1.+5.0*np.exp(-0.5*(x0-x)))**(1./5.)
    



class evaluate_cascade:
    def __init__(self, ff, X, Y, p,q):

        self.pstd = [np.sqrt(np.diag(i)) for i in q]
        
        self.YF = np.asarray([ff(np.asarray(X), *np.asarray(i)) for i in p])
        self.L1 = np.abs(Y-self.YF)
        self.L1stat = self.metrics(self.L1)
        self.L2 = self.L1**2
        self.L2stat = self.metrics(self.L2)
        
        self.LG = np.sqrt(np.abs(Y**2-self.YF**2))

    def metrics(self, data):
        return np.array([np.sqrt(np.sum(data, 1)),
                         np.mean(data, 1),
                         np.median(data, 1),
                         np.std(data, 1),
                         np.var(data, 1),
                         np.sqrt(np.sum(data, 1))])

class prediction_cascade:
    def __init__(self, DATA, ff=None):

        X = DATA[1].astype(float)
        Y = DATA[2].astype(float)
       
        self.dX = X[-1] - X[2]
        self.HV = Y[-1]
        
        self.X = [X]+[X[:-i] for i in range(1,len(X)-3)]
        self.Y = [Y]+[Y[:-i] for i in range(1,len(Y)-3)]

        self.days_to_harvest  = np.asarray([self.X[0][-1]-i[-1] for i in self.X])
        self.abs_maturity     = np.asarray([i[-1]-self.X[0][0] for i in self.X])
        self.rel_maturity     = np.asarray([(i[-1]-self.X[0][0])/(self.X[0][-1]-self.X[0][0]) for i in self.X])

        self.p0 = (self.X[0][3], 35)
        self.bounds = ([0, 10], [400, 40])

        if ff is None:
            self.ff = models.itrf2_const

        self.p, self.q = self.predict()
        
        self.stats = evaluate_cascade(self.ff, X, Y, self.p, self.q)
    

    def predict(self):
        P,Q = [],[]
        for i in range(len(self.X)):
            p,q = optimize.curve_fit(self.ff, self.X[i], self.Y[i],
                                     p0=self.p0,
                                     bounds=self.bounds,
                                     maxfev=5000000,
                                     )
            P.append(p)
            Q.append(q)
        return P,Q
    

        

class prediction_model:
    def __init__(self, DATA, ff=None):
        
        self.X = DATA[1].astype(float)
        self.Y = DATA[2].astype(float)
        
        self.dX = self.X[-1] - self.X[0]

        

        self.p0     = (self.X[3],35)
        self.bounds = ([0, 10], [400, 40])
        
        self.b = 5.0
        self.d = 0.5
        
        if ff is None:
            self.ff = models.itrf2_const

        self.p, self.q = self.predict()

        self.evaluate()        

    def evaluate(self):
        def metrics(data):
            return np.array([np.mean(data),
                             np.median(data),
                             np.std(data),
                             np.var(data)])
        
        self.pstd   = np.sqrt(np.diag(self.q))
        self.YF     = self.ff(np.asarray(self.X),*np.asarray(self.p))
        self.L1     = np.abs(self.Y-self.YF)
        self.L1stat = metrics(self.L1)
        self.L2     = self.L1**2
        self.L2stat = metrics(self.L2)
        
        self.LG     = np.sqrt(np.abs(self.Y**2-self.YF**2))
        
    def predict(self):
        return optimize.curve_fit(self.ff, self.X, self.Y,
                                  p0    = self.p0,
                                  bounds= self.bounds,
                                  maxfev=5000000,
                                  )

def lmfit_fit(ff, Days, Diameter,AMIN=15):

    fmodel = Model(ff)
    params = fmodel.make_params(x0=Days[3], A=35)  # , b=5., d=0.5,)
    
    params["x0"].min = 0
    params["A"].min  = 10
    params["x0"].max = 400
    params["A"].max  = 50
    #params['b'].vary = False
    #params['d'].vary = False
    
    result = fmodel.fit(Diameter, params, x=Days, max_nfev=5000000)

    return list(result.values.values())


def lmfit_2_fit(ff, Days, Diameter, AMIN=10):

    fmodel = Model(ff)
    params = fmodel.make_params(x0=Days[3], A=35,b=5.,d=0.5)  # , b=5., d=0.5,)

    params["x0"].initial_value = Days[3]
    params["A"].initial_value = 35
    params["x0"].min = 0
    params["A"].min = 10
    params["x0"].max = 400
    params["A"].max = 50
    params['b'].vary = False
    params['d'].vary = False

    result = fmodel.fit(Diameter, params, x=Days, max_nfev=5000000)
    return list(result.values.values())
