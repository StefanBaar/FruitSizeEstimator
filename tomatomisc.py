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

import openpyxl

from joblib import Parallel, delayed

import mediapy
from moviepy.video.io.bindings import mplfig_to_npimage as fig_to_np

import xlsio

import logistic_models as LM

class EnvironmentSeries:
    def __init__(self, path):
        self.DF   = pd.read_csv(path)
        self.data = self.DF

        self.date = self.get_date()
        self.days = np.asarray([self.get_float_day(i) for i in self.date])

        self.temp = np.asarray(self.data["気温"])
        self.hum  = np.asarray(self.data["湿度"])
        self.bor  = np.asarray(self.data["飽差"])
        self.rad  = np.asarray(self.data["日射量"])
        self.co2  = np.asarray(self.data["CO2濃度.1"])

        self.day, self.inds = self.daily_mask(self.days)

        self.day_date = self.get_day_date()
        self.day_temp = self.daily_data(self.temp, self.inds)
        self.day_hum  = self.daily_data(self.hum , self.inds)
        self.day_bor  = self.daily_data(self.bor , self.inds)
        self.day_rad  = self.daily_data(self.rad , self.inds)
        self.day_co2  = self.daily_data(self.co2 , self.inds)

    def get_float_day(self,DATETIME):
        day = float(DATETIME.timetuple().tm_yday)
        day = day + (float(DATETIME.hour)+(float(DATETIME.minute)/60.))/24.
        return day

    def get_date(self):
        def to_date(STR):
            return datetime.datetime.strptime(STR, "%Y-%m-%d %H:%M:%S")
        return np.asarray([to_date(i) for i in self.data["登録日時"]])

    def get_day_date(self):
        return np.asarray([self.date[0] + datetime.timedelta(days=int(i-self.day[0])) for i in self.day])

    def daily_data(self, DATA, INDS,bins= 20):

        MEAN, MEDIAN, MODE, STD, VAR, MIN, MAX, HIST = [], [], [], [], [], [], [], []

        for i in INDS:

            pts  = np.asarray([DATA[j] for j in i])
            MEAN  .append(np.mean(     pts))
            MEDIAN.append(np.median(   pts))
            MODE  .append(stats.mode(  pts)[0])
            STD   .append(np.std(      pts))
            VAR   .append(np.var(      pts))
            MIN   .append(np.min(      pts))
            MAX   .append(np.max(      pts))
            HIST  .append(np.histogram(pts, bins=bins, density=True))

        return [np.asarray(MEAN),
                np.asarray(MEDIAN),
                np.asarray(MODE),
                np.asarray(STD),
                np.asarray(VAR),
                np.asarray(MIN),
                np.asarray(MAX),
                np.asarray(HIST)]

    def daily_mask(self,DAYS):
        """produce instruction to fold data"""
        intdays = np.round(DAYS,0).astype(int)

        unis = np.unique(intdays)
        inds = []
        for u in unis:
            ind = np.argwhere(intdays == u)[:, 0]
            inds.append(ind)
        return unis,inds

class TomatoData:
    def __init__(self, path, days=3, d0=3):
        self.xls_data  = openpyxl.load_workbook(path, data_only=True)
        self.data      = self.get_data()

        self.diameter  = self.propery_date_pair(5)
        self.diameter0 = [self.add_zero_days(i, days, d0) for i in self.diameter]
        self.color     = self.propery_date_pair(4)
        self.cluster   = self.get_cluster_number()


    def get_data(self):
        return xlsio.get_all_sheet_data(self.xls_data)

    def get_pair(self, PROPERTY, DATE):
        """   0        1       2       3       4       5    """
        """filename, tomato, DATE, position, color, diameter"""
        RES = []
        for i in PROPERTY:
            mask  = np.where(i != -1)
            DATES = DATE[mask]
            DAYS  = self.convert_to_day(DATES)
            RES.append(np.asarray([DATES, DAYS, i[mask]]))
        return RES

    def propery_date_pair(self,IND):
        DATA = self.data
        pair = self.get_pair(DATA[0][IND], DATA[0][2])
        for i in DATA[1:]:
            pair += self.get_pair(i[IND], i[2])
        return pair

    def convert_to_day(self,dates):
        return np.asarray([i.timetuple().tm_yday for i in dates])

    def get_cluster_number(self):
        """reading from xls is not working properly for files containing multiple tomato cluster
           only ..."""
        cluster_IDS = []
        for n,i in enumerate(self.data):
            for j in range(len(i[-1])):
                cluster_IDS.append(n)
        return np.asarray(cluster_IDS)


    def add_zero_days(self, data, days=3, d0=3):
        T0 = np.array([[data[0][0]-datetime.timedelta(days=d0+1*days), data[1][0]-d0-days, 0],
                    [data[0][0]-datetime.timedelta(days=d0+2*days), data[1][0]-d0-2*days, 0]])[::-1]  # type: ignore
        return np.vstack([T0, data.T]).T

class logistic_fit:
    def __init__(self, ff, input_data, xmax=60):
        """input_data: list of [datetime,value] pairs"""

        self.ff     = ff
        self.pnames = ["x0","proj. diameter","b","d","harvest dimater","xs","growth rate"]
        self.data   = input_data
        self.regression(ff, xmax=xmax) ### compute regression results
        self.get_stats()               ### compute regression statistics

    def dt_from_days(self,int_days,float_days):

        time_table = []
        for n, d in enumerate(int_days):
            day = datetime.datetime.strptime('2021 12 '+str(d), '%Y %H %j')
            days = np.asarray([day+datetime.timedelta(days=i) for i in float_days[n]])
            time_table.append(days)
        #print(time_table)
        return time_table

    def regression(self, ff, xmax=60):

        XE,YE,XEF,YEF,XTS = [],[],[],[],[]

        PE,QE  = [],[]
        L1,L2  = [],[]
        ME,MSE = [],[]

        for i in tqdm(self.data):

            xt,y0    = self.get_nod(i[1:])
            xt0      = xt[0]
            x0       = xt-xt0
            x,y,p,q  = LM.get_fits(x0,y0,ff,XMAX=xmax)
            #print(p)
            l1,l2    = LM.get_fit_error(x0,y0,ff,p)

            sn,slope = self.get_max_diff(x,y)

            XTS.append(xt0)
            XE.append(x0)
            YE.append(y0)
            XEF.append(x)
            YEF.append(y)
            p = list(p)
            p.append(y[-1])
            p.append(x[sn])
            p.append(slope)
            PE.append(list(p))
            QE.append(q)

            L1.append(l1)
            L2.append(l2)

            ME .append(np.mean(l1))
            MSE.append(np.mean(l2))

        self.XTS   = np.asarray(XTS)
        self.PET   = np.asarray(PE)
        self.QET   = np.asarray(QE)

        self.XE      = XE ## input days from flowering
        self.XEdates = self.dt_from_days(self.XTS,self.XE)
        self.YE      = YE ## input size from flowering

        self.XEF      = np.asarray(XEF)                         ## fitted days from flowering
        self.XEFdates = self.dt_from_days(self.XTS,self.XEF)    # produce datetime
        self.YEF      = np.asarray(YEF)                         ## fitted size from flowering

        self.SMASK = np.argsort(XTS)

        self.XTS_sorted =  self.XTS[self.SMASK]
        self.PET_sorted = (self.PET[self.SMASK]).T
        self.QET_sorted = (self.QET[self.SMASK]).T

        self.ME  = np.asarray(ME)
        self.MSE = np.asarray(MSE)

        #return np.asarray(XTS),PET,QET,XE,YE
        #return 0


    def filter_params(self,upper_b=50, upper_d=5):

        def filter_up(X,Y,val,ind):
            MASK = np.ones_like(X,dtype=bool)
            MASK[Y[ind]>val] = False
            return X[MASK],Y.T[MASK].T

        def med_filter(X,Y):
            NX = np.unique(X)
            NY = []
            SY = []
            MY = []
            for i in NX:
                medlist = []
                for n in range(len(Y)):
                    if i == X[n]:
                        medlist.append(Y[n])
                NY.append(np.median(medlist))
                MY.append(np.std(medlist))
                SY.append(np.mean(medlist))

            return np.asarray([NY,MY,SY])


        # NXTS, PETF = filter_up(self.XTS, self.PET.T, upper_b, 2)
        # NXTS, PETF = filter_up(NXTS    ,       PETF, upper_d, 3)
        # if self.PET.shape[1] == 7:
        #     NXTS, PETF = filter_up(NXTS,       PETF, 25, 4)

        # PETM, sig, PETN = np.rollaxis(np.asarray([med_filter(NXTS, i) for i in PETF]), 1)

        # return np.asarray([signal.medfilt(i, 5) for i in PETM])

        #else:

        return self.filter_nan(np.asarray(self.PET.T))

    def filter_nan(self,ARR):
        rem = np.unique(np.argwhere(np.isnan(ARR))[:, 1])[::-1]
        print("removing the folowing rows: ",rem)
        for i in rem:

            ARR = np.delete(ARR, i, 1)

        return ARR




    def get_stats(self):
        PET = self.filter_params()

        #print(PET.shape)
        #print(np.unique(np.argwhere(np.isnan(self.PET))[:, 0]))

        self.PETfuncs = ["median","mean","std.","var","min","max"]
        self.PETstats = np.asarray([np.median(PET, axis=1),
                                    np.mean(PET, axis=1),
                                    np.std(PET, axis=1),
                                    np.var(PET, axis=1),
                                    np.min(PET, axis=1),
                                    np.max(PET, axis=1),
                                     ]).T


        pnum = len(signature(self.ff).parameters)-1

        self.Y_mean   = self.ff(self.XEF[0], *self.PETstats[:pnum, 1])
        self.Y_median = self.ff(self.XEF[0], *self.PETstats[:pnum, 0])


    def get_nod(self,DATA):
        """Add 0 values to data"""
        # D0 = np.array([[DATA[0][0]-3, 0],[DATA[0][0]-6, 0],[DATA[0][0]-9, 0]])[::-1]
        D0 = np.array([[DATA[0][0]-6, 0], [DATA[0][0]-9, 0]])[::-1]
        # D0 = np.array([[DATA[0][0]-3,0]])
        return np.vstack([D0, DATA.T]).T

    def get_max_diff(self,X,Y):
        dX,dY = np.diff(X),np.diff(Y)
        slope = np.diff(Y)/np.diff(X)
        return np.argmax(slope),np.max(slope)


def same_day_mask(A, B):
    Amask = []
    for n, i in enumerate(A):
        if i in B:
            Amask.append(n)
    AA = A[Amask]
    Bmask = []
    for n, i in enumerate(B):
        if i in AA:
            Bmask.append(n)
    return Amask, Bmask


#def moving_sum
