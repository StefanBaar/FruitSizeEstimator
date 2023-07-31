from glob import glob
from tqdm import tqdm

import datetime

import numpy as np
import pandas as pd
from scipy import ndimage, spatial, optimize, signal, stats

from skimage import io,morphology, exposure
from skvideo import io as vio
import cv2

from joblib import Parallel, delayed

import mediapy
from moviepy.video.io.bindings import mplfig_to_npimage as fig_to_np

########################################
### Timing stuff

def get_delta(A):
    dt = np.diff(A)
    mt = np.median(dt[1:2])
    dt = np.hstack([mt,dt])
    return np.asarray([i.days for i in np.cumsum(dt)])

def convert_to_day(dates):
    return np.asarray([i.timetuple().tm_yday for i in dates])

#########
#Exel stuff

def get_xy_of_name(sheet,STR="計測日"):
    avg_coords = []
    for n,i in enumerate(sheet.rows):
        for m,j in enumerate(i):
            if j.value == STR:
                avg_coords.append([n,m])
    return avg_coords

def get_xy_of_dates(sheet,y_coord,x_ind=1):
    dates = []
    for i in range(x_ind,sheet.max_column):
        val = sheet.cell(row=y_coord+1, column=x_ind+i).value

        dates.append(val)
    xd    = range(len(dates))
    dates = np.vstack([xd,dates])
    dates = dates.T[np.argwhere(dates[1]!=None)][:,0]
    return dates

def get_data(sheet,y_coord):
    dates = get_xy_of_dates(sheet,y_coord)
    xframe= [dates[0,0]+2,dates[-1,0]+3]
    ystart= y_coord+2

    fname = None

    yx0 = sheet.cell(row=ystart, column=dates[0,0]+1).value
    if yx0 == None:

        fname = []
        for i in range(xframe[0],xframe[1]):
            val = sheet.cell(row=ystart, column=i).value
            fname.append(val)
        fname = np.asarray(fname)
        ystart+=1

    n   = ystart
    val = 1
    num = []
    while val != None:
        val = sheet.cell(row=n, column=dates[0,0]+1).value
        n   +=1
        if val != None:
            num.append(val)

    yframe = [ystart,n-1]
    ROW = []
    for j in range(yframe[0],yframe[1]):
        COL = []
        for i in range(xframe[0],xframe[1]):
            val = sheet.cell(row=j, column=i).value
            if val == None:
                val=-1
            COL.append(val)

        ROW.append(COL)
    return np.asarray(num),dates[:,1],np.asarray(ROW),fname

def format_pos(arr):
    narr = []
    for i in arr:
        temp = []
        for j in i:
            a = j
            if "," in a:
                a = a.split(",")
                b = [int(a[0]),int(a[1])]
            elif "." in a:
                a = a.split(".")
                b = [int(a[0]),int(a[1])]
            else:
                b = [int(a),int(a)]
            temp.append(b)
        narr.append(temp)
    return np.asarray(narr)

def get_all_data(sheet):
    n_list = get_xy_of_name(sheet)

    _,_,color   ,_  = get_data(sheet,n_list[0][0])
    _,_,diameter,_  = get_data(sheet,n_list[1][0])
    tomato,DATE,position,fname = get_data(sheet,n_list[2][0])

    return [fname,tomato,DATE,format_pos(position),diameter,color]

def get_all_sheet_data(sheets):
    DATAS = []
    for i in tqdm(sheets.sheetnames[:-1]):
        sheet    = sheets.get_sheet_by_name(i)
        DATAS.append(get_all_data(sheet))
    return DATAS

def color_date_pair(COLOR,DATE):
    RES = []
    for i in COLOR:
        mask = np.where(i!=-1)
        DAYS = convert_to_day(DATE[mask])
        RES.append(np.asarray([DAYS,i[mask]]))
    return RES



########################################
### Imaging stuff

def find_images(names, LIST):
    IMAGES = []
    for i in names:
        IMAGE = []
        for j in LIST:
            if str(i)+".JPG" in j:
                IMAGE.append(j)
        IMAGES.append(IMAGE[0])
    return IMAGES


def get_crop_stack(ims,pos,size=75):
    CROP=[]
    INDS=[]
    for n,i in enumerate(pos):
        x,y = i
        if x != -1:
            CROP.append(ims[n,y-size:y+size,x-size:x+size])
            INDS.append(n)


    return np.asarray(CROP),np.asarray(INDS)


#### modeling stuff

def get_nod(DATA):
    #D0 = np.array([[DATA[0][0]-3, 0],[DATA[0][0]-6, 0],[DATA[0][0]-9, 0]])[::-1]
    D0 = np.array([[DATA[0][0]-6, 0],[DATA[0][0]-9, 0]])[::-1]
    #D0 = np.array([[DATA[0][0]-3,0]])
    return np.vstack([D0,DATA.T]).T


def trf2(x,x0,A,t2,t3):
    return A/(1.+t2*np.exp(-t3*(x-x0)))

def trf1(x,x0,A,t2,t3):
    return A/(1.+t2*np.exp(-t3*(x-x0)))**(1/t2)

def trf(x,x0,A,t2,t3,t4):
    return A/(1.+t2*np.exp(-t3*(x-x0)))**(1/t4)

def gomperz(x,x0,A,N,a):
    return A*np.exp(np.log(N/A)*np.exp(-a*(x-x0)))

#### inverted functions

def itrf1(x,x0,A,t2,t3):
    return A-A/(1.+t2*np.exp(-t3*(x0-x)))**(1/t2)

def itrf12(x, x0, A):
    return A-A/(1.+5.0*np.exp(-0.5*(x0-x)))**(1./5.)


def fit(func,X,Y,p0=None):

    smask  = np.ones_like(X).astype(bool)
    #smask[3] = False
    #smask[-1] = False
    bounds = ([0,0,0,0],[np.inf,np.inf,100,10])
    return optimize.curve_fit(func,X[smask],Y[smask],
                              #sigma = Z[smask],#np.ones_like(Z)*np.max(Z),
                              #bounds= (0.00000000000000000001,np.inf),
                              #bounds=([10,0.001,0,0],[np.inf,np.inf,np.inf,np.inf]),
                              bounds = bounds,
                              p0=p0,
                              maxfev=5000,
                              )


def get_fits(X, Y, res=200, ff=None, p1=None, XMAX=None):

    if p1 == None:
        #p0 = [X[3],Y[-1], 7]
        p0 = [X[3],Y[-1], 7, 0.7,1]
    else:
        p0 = p1
    #p0    = None
    if ff==None:
        ff = itrf1

    else:
        p0 = p0[:-1]
    p ,q  = fit(ff,X,Y,p0)

    if XMAX == None:
        XMAX = X[-1]+1

    #if XF == None:
    #   XF    = np.linspace(X[0],X[-1]+1,res)
    XF = np.linspace(X[0], XMAX, res)
    YF    = ff(XF,*p)

    return XF,YF,p,q


def get_sfits(X, Y, res=200, ff=None, p0=None,XMAX=None):

    p, q = sfit(ff, X, Y, p0)

    if XMAX==None:
        XMAX = X[-1]+1

    XF = np.linspace(X[0], XMAX, res)
    YF = ff(XF, *p)

    return XF, YF, p, q


def lim_fit(func,X,Y,p0=None):

    smask  = np.ones_like(X).astype(bool)
    #smask[3] = False
    #smask[-1] = False
    bounds = ([4.,10,0,0],[30,40,50,10])
    if p0 == None:
        p0 = [12.,30.,8.,]
    return optimize.curve_fit(func,X[smask],Y[smask],
                              #sigma = Z[smask],#np.ones_like(Z)*np.max(Z),
                              #bounds= (0.00000000000000000001,np.inf),
                              #bounds=([10,0.001,0,0],[np.inf,np.inf,np.inf,np.inf]),
                              bounds = bounds,
                              p0=p0,
                              maxfev=50000,
                              )

def get_lim_fits(X,Y,res=200,ff=None,p1=None,XF=None):

    if p1 == None:
        p0 = [X[3],30, 7, 0.7,1]
    else:
        p0 = p1
    #p0    = None
    if ff==None:
        ff = itrf1

    else:
        p0 = p0[:-1]
    p ,q  = lim_fit(ff,X,Y,p0)


    #if XF == None:
    #   XF    = np.linspace(X[0],X[-1]+1,res)

    YF    = ff(XF,*p)

    return XF,YF,p,q


def lim_2_fit(func, X, Y, p0=None):

    smask = np.ones_like(X).astype(bool)
    # smask[3] = False
    # smask[-1] = False
    bounds = ([4., 20], [30, 40])
    if p0 == None:
        p0 = [5., 30.]
    return optimize.curve_fit(func, X[smask], Y[smask],
                              # sigma = Z[smask],#np.ones_like(Z)*np.max(Z),
                              # bounds= (0.00000000000000000001,np.inf),
                              # bounds=([10,0.001,0,0],[np.inf,np.inf,np.inf,np.inf]),
                              bounds=bounds,
                              p0=p0,
                              maxfev=50000,
                              )

def get_2_lim_fits(X, Y, res=200, ff=None, p1=None, XF=None):

    if p1 == None:
        p0 = None
    else:
        p0 = p1
    # p0    = None
    if ff == None:
        ff = itrf12

    else:
        p0 = p0[:-1]
    p, q = lim_2_fit(ff, X, Y, p0)

    # if XF == None:
    #   XF    = np.linspace(X[0],X[-1]+1,res)

    YF = ff(XF, *p)

    return XF, YF, p, q


def sfit(func, X, Y, p0=None):
    return optimize.curve_fit(func, X, Y,
                              p0=p0,
                              maxfev=5000000,
                              )





def get_max_diff(X,Y):
    dX,dY = np.diff(X),np.diff(Y)
    slope = np.diff(Y)/np.diff(X)
    return np.argmax(slope),np.max(slope)



def get_tomato_group(DATA, ff):

    XE,YE,XEF,YEF,PE,QE,XTS = [],[],[],[],[],[],[]

    for i in tqdm(DATA):
        xt,y0    = get_nod(i)
        xt0      = xt[0]
        x0       = xt-xt0
        x,y,p,q  = get_fits(x0,y0,ff=ff,XMAX=60)

        sn,slope = get_max_diff(x,y)
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

    PET   = np.asarray(PE)
    QET   = np.asarray(QE)
    XTS   = np.asarray(XTS)
    SMASK = np.argsort(XTS)
    XTS   = XTS[SMASK]
    PET   = (PET[SMASK]).T
    QET   = (QET[SMASK]).T
    return np.asarray(XTS),PET,QET,XE,YE



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


def filter_up(X,Y,val,ind):
    MASK = np.ones_like(X,dtype=bool)
    MASK[Y[ind]>val] = False

    return X[MASK],Y.T[MASK].T

def get_permutations(dim = 4):
    d2 = dim*dim
    return np.unpackbits(np.expand_dims(np.arange(d2, dtype=np.uint8),1),axis=1)[:,-dim:]


def mix_parameters(L1, L2):
    """Mixes two lists of paramters"""

    pmask = get_permutations(dim=len(L1))
    MINS  = np.tile(L1, (len(pmask), 1))
    MINS[pmask == 1] = 0

    MAXS  = np.tile(L2, (len(pmask), 1))
    MAXS[pmask == 0] = 0

    return MAXS+MINS

def get_error_fit(XF,P,Q, uL=40,lL=0):
    pqF = np.stack([P, np.sqrt(np.diag(Q))])
    pqFU = pqF[0]+pqF[1]
    pqFL = pqF[0]-pqF[1]

    sYFU = itrf12(XF, *pqFU)
    sYFL = itrf12(XF, *pqFL)

    pqALL = mix_parameters(pqFL, pqFU)
    FITS  = np.asarray([itrf12(XF, *i) for i in pqALL])
    FITS[FITS>uL] = uL
    FITS[FITS<lL] = lL
    return FITS

def get_sparse_data(X,Y):


    LEN   = len(X)
    XF    = np.linspace(0, 60, 200)
    XN,YN = [],[]
    YFN, pF, qF, dYFN, SP = [], [], [], [], []

    for i in range(4,LEN+1):
        XN.append(X[:i])
        YN.append(Y[:i])

    for i in range(len(XN)):
        _, yf, p, q = get_2_lim_fits(XN[i], YN[i], XF=XF, p1=[5., 30.])
        dyf = get_error_fit(XF, p, q)
        YFN.append(yf)
        dYFN.append(dyf)
        pF.append(p)
        qF.append(q)

        spread = dyf.max(0)-dyf.min(0)
        SP.append(spread)

    return XN, YN, XF, YFN, dYFN, pF, qF, np.asarray(SP)
