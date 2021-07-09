import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
import math
import string
import pandas as pd
from scipy.stats import t 
from scipy.stats import norm
from statistics import mean
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Fig 4 global variables
u2     = 0  # data mean
sigma2 = 1 #data std
theta  = 0.01
alpha  = 10
kappa  = 20
c1     = (alpha-1)/(alpha-0.5)
c2     = 0.5*(alpha-1)/(alpha-0.5)*kappa/(kappa+1) # constant in betanew = c1*beta + c2*(x-mu)**2

#gride
NX   = 101
NY   = 101
xmax = 0.5
xmin = -0.5
ymax = 15
ymin = 5
dx   = (xmax-xmin)/(NX-1)
dy   = (ymax-ymin)/(NY-1)

# Fig 4 1/3 Steady state distribution of immune cell adaptation and responsiveness due to finite memory.
# dataw.txt

# Set maximum iteration
totaltime = 200
dt=0.001
NumOfTimeSteps=int(totaltime/dt)

#normalize functions
def normalize(raw):
	new = raw/np.sum(raw)/dx/dy
	return(new)

def coefficient(i,j):
	x  = xmin + dx*i
	y  = ymin + dy*j
	a1 = (u2-x)/(kappa+1)   #du=a1*dt+b1*dw dbeta=a2*dt+b2*dw
	b1 = (sigma2/(kappa+1))**2
	a2 = (c1-1)*y+c2*(u2-x)**2+c2*sigma2**2
	b2 = 2*c2**2*(sigma2**4+2*sigma2**2*(u2-x)**2)
	coef    = c2/(kappa+1)*2*(u2-x)*sigma2**2
	cow     = 1/(kappa+1) + 1 - c1 #dwdt = cow*w + codwdx*dwdx + codwdy*dwdy + codwdxx*dwdxx + codwdxy*dwdxy + codwdyy*dwdyy
	codwdx  = -a1
	codwdy  = -2*c2*sigma2**2/(kappa+1) - a2
	codwdxx = 0.5*b1
	codwdxy = coef
	codwdyy = 0.5*b2
	return(cow,codwdx,codwdy,codwdxx,codwdxy,codwdyy)
# get w and A0 ----------------------------------------------------------------
def initialize():
	# initial w
	w = np.empty(NX*NY)
	w.fill(1)
	#set boundary =0
	for ix in [0,NX-1]:
		for iy in range(NY):
			w[ix*NY+iy]=0
	for iy in [0,NY-1]:
		for ix in range(NX):
			w[ix*NY+iy]=0
	w = normalize(w)
	# calculate A0
	A0 = np.zeros((NX*NY,NX*NY))
	for ix in range(1,NX-1):
		for iy in range(1,NY-1):
			[cow,codwdx,codwdy,codwdxx,codwdxy,codwdyy]=coefficient(ix,iy)
			A0[ix*NY+iy,(ix-1)*NY+iy-1] = codwdxy/(4*dx*dy)
			A0[ix*NY+iy,(ix-1)*NY+iy+1] = -codwdxy/(4*dx*dy)
			A0[ix*NY+iy,(ix-1)*NY+iy] = -codwdx/(2*dx)+codwdxx/(dx**2)
			A0[ix*NY+iy,ix*NY+iy-1] = -codwdy/(2*dy)+codwdyy/(dy**2)
			A0[ix*NY+iy,ix*NY+iy] = cow-2*codwdxx/(dx**2)-2*codwdyy/(dy**2)
			A0[ix*NY+iy,ix*NY+iy+1] = codwdy/(2*dy)+codwdyy/(dy**2)
			A0[ix*NY+iy,(ix+1)*NY+iy-1] = -codwdxy/(4*dx*dy)
			A0[ix*NY+iy,(ix+1)*NY+iy] = codwdx/(2*dx)+codwdxx/(dx**2)
			A0[ix*NY+iy,(ix+1)*NY+iy+1] = codwdxy/(4*dx*dy)
	return(w,A0)
#------------------------------------------------------------------------
# get A in AX=b. A=I-A0/2*dt
def getA(A0):
	A=np.zeros((NX*NY,NX*NY))
	for ix in range(1,NX-1):
		for iy in range(1,NY-1):
			A[ix*NY+iy,(ix-1)*NY+iy-1] = -dt/2*A0[ix*NY+iy,(ix-1)*NY+iy-1]
			A[ix*NY+iy,(ix-1)*NY+iy+1] = -dt/2*A0[ix*NY+iy,(ix-1)*NY+iy+1]
			A[ix*NY+iy,(ix-1)*NY+iy] = -dt/2*A0[ix*NY+iy,(ix-1)*NY+iy]
			A[ix*NY+iy,ix*NY+iy-1] = -dt/2*A0[ix*NY+iy,ix*NY+iy-1]
			A[ix*NY+iy,ix*NY+iy] = -dt/2*A0[ix*NY+iy,ix*NY+iy]
			A[ix*NY+iy,ix*NY+iy+1] = -dt/2*A0[ix*NY+iy,ix*NY+iy+1]
			A[ix*NY+iy,(ix+1)*NY+iy-1] = -dt/2*A0[ix*NY+iy,(ix+1)*NY+iy-1]
			A[ix*NY+iy,(ix+1)*NY+iy] = -dt/2*A0[ix*NY+iy,(ix+1)*NY+iy]
			A[ix*NY+iy,(ix+1)*NY+iy+1] = -dt/2*A0[ix*NY+iy,(ix+1)*NY+iy+1]
	for ixa0 in range(NX*NY):
		A[ixa0,ixa0]+=1
	return(A)
#Atemp=(I+A/2*dt)
def getAtemp(A0):
	Atemp=np.zeros((NX*NY,NX*NY))
	for ix in range(1,NX-1):
		for iy in range(1,NY-1):
			Atemp[ix*NY+iy,(ix-1)*NY+iy-1] = dt/2*A0[ix*NY+iy,(ix-1)*NY+iy-1]
			Atemp[ix*NY+iy,(ix-1)*NY+iy+1] = dt/2*A0[ix*NY+iy,(ix-1)*NY+iy+1]
			Atemp[ix*NY+iy,(ix-1)*NY+iy] = dt/2*A0[ix*NY+iy,(ix-1)*NY+iy]
			Atemp[ix*NY+iy,ix*NY+iy-1] = dt/2*A0[ix*NY+iy,ix*NY+iy-1]
			Atemp[ix*NY+iy,ix*NY+iy] = dt/2*A0[ix*NY+iy,ix*NY+iy] 
			Atemp[ix*NY+iy,ix*NY+iy+1] = dt/2*A0[ix*NY+iy,ix*NY+iy+1] 
			Atemp[ix*NY+iy,(ix+1)*NY+iy-1] = dt/2*A0[ix*NY+iy,(ix+1)*NY+iy-1]
			Atemp[ix*NY+iy,(ix+1)*NY+iy] = dt/2*A0[ix*NY+iy,(ix+1)*NY+iy]
			Atemp[ix*NY+iy,(ix+1)*NY+iy+1] = dt/2*A0[ix*NY+iy,(ix+1)*NY+iy+1] 
	for ixa0 in range(NX*NY):
		Atemp[ixa0,ixa0]+=1
	return(Atemp)
#get b = Atemp*w
def getb(Atemp,w):
	b=[0]*(NX*NY)
	for ixa0 in range(NX*NY):
		b[ixa0] = np.dot(Atemp[ixa0],w)
	return(b)
#initial global constant------------------------------------------------------
[w,A0] = initialize()
A = csr_matrix(getA(A0))
Atemp=getAtemp(A0)

#data output
df = pd.DataFrame()
df['alpha'] = [alpha]*(NX*NY)
df['kappa'] = [kappa]*(NX*NY)
df['True mean'] = [u2]*(NX*NY)
df['True std'] = [sigma2]*(NX*NY)
m  = [0]*(NX*NY)
beta= [0]*(NX*NY)
for i in range(NX):
    for j in range(NY):
        m[i*NY+j] = xmin+i*dx
        beta[i*NY+j] = ymin+j*dy
df['M']   = m
df['Beta']= beta
df['w0'] = w

#main loop------------------------------------------------
for iteration in range(1,NumOfTimeSteps):
	b = getb(Atemp,w)
	w = spsolve(A,b)
	for i in range(len(w)):
		if w[i] < 0:
			w[i] = 0
	w = normalize(w)
	#output----------------------------------------------------
	if iteration%500==0:
		df['weight@step#%d'%iteration] = w
		df.to_csv('Fokker_planck_solution.csv.gz', index = None, header=True,compression='gzip')
