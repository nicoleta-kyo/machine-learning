# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import scipy.io as spio
from scipy.stats import norm
from numpy.random import normal
import matplotlib.pyplot as plt
import numpy as np
from numpy import mean, std


# QUESTION 1

#Part 1

#mu = 0
#sigma = 30
#nbpoints = 5000
#mydata = np.random.normal(mu, sigma, nbpoints)
#plt.hist(mydata, bins = 50, normed=True)
#plt.hold(True)
#x = np.linspace(-150, 150, nbpoints)
#y = norm.pdf(x, mu, sigma)
#plt.plot(x,y)
##plt.show()

#Part 2

#Create 50 random 
#from sklearn.neighbors import KernelDensity
#nbpts = 50 # Number of points to generate (50 in the brief, can be varied)
#m= 1 # the mean
#s = 2 # the standard deviation
#
#data = normal(m,s,(nbpts,1))
#
## instantiate and fit the KDE model
#kde = KernelDensity(bandwidth=0.3, kernel='gaussian') #bandwidth is window, or "volume" - we fix it
#kde.fit(data)
#x_ax = np.linspace(-7, 7, nbpoints)
#
## score_samples returns the log of the probability density - we determine k
#logprob = kde.score_samples(x_ax[:, None])
#
#plt.fill_between(x_ax, np.exp(logprob), alpha=1.5) #check what's alpha???
#plt.plot(data, np.full_like(data, -0.01), '|k', markeredgewidth=1)
#plt.ylim(-0.03, 0.25)



# QUESTION 2

mat = spio.loadmat('sem3_q2_data.mat', squeeze_me=True);
data = mat['data']
#plt.hist(data) 
#
plt.figure(1)
plt.scatter(data[:,0],data[:,1], c='g');
data_norm = (data-mean(data,0))/std(data,0); # Note vectorised operations for mean and standard deviation. NB: Unlike Matlab, python will divide by element... so no need for a element-wise operator. You need to specify along which axis (row or column) the mean and the standard deviation are calculated. mean(data) would return a single number (mean over all columns/rows)
plt.scatter(data_norm[:,0],data_norm[:,1],c='r');

##the rest -- look at answers+explanations there!!

