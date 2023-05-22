###################################################
# Title: Bayesian regression reconstructions with leads and lags
#
# Marco A. Aquino-López
###################################################
from os.path import expanduser
import sys
from numpy import savetxt, array, append,arange,  where, std, mean, sqrt, quantile, dot
from numpy.random import  uniform
from pandas import DataFrame, read_csv
from matplotlib.pyplot import subplots, plot, scatter, hist, figure, title, savefig, \
    show, xlim, ylim, xlabel, ylabel,  legend, fill_betweenx, specgram  # , axline, axhline
from tqdm import tqdm
import pytwalk

# This sets the directory where the data is 
directorio = expanduser("~") + '/MEGA_A/Cambridge/Bayesian_Correlation/'
'''
Model parameters
'''
# variables for the MCMC
iteratations = int(1.5e+03)
burnin = int(1e+3)
thi = int(20)
fast = True
# Prior standard deviations of the leads and the models sd
sd_delay = .5 / (5**2)  # sd = 5
sd_sd = .5 / (.1**2)  # sd =.1
"""
Loading data
"""
sys.path.append(directorio)
Data = read_csv(directorio + 'cesm.csv', sep=',')
# removes the first columns which is the row names
heads = array(Data.columns)
Data = Data.drop(columns=heads[0]
# creates the name columns for later use 
heads = array(Data.columns)
# Calculates the number of parameters
n_param = 2 * (len(heads) - 1)
# calculates the number of variables
n = int(n_param / 2)
# Defines the variables for the regression
X = Data.drop(columns=heads[0]).to_numpy()
# X has to be a nxm matrix where m 
y = Data[heads[0]]
# y has to be a 1xm matrix
'''
Model functions
'''


def supp(param):
    # Support function of the model 
    sd = param[0]
    shifts = param[n + 1:].astype('int')
    betas = param[1:n + 1]
    # checks that the sd is greater than 0
    if sd <= 0:
        return False
    else:
        # checks that the shifts are no greater than 5
        if max(abs(shifts)) > 5:
            return False
        else:
            return True


def shift(param):
    # Creates the "new dataset" given the leads and lags 
    shifts = param[n:].astype('int')
    mn, mx = [min(shifts), max(shifts)]
    new_y = y[-mn:-mx]
    len_y = len(new_y)
    new_X = DataFrame()
    for i in range(n):
        if shifts[i] >= 0:
            new_X[heads[i + 1]] = X[-mn + shifts[i]:-mn + shifts[i] + len_y, i]
        else:
            new_X[heads[i + 1]] = X[-mn + shifts[i]:-mn + shifts[i] + len_y, i]
    return [new_y.to_numpy(), new_X.to_numpy()]


def model(param):
    # Calculates difference between the model (calculated by the dot product)
    # and the observed y. 
    # we do it all in a single function in order to save one call to shift
    new_y, new_X = shift(param)
    diferencias = new_y - dot(new_X, param[:n])
    return diferencias


def Ux(param):  # -log-likelihood of the model 
    sd = param[0]
    param = param[1:]
    diffs = model(param)
    # here I calculate the variance as it is faster
    std = .5 / (sd**2)
    u = array(std * (diffs**2)).sum()
    return u

def prior(param): # -log-priors of the model
    sd = param[0]
    param = param[1:]
    prior = sum(.5 * (param[:n]**2))
    prior += sum(sd_delay * (param[n:]**2))
    prior += sd_sd * (sd**2)
    return prior


def ini(): # This function creates initial points for the MCMC
    sd = uniform(0, 10, size=1)
    betas = uniform(low=-.5, high=.5, size=n)
    deltas = uniform(low=-5, high=5, size=n)
    return append(sd, append(betas, deltas))


def obj(param): # Objective function fo the MCMC
    return(Ux(param) + prior(param))


'''
Run twalk
'''

if not(fast): # this checks for a quick MCMC or a complete one
    thi = thi * n_param
    burnin = burnin * n_param
    total_iterations = n_param * (burnin + (thi * iteratations))
else:
    total_iterations = burnin + (thi * iteratations)

# Run the mcmc
runtwalk = pytwalk.pytwalk(n=n_param + 1, U=obj, Supp=supp)
runtwalk.Run(T=total_iterations, x0=ini(), xp0=ini(), k=thi)
# save last points so they can be used as initial points in later runs
initial_poitns = DataFrame([runtwalk.x, runtwalk.xp])
initial_poitns.to_csv('inis_pf_spl.csv',   index=False)
# saves the results in a matrix file and puts them in the Output object
Output = runtwalk.Output[burnin:, ][::thi]
Output = Output[-iteratations:, :]
savetxt("twalk" + str(n_param) + "correlation_twalk.dat", Output, delimiter=",")

# This can call the file 
#Output = read_csv(directorio + 'twalk40correlation_twalk-Niceone.dat', sep=',')

# Plot the energy to check for convergence 
figure(figsize=(10, 6))
plot(Output[:, -1])

figure(figsize=(10,6))
hist(Output[:, 0])
title('posterior of the standard deviation')

# plot the leas and lags
figure(figsize=(10, 6))
hist(Output[:, n + 1:-1].astype('int'))
title('posterior of the leads and lags')

# Plot the results 
figure(figsize=(10, 6))
for i in tqdm(Output[:, 1:-1]):
    new_y, new_X = shift(i)
    modelo = dot(new_X, i[:n])
    indix0 = where(y.to_numpy() == new_y[0])[0]
    indix1 = where(y.to_numpy() == new_y[-1])[0]
    plot(arange(indix0, indix1 + 1), modelo, alpha=.01, color='black')
title('AMOC vs results')
plot(y.to_numpy(), color='red')

# Plot the differences
figure(figsize=(10, 6))
for i in tqdm(Output[:, 1:-1]):
    differencias = model(i)
    new_y, new_X = shift(i)
    indix0 = where(y.to_numpy() == new_y[0])[0]
    indix1 = where(y.to_numpy() == new_y[-1])[0]
    plot(arange(indix0, indix1 + 1), differencias, alpha=.01, color='black')
title('Differences between AMOC and regressions')

for i in range(len(heads)-1):
    figure(figsize=(10, 6))
    hist(Output[:, i+1])
    title('posterior of the contribution of ' + heads[i+1] +"to the regression")

show()


# hist(Output[:, n+5].astype('int'))
# show()




