"""
 An implementation of:
 @TECHREPORT{ adams-mackay-2007,
    AUTHOR = {Ryan Prescott Adams and David J.C. MacKay},
    TITLE  = "{B}ayesian Online Changepoint Detection",
    INSTITUTION = "University of Cambridge",
    ADDRESS = "Cambridge, UK",
    YEAR = "2007",
    NOTE = "arXiv:0710.3742v1 [stat.ML]"
 }
"""
from __future__ import print_function, division
import numpy as np
from numpy.random import gamma, randn, rand
from scipy.special import gammaln
import matplotlib.pyplot as plt


def constant_hazard(r, _lambda):
    """
    A simple constant-rate hazard function that gives geomtrically-drawn
    intervals between changepoints.  We'll specify the rate via a mean.

    To quote the paper (section 2.1: "THE CHANGEPOINT PRIOR"):

        "In the special case where P_{gap}(g) is a discrete exponential
        (geometric) distribution with timescale lambda, the process is
        memoryless and the hazard function is constant at H(tau) = 1/lambda"

    Args:
      * r (np.ndarray or scalar)
      * _lambda (float)

    Returns:
      * p (np.ndarray with shape = r.shape): probability of a changepoint
    """
    if isinstance(r, np.ndarray):
        shape = r.shape
    else:
        shape = 1

    probability = np.ones(shape) / _lambda
    return probability


def studentpdf(x, mu, var, nu):
    """
    Returns the pdf(x) for Student T distribution.

    scipy.stats.distributions.t.pdf(x=x-mu, df=nu) comes close
    to replicating studentpdf but Kevin Murphy's studentpdf
    function includes a 'var' variable which scipy's version does not.
    """
    # Using a mixture of code from studentpdf.m and
    # scipy.stats.distributions.t_gen._pdf()
    r = np.asarray(nu*1.0)
    c = np.exp(gammaln((r+1)/2) - gammaln(r/2))
    c /= np.sqrt(r * np.pi * var) * (1+((x-mu)**2)/(r*var))**((r+1)/2)
    return c


def row_to_column_vector(row_vector):
    return np.matrix(row_vector).transpose()


def inference(x, hazard_func, mu0=0, kappa0=1, alpha0=1, beta0=1):
    """
    Args:
      * x (np.ndarray): data
      * hazard_func (function):
        This is a handle to a function that takes one argument, the number of
        time increments since the last changepoint, and returns a value in the
        interval [0,1] that is the probability of a changepoint.
        e.g. hazard_func=lambda beliefs: constant_hazard(beliefs, 200)

      * mu0, kappa0, alpha0, beta0 (float): specify normal-inverse-gamma prior.
        This data is Gaussian with unknown mean and variance.  We are going to
        use the standard conjugate prior of a normal-inverse-gamma.  Note that
        one cannot use non-informative priors for changepoint detection in
        this construction.  The normal-inverse-gamma yields a closed-form
        predictive distribution, which makes it easy to use in this context.
        There are lots of references out there for doing this kind of inference:
          - Chris Bishop's "Pattern Recognition and Machine Learning" Chapter 2
          - Also, Kevin Murphy's lecture notes.
    """

    # First, setup the matrix that will hold our beliefs about the current
    # run lengths.  We'll initialize it all to zero at first.  Obviously
    # we're assuming here that we know how long we're going to do the
    # inference.  You can imagine other data structures that don't make that
    # assumption (e.g. linked lists).  We're doing this because it's easy.
    beliefs = np.zeros([x.size+1, x.size+1])

    # At time t=0, we actually have complete knowledge about the run
    # length.  It is definitely zero.  See the paper for other possible
    # boundary conditions.  'beliefs' is called 'R' in gaussdemo.m.
    beliefs[0,0] = 1.0

    # Convert floats to arrays
    mu0    = np.array([mu0])
    kappa0 = np.array([kappa0])
    alpha0 = np.array([alpha0])
    beta0  = np.array([beta0])

    # Track the current set of parameters.  These start out at the prior and
    # accumulate data as we proceed.
    muT    = mu0
    kappaT = kappa0
    alphaT = alpha0
    betaT  = beta0

    # Keep track of the maximums.
    maxes  = np.zeros([x.size+1, x.size+1])

    # Loop over the data like we're seeing it all for the first time.
    for t in range(x.size):

        # Evaluate the predictive distribution for the new datum under each of
        # the parameters.  This is the standard thing from Bayesian inference.
        predprobs = studentpdf(x[t], muT,
                               betaT*(kappaT+1)/(alphaT*kappaT),
                               2 * alphaT)

        # Evaluate the hazard function for this interval.
        haz = hazard_func(np.arange(t+1))

        # Evaluate the growth probabilities - shift the probabilities down and to
        # the right, scaled by the hazard function and the predictive
        # probabilities.
        beliefs[1:t+2,t+1] = beliefs[0:t+1,t] * predprobs * (1-haz)

        # Evaluate the probability that there *was* a changepoint and we're
        # accumulating the mass back down at beliefs = 0.
        beliefs[0,t+1] = (beliefs[0:t+1,t] * predprobs * haz).sum()

        # Renormalize the run length probabilities for improved numerical
        # stability.
        beliefs[:,t+1] = beliefs[:,t+1] / beliefs[:,t+1].sum()

        # Update the parameter sets for each possible run length.
        # TODO: continue porting from here...

        muT0    = np.concatenate([mu0   , (kappaT*muT + x[t]) / (kappaT+1) ])
        kappaT0 = np.concatenate([kappa0, kappaT + 1 ])
        alphaT0 = np.concatenate([alpha0, alphaT + 0.5 ])
        betaT0  = np.concatenate([beta0 , kappaT +
                                          (kappaT*(x[t]-muT)**2)/(2*(kappaT+1))])
        muT     = muT0
        kappaT  = kappaT0
        alphaT  = alphaT0
        betaT   = betaT0

        # Store the maximum, to plot later.
        maxes[t] = np.where(beliefs[:,t]==beliefs[:,t].max())[0]

    return beliefs, maxes

def generate_test_data(n, hazard_func, mu0=0, kappa0=1, alpha0=1, beta0=1):
    """
    Args:
      * n (int): number of data elements to return
      * hazard_func, mu0, kappa0, alpha0, beta0: see doc for inference()

    Returns: x, changepoints
      * x (np.ndarray of length n): data
      * changepoints (list of ints): indices of changepoints
    """
    x = np.zeros(n) # this will hold the data
    changepoints = [0] # Store the times of changepoints.  It's useful to see them.

    def generate_params():
        # Generate the parameters of the Gaussian from the prior.
        curr_ivar = gamma(alpha0) * beta0
        curr_mean = (((kappa0 * curr_ivar) ** -0.5) * randn()) + mu0
        return curr_ivar, curr_mean

    curr_ivar, curr_mean = generate_params()
    curr_run = 0 # Initial run length is zero

    # Now, loop forward in time and generate data.
    for t in range(n):

        # Get the probability of a new changepoint.
        p = hazard_func(curr_run)

        # Randomly generate a changepoint, perhaps.
        if rand() < p:
            # Generate new Gaussian parameters from the prior.
            curr_ivar, curr_mean = generate_params()

            # The run length drops back to zero.
            curr_run = 0

            # Add this changepoint to the end of the list.
            changepoints.append(t)
        else:
            # Increment the run length if there was no changepoint.
            curr_run += 1

        # Draw data from the current parameters.
        x[t] = (curr_ivar ** -0.5) * randn() + curr_mean

    return x, changepoints

def test(data_input='random'):
    # First, we will specify the prior.  We will then generate some fake data
    # from the prior specification.  We will then perform inference. Then
    # we'll plot some things.

    hazard_func = lambda r: constant_hazard(r, _lambda=200)

    if data_input == 'random':
        # generate test data
        N = 100 # how many data points to generate?
        x, changepoints = generate_test_data(N, hazard_func)
    elif data_input == 'ones':
        x = np.ones(N)
        changepoints = []
    elif data_input == 'signature':
        from pda.channel import Channel
        from os import path
        DATA_DIR = '/data/mine/domesticPowerData/BellendenRd/wattsUp'
        #SIG_DATA_FILENAME = 'breadmaker1.csv'
        SIG_DATA_FILENAME = 'washingmachine1.csv'
        chan = Channel()
        chan.load_wattsup(path.join(DATA_DIR, SIG_DATA_FILENAME))
        x = chan.series.values[142:1647]
        N = x.size

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.plot(x)
    ylim = ax.get_ylim()
    for cp in changepoints:
        ax.plot([cp, cp], ylim, color='k')

    # do inference
    beliefs, maxes = inference(x, hazard_func)

    # plot beliefs
    beliefs = beliefs.astype(np.float32)
    #print(beliefs)
    ax2 = fig.add_subplot(2,1,2, sharex=ax)
    ax2.imshow(-np.log(beliefs), interpolation='none', aspect='auto',
               origin='lower', cmap=plt.cm.Blues)
    ax2.plot(maxes, color='r')
    ax2.set_xlim([0, N])
    ax2.set_ylim([0, ax2.get_ylim()[1]])
    plt.draw()
    return beliefs, maxes

#test()
