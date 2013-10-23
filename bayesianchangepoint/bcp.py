"""
 Implementation of:
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


def constant_hazard(r, _lambda):
    """
    A simple constant-rate hazard function that gives geomtrically-drawn
    intervals between changepoints.  We'll specify the rate via a mean.

    To quote the paper (section 2.1: "THE CHANGEPOINT PRIOR"):

        "In the special case where P_{gap}(g) is a discrete exponential 
        (geometric) distribution with timescale lambda, the process is
        memoryless and the hazard function is constant at H(tau) = 1/lambda"

    Args:
      * r (np.ndarray)
      * _lambda (float)

    Returns: probability of a changepoint
    """
    return (1 / _lambda) * np.ones(r.size)


def bcp(x, hazard_func, mu0=0, kappa0=1, alpha0=1, beta0=1):
    """
    Args:
      * hazard_func (function): 
        This is a handle to a function that takes one argument, the number of 
        time increments since the last changepoint, and returns a value in the
        interval [0,1] that is the probability of a changepoint.  
        e.g. hazard_func=lambda r: constant_hazard(r, 200)

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
    pass


def test():
    # First, we wil specify the prior.  We will then generate some fake data
    # from the prior specification.  We will then perform inference. Then
    # we'll plot some things.

    N = 1000 # how many data points to generate?
    hazard_func = lambda r: constant_hazard(r, _lambda=200)
    x = np.zeros(N) # this will hold the data
    cp = [0];
