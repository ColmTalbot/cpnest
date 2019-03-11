from __future__ import division
import sys
from math import log
from collections import deque
from random import random
import pickle

import numpy as np
from tqdm import tqdm

from .proposal import DefaultProposalCycle
from .utils import CheckPoint


class Sampler(object):
    """
    Sampler class.
    ---------
    
    Initialisation arguments:
    
    args:
    model: :obj:`cpnest.Model` user defined model to sample
    
    maxmcmc:
        :int: maximum number of mcmc steps to be used in the
        :obj:`cnest.sampler.Sampler`
    
    ----------
    kwargs:
    
    verbose:
        :int: display debug information on screen
        Default: 0
    
    poolsize:
        :int: number of objects for the affine invariant sampling
        Default: 1000
    
    seed:
        :int: random seed to initialise the pseudo-random chain
        Default: None
    
    proposal:
        :obj:`cpnest.proposals.Proposal` to use
        Defaults: :obj:`cpnest.proposals.DefaultProposalCycle`)
    
    resume_file:
        File for checkpointing
        Default: None
    
    manager:
        :obj:`multiprocessing.Manager` hosting all communication objects
        Default: None
    """

    def __init__(self, model, maxmcmc, seed=None, output=None, verbose=False,
                 poolsize=1000, proposal=None, resume_file=None):

        self.seed = seed
        self.model = model
        self.initial_mcmc = maxmcmc // 10
        self.maxmcmc = maxmcmc
        self.resume_file = resume_file
        # self.manager = manager
        self.logLmin = -np.inf
        
        if proposal is None:
            self.proposal = DefaultProposalCycle()
        else:
            self.proposal = proposal

        self.Nmcmc = self.initial_mcmc
        self.Nmcmc_exact = float(self.initial_mcmc)

        self.poolsize = poolsize
        self.evolution_points = deque(maxlen=self.poolsize)
        self.verbose = verbose
        self.acceptance = 0.0
        self.sub_acceptance = 0.0
        self.mcmc_accepted = 0
        self.mcmc_counter = 0
        self.initialised = False
        self.output = output
        # the list of samples from the mcmc chain
        self.samples = list()
        # the history of the ACL of the chain, will be used to thin the output,
        # if requested
        self.ACLs = []
        # self.producer_pipe, self.thread_id = self.manager.connect_producer()
        self.live = None
        
    def reset(self):
        """
        Initialise the sampler by generating :int:`poolsize`
        `cpnest.parameter.LivePoint` and distributing them according to
        :obj:`cpnest.model.Model.log_prior`
        """
        np.random.seed(seed=self.seed)
        for _ in tqdm(
                range(self.poolsize),
                desc='SMPLR init draw'.format(),
                disable=not self.verbose, leave=False):
            while True:  # Generate an in-bounds sample
                p = self.model.new_point()
                p.logP = self.model.log_prior(p)
                if np.isfinite(p.logP):
                    break
            p.logL = self.model.log_likelihood(p)
            if p.logL is None or not np.isfinite(p.logL):
                print("Warning: received non-finite logL value {0} with "
                      "parameters {1}".format(str(p.logL), str(p)))
                print("You may want to check your likelihood function to "
                      "improve sampling")
            self.evolution_points.append(p)

        self.proposal.set_ensemble(self.evolution_points)
        # Now, run evolution so samples are drawn from actual prior
        for _ in tqdm(
                range(self.poolsize),
                desc='SMPLR init evolve',
                disable=not self.verbose, leave=False):
            _, p = next(self.yield_sample(-np.inf))

        self.proposal.set_ensemble(self.evolution_points)
        self.counter = 0
        self.initialised = True

    def estimate_nmcmc(self, safety=3, tau=None):
        """
        Estimate autocorrelation length of chain using acceptance fraction
        ACL = (2/acc) - 1
        multiplied by a safety margin of 5
        Uses moving average with decay time tau iterations
        (default: :int:`self.poolsize`)
        
        Taken from http://github.com/farr/Ensemble.jl
        """
        if tau is None:
            tau = self.maxmcmc / safety  # self.poolsize

        if self.sub_acceptance == 0.0:
            self.Nmcmc_exact = (1.0 + 1.0 / tau) * self.Nmcmc_exact
        else:
            self.Nmcmc_exact = (
                (1.0 - 1.0 / tau) * self.Nmcmc_exact + (safety / tau) *
                (2.0 / self.sub_acceptance - 1.0))

        self.Nmcmc_exact = min(self.Nmcmc_exact, self.maxmcmc)
        self.Nmcmc = int(self.Nmcmc_exact)

        return self.Nmcmc

    def produce_sample(self):
        try:
            return self._produce_sample()
        except CheckPoint:
            self.checkpoint()
    
    def _produce_sample(self):
        """
        main loop that takes the worst :obj:`cpnest.parameter.LivePoint` and
        evolves it. Proposed sample is then sent back
        to :obj:`cpnest.NestedSampler`.
        """
        if not self.initialised:
            self.reset()

        self.evolution_points.append(self.live)
        self.logLmin = self.live.logL
        Nmcmc, outParam = next(self.yield_sample(self.logLmin))
        self.counter += 1

        if (self.counter % (self.poolsize//10)) == 0:
            self.proposal.set_ensemble(self.evolution_points)

        return self.acceptance, self.sub_acceptance, Nmcmc, outParam

    def checkpoint(self):
        """
        Checkpoint its internal state
        """
        print('Checkpointing Sampler')
        with open(self.resume_file, "wb") as f:
            pickle.dump(self, f)
        sys.exit(0)

    @classmethod
    def resume(cls, resume_file, model):
        """
        Resumes the interrupted state from a
        checkpoint pickle file.
        """
        print('Resuming Sampler from '+resume_file)
        with open(resume_file, "rb") as f:
            obj = pickle.load(f)
        obj.model = model
        return(obj)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['model']
        # del state['thread_id']
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self.manager = None

    def yield_sample(self, logLmin):
        raise NotImplementedError()


class MetropolisHastingsSampler(Sampler):
    """
    metropolis-hastings acceptance rule
    for :obj:`cpnest.proposal.EnembleProposal`
    """
    def yield_sample(self, logLmin):
        
        while True:
            
            sub_counter = 0
            sub_accepted = 0
            oldparam = self.evolution_points[0]
            logp_old = oldparam.logP

            while True:
                
                sub_counter += 1
                newparam = self.proposal.get_sample(oldparam.copy())
                newparam.logP = self.model.log_prior(newparam)
                
                if newparam.logP-logp_old + self.proposal.log_J > log(random()):
                    newparam.logL = self.model.log_likelihood(newparam)
                    if newparam.logL > logLmin:
                        oldparam = newparam.copy()
                        logp_old = newparam.logP
                        sub_accepted += 1
            
                if (sub_counter >= self.Nmcmc and sub_accepted > 0) or\
                        (sub_counter >= self.maxmcmc):
                    break

            self.sub_acceptance = sub_accepted / sub_counter
            self.estimate_nmcmc()
            self.mcmc_accepted += sub_accepted
            self.mcmc_counter += sub_counter
            self.acceptance = self.mcmc_accepted / self.mcmc_counter
            yield (sub_counter, oldparam)


class HamiltonianMonteCarloSampler(Sampler):
    """
    HamiltonianMonteCarlo acceptance rule
    for :obj:`cpnest.proposal.HamiltonianProposal`
    """
    def yield_sample(self, logLmin):
        
        while True:
            
            sub_accepted = 0
            sub_counter = 0
            oldparam = self.evolution_points[0]

            while sub_accepted == 0:

                sub_counter += 1
                newparam = self.proposal.get_sample(
                    oldparam.copy(), logLmin=np.minimum(oldparam.logL, logLmin))
                
                if self.proposal.log_J > np.log(random()):
                    if newparam.logL > logLmin:
                        oldparam = newparam.copy()
                        sub_accepted += 1
        
            # if self.verbose >= 3:
            #     self.samples.append(oldparam)
            
            self.sub_acceptance = sub_accepted / sub_counter
            self.mcmc_accepted += sub_accepted
            self.mcmc_counter += sub_counter
            self.acceptance = self.mcmc_accepted / self.mcmc_counter
#
            for p in self.proposal.proposals:
                p.update_time_step(self.acceptance, self.estimate_nmcmc())

            yield (sub_counter, oldparam)

    def insert_sample(self, p):
        # if we did not accept, inject a new particle in the system
        # (grand-canonical) from the prior by picking one from the existing
        # pool and giving it a random trajectory
        k = np.random.randint(self.evolution_points.maxlen)
        self.evolution_points.rotate(k)
        p = self.evolution_points.pop()
        self.evolution_points.append(p)
        self.evolution_points.rotate(-k)
        return self.proposal.get_sample(p.copy(), logLmin=p.logL)


