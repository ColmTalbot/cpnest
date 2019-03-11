import os
import sys
import signal

import numpy as np

# import cProfile

from .sampler import HamiltonianMonteCarloSampler, MetropolisHastingsSampler
from .NestedSampling import NestedSampler
from .proposal import DefaultProposalCycle, HamiltonianProposalCycle
from .utils import CheckPoint


def sighandler(signal, frame):
    raise CheckPoint


class CPNest(object):
    """
    Class to control CPNest sampler
    cp = CPNest(usermodel, nlive=100, output='./', verbose=0, seed=None,
                maxmcmc=100, nthreads=None, balanced_sampling=True)
    
    Input variables:
    usermodel : an object inheriting cpnest.model.Model that defines the user's
                problem
    nlive : Number of live points (100)
    poolsize: Number of objects in the sampler pool (100)
    output : output directory (./)
    verbose: Verbosity, 0=silent, 1=progress, 2=diagnostic,
             3=detailed diagnostic
    seed: random seed (default: 1234)
    maxmcmc: maximum MCMC points for sampling chains (100)
    nthreads: number of parallel samplers. Default (None) uses mp.cpu_count()
              to autodetermine
    nhamiltomnian: number of sampler threads using an hamiltonian samplers.
                   Default: 0
    resume: determines whether cpnest will resume a run or run from scratch.
            Default: False.
    """
    def __init__(self, usermodel, nlive=100, poolsize=100, output='./',
                 verbose=0, seed=None, maxmcmc=100, nthreads=1,
                 nhamiltonian=0, resume=False):
        self.nthreads = nthreads
        self.nlive = nlive
        self.verbose = verbose
        self.output = output
        self.poolsize = poolsize
        self.posterior_samples = None
        self.user = usermodel
        self.resume = resume

        if seed is None:
            self.seed = 1234
        else:
            self.seed = seed
        
        self.process_pool = list()

        # instantiate the sampler class
        if nhamiltonian == 0:
            resume_file = os.path.join(output, "sampler.pkl")
            if not os.path.exists(resume_file) or not resume:
                sampler = MetropolisHastingsSampler(
                    self.user, maxmcmc, verbose=verbose, output=output,
                    poolsize=poolsize, seed=self.seed,
                    proposal=DefaultProposalCycle(), resume_file=resume_file)
            else:
                sampler = MetropolisHastingsSampler.resume(
                    resume_file, self.user)

        else:
            resume_file = os.path.join(output, "sampler.pkl")
            if not os.path.exists(resume_file) or not resume:
                sampler = HamiltonianMonteCarloSampler(
                    self.user, maxmcmc, verbose=verbose, output=output,
                    poolsize=poolsize, seed=self.seed,
                    proposal=HamiltonianProposalCycle(model=self.user),
                    resume_file=resume_file)
            else:
                sampler = HamiltonianMonteCarloSampler.resume(
                    resume_file, self.user)

        # instantiate the nested sampler class
        resume_file = os.path.join(output, "nested_sampler_resume.pkl")
        if not os.path.exists(resume_file) or not resume:
            self.NS = NestedSampler(
                self.user, nlive=nlive, output=output, verbose=verbose,
                seed=self.seed, prior_sampling=False, sampler=sampler)
        else:
            self.NS = NestedSampler.resume(resume_file, self.user)

    def run(self):
        """
        Run the sampler
        """
        if self.resume:
            signal.signal(signal.SIGTERM, sighandler)
            signal.signal(signal.SIGQUIT, sighandler)
            signal.signal(signal.SIGINT, sighandler)
            signal.signal(signal.SIGUSR2, sighandler)
        
        try:
            self.NS.nested_sampling_loop()
            for each in self.process_pool:
                each.join()
        except CheckPoint:
            self.checkpoint()
            sys.exit()

        self.posterior_samples = self.get_posterior_samples(filename=None)
        if self.verbose > 1:
            self.plot()
    
        #TODO: Clean up the resume pickles

    def get_nested_samples(self, filename='nested_samples.dat'):
        """
        returns nested sampling chain
        Parameters
        ----------
        filename : string
                   If given, file to save nested samples to

        Returns
        -------
        pos : :obj:`numpy.ndarray`
        """
        import numpy.lib.recfunctions as rfn
        self.nested_samples = rfn.stack_arrays(
            [s.asnparray() for s in self.NS.nested_samples], usemask=False)
        if filename:
            np.savetxt(os.path.join(
                self.NS.output_folder, 'nested_samples.dat'),
                self.nested_samples.ravel(),
                header=' '.join(self.nested_samples.dtype.names),
                newline='\n', delimiter=' ')
        return self.nested_samples

    def get_posterior_samples(self, filename='posterior.dat'):
        """
        Returns posterior samples

        Parameters
        ----------
        filename : string
                   If given, file to save posterior samples to

        Returns
        -------
        pos : :obj:`numpy.ndarray`
        """
        from .nest2pos import draw_posterior_many
        nested_samples = self.get_nested_samples()
        posterior_samples = draw_posterior_many(
            [nested_samples], [self.nlive], verbose=self.verbose)
        posterior_samples = np.array(posterior_samples)
        # TODO: Replace with something to output samples in whatever format
        if filename:
            np.savetxt(os.path.join(
                self.NS.output_folder, 'posterior.dat'),
                self.posterior_samples.ravel(),
                header=' '.join(posterior_samples.dtype.names),
                newline='\n', delimiter=' ')
        return posterior_samples

    def plot(self, corner=False):
        """
        Make diagnostic plots of the posterior and nested samples
        """
        pos = self.posterior_samples
        from . import plot
        for n in pos.dtype.names:
            plot.plot_hist(pos[n].ravel(), name=n,
                           filename=os.path.join(
                               self.output, 'posterior_{0}.png'.format(n)))
        for n in self.nested_samples.dtype.names:
            plot.plot_chain(self.nested_samples[n], name=n,
                            filename=os.path.join(
                                self.output, 'nschain_{0}.png'.format(n)))
        plotting_posteriors = np.squeeze(
            pos.view((pos.dtype[0], len(pos.dtype.names))))
        if corner:
            plot.plot_corner(plotting_posteriors, labels=pos.dtype.names,
                             filename=os.path.join(self.output, 'corner.png'))

    # def worker_sampler(self, producer_pipe, logLmin):
    #     cProfile.runctx('self.sampler.produce_sample(producer_pipe, logLmin)',
    #                     globals(), locals(), 'prof_sampler.prof')
    
    # def worker_ns(self):
    #     cProfile.runctx('self.NS.nested_sampling_loop(self.consumer_pipes)',
    #                     globals(), locals(), 'prof_nested_sampling.prof')

    # def profile(self):
    #     for i in range(0, self.NUMBER_OF_PRODUCER_PROCESSES):
    #         p = mp.Process(
    #             target=self.worker_sampler,
    #             args=(self.queues[i % len(self.queues)], self.NS.logLmin))
    #         self.process_pool.append(p)
    #     for i in range(0, self.NUMBER_OF_CONSUMER_PROCESSES):
    #         p = mp.Process(
    #             target=self.worker_ns,
    #             args=(self.queues, self.port, self.authkey))
    #         self.process_pool.append(p)
    #     for each in self.process_pool:
    #         each.start()

    def checkpoint(self):
        self.checkpoint_flag = 1
