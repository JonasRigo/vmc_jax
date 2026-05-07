##############################################
# J Rigo
# rigojonas@gmail.com
# Regensburg 20/11/2023
# Implementation of https://arxiv.org/abs/2305.14294
# by Sinibaldi et al. 
##############################################

import jax
from jax import jit, vmap, grad#, partial
import jax.numpy as jnp

import jVMC.global_defs as global_defs
import jVMC.mpi_wrapper as mpi
from jVMC.stats import SampledObs
import jVMC.operator as op


import functools
import itertools


class MovingAverage(object):
    """This class provides functionality to compute the moving average of a given quantity.
        :math:`\\frac{1}{N}\\sum_{i=k}^{N} x_{k+i}`
    """
    def __init__(self,width = 10, init = 0.):
        self.values = jnp.zeros(width)

    def update(self,value):
        self.values = jnp.roll(self.values,-1)
        self.values =  self.values.at[-1].set(value)
        return jnp.sum(self.values)/jnp.array(self.values).shape[-1]

def OperatorWrapper(Operator, psi):
    """This function wraps a given operator around a neural quantum state.

    Arguments:
        * ``Operator``: Operator to be wrapped.
        * ``psi``: Neural quantum state.

    Returns:
        Wrapped neural quantum state.
    """
    def wrapped_psi(s):
        # Get connected configurations and corresponding matrix elements
        _, _ = Operator.get_s_primes(s)

        # Use the operator-internal (unflattened) connected states so we keep
        # the mapping between each input s and its specific s' set.
        sp = Operator.sp
        matEl = Operator.matEl

        # Flatten connected configs to evaluate psi
        sp_flat = sp.reshape(sp.shape[0], -1, *sp.shape[3:])
        logPsiSP = psi(sp_flat)
        if not getattr(psi, "logarithmic", True):
            logPsiSP = jnp.log(logPsiSP)

        # Reshape to match matEl layout and apply the operator
        logPsiSP = logPsiSP.reshape(matEl.shape)
        amp = jnp.sum(matEl * jnp.exp(logPsiSP), axis=-1)
        return jnp.log(amp)

    wrapped_psi.logarithmic = True
    return wrapped_psi

class _InfidelityP(op.Operator):
    """This class provides functionality to compute ratios of neural quantum states :math:`\\frac{\\chi(s)}{\\psi(s)}`. More specifically it implements

        :math:`\sum_{s}\\delta_{ss'}\\frac{\\chi(s)}{\\psi(s')}` 


    Initializer arguments:
        None
    """   

    def compile(self):

        def map_function(s, *args,):

            sp = s.copy()
            configShape = sp.shape

            matEl = jnp.array([1.], dtype=global_defs.tCpx)

            return sp.reshape(1,*configShape), matEl

        return map_function

    def get_O_loc(self, samples, logPsiS, chi, Observables, *args):
        """Compute :math:`O_{loc}(s)`.

        If the instance parameter ElocBatchSize is larger than 0 :math:`O_{loc}(s)` is computed in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``*args``: Further positional arguments for the operator.

        Returns:
            :math:`O_{loc}(s)` for each configuration :math:`s`.
        """

        if self.ElocBatchSize > 0:
            return self.get_O_loc_batched(samples, logPsiS, chi, self.ElocBatchSize, Observables, *args)
        else:
            return self.get_O_loc_unbatched(logPsiS, chi(samples), Observables)
        

    def get_O_loc_unbatched(self, logPsiS, logChiS, Observables=[lambda x: x]):
        """Compute :math:`O_{loc}(s)`.

        This member function assumes that ``get_s_primes(s)`` has been called before, as \
        internally stored matrix elements :math:`O_{s,s'}` are used.

        Computes :math:`O_{loc}(s)=\sum_{s'} O_{s,s'}\\frac{\psi(s')}{\psi(s)}`, given the \
        logarithmic wave function amplitudes of the involved configurations :math:`\\ln(\psi(s))` \
        and :math:`\\ln\psi(s')`

        Arguments:
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``logPsiSP``: Logarithmic amplitudes :math:`\\ln(\psi(s'))`

        Returns:
            :math:`O_{loc}(s)` for each configuration :math:`s`.
        """

        return [self._get_O_loc_pmapd(self.matEl, vmap(ol)(logPsiS), vmap(ol)(logChiS)) for ol in Observables]
    

    def get_O_loc_batched(self, samples, logPsiS, chi, batchSize, Observables, *args):
        """Compute :math:`F^\\psi_{\\rm loc}(s)` in batches.

        Computes :math:`F^\\psi_{\\rm loc}(s)=F^\\chi\\frac{\chi(s)}{\psi(s)}` in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``batchSize``: Batch size.
            * ``Observables``: functions modifying the Logarithmic amplitudes.
            * ``*args``: Further positional arguments for the operator.

        Returns:
            :math:`1 - F^\\psi_{\\rm loc}(s)F^\\chi` for each configuration :math:`s`.
        """

        Oloc_set = [None for _ in Observables]

        numSamples = samples.shape[1]
        numBatches = numSamples // batchSize
        remainder = numSamples % batchSize

        # Minimize mismatch
        if remainder > 0:
            batchSize = numSamples // (numBatches+1)
            numBatches = numSamples // batchSize
            remainder = numSamples % batchSize

        for b in range(numBatches):

            batch = self._get_config_batch_pmapd(samples, b * batchSize, batchSize)
            logPsiSbatch = self._get_logPsi_batch_pmapd(logPsiS, b * batchSize, batchSize)

            # internalize the new batch
            sp, _ = self.get_s_primes(batch, *args)
            # get log chi batched
            logChiSbatch = chi(sp)#self._flatten_pmapd(sp))
            # modified the get_O_loc_unbatched function
            OlocBatch_set = self.get_O_loc_unbatched(logPsiSbatch,logChiSbatch,Observables)
            
            for ol_ind,ol in enumerate(Oloc_set):
                if ol is None:
                    if OlocBatch_set[ol_ind].dtype == global_defs.tCpx:
                        Oloc_set[ol_ind] = self._alloc_Oloc_cpx_pmapd(samples)
                    else:
                        Oloc_set[ol_ind] = self._alloc_Oloc_real_pmapd(samples)

            ##############
            # regular infideltiy
            Oloc_set = [self._insert_Oloc_batch_pmapd(Oloc, OlocBatch, b * batchSize) for Oloc, OlocBatch in zip(Oloc_set, OlocBatch_set)]
        
        if remainder > 0:

            batch = self._get_config_batch_pmapd(samples, numBatches * batchSize, remainder)
            batch = global_defs.pmap_for_my_devices(expand_batch, static_broadcasted_argnums=(1,))(batch, batchSize)
            logPsiSbatch = self._get_logPsi_batch_pmapd(logPsiS, numBatches * batchSize, numSamples % batchSize)
            logPsiSbatch = global_defs.pmap_for_my_devices(expand_batch, static_broadcasted_argnums=(1,))(logPsiSbatch, batchSize)

            # internalize the new batch
            sp, _ = self.get_s_primes(batch, *args)
            # get log chi batched
            logChiSbatch = chi(sp)#self._flatten_pmapd(sp))
            ##############
            # regular infideltiy
            # modified the get_O_loc_unbatched function
            OlocBatch_set = self.get_O_loc_unbatched(logPsiSbatch,logChiSbatch,Observables)
        
            OlocBatch_set = [self._get_Oloc_slice_pmapd(OlocBatch, 0, remainder) for OlocBatch in OlocBatch_set]
            # get the second part of the infideltiy
            Oloc_set = [self._insert_Oloc_batch_pmapd(Oloc, OlocBatch, b * batchSize) for Oloc, OlocBatch in zip(Oloc_set, OlocBatch_set)]
                
        # return the local quantity
        return Oloc_set

#########################################################

class Infidelity(op.Operator):
    """This class provides functionality to compute and optimize the infidelity given a target wave function :math:`\chi(s')` and a trial wave function :math:`\psi(s)`.

    The infidelity for two pure states is defined as
    
        :math:`\\mathcal{I} = 1 - \\sum_s \\Psi(s)\\frac{\\chi(s)}{\\psi(s)}\\times\\sum_{s'} \\Chi(s')\\frac{\\psi(s')}{\\chi(s')} = 1 - \\mathbb{E}_\\Psi\\frac{\\chi(s)}{\\psi(s)}\\times\\mathbb{E}_\\Chi\\frac{\\psi(s')}{\\chi(s')}`

    where we have introduced the short-hand for the Born distribution for the target and trial wave function
    
        :math:`\\Psi(s) = \\frac{\\psi(s)}{\\sum_s|\\psi(s)|^2} \\qquad \\Chi(s') = \\frac{\\chi(s')}{\\sum_{s'}|\\chi(s')|^2}` 

    With the local infidelity 
    
        :math:`F^\\psi_{\\rm loc}(s) = \\frac{\\chi(s)}{\\psi(s)}\\times\\sum_{s'} \\qquad F^\\chi_{\\rm loc}(s') = \\frac{\\psi(s')}{\\chi(s')}`

    the infidelity can be expressed as 

        :math:`\\mathcal{I} = 1 - \\mathbb{E}_\\Psi F^\\psi_{\\rm loc}(s)\\times\mathbb{E}_\\Chi F^\\chi_{\\rm loc}(s')`

    In this class the :math:`F^\\psi` and :math:`F^\\chi` are computed separately, such that :math:`F^\\chi` can be computed once and reused for any :math:`\\psi`.


    Initializer arguments:

        * ``chi``: Target wave function as neural quantum state.
        * ``chiSampler``: Sampler initialized with ``chi``
        * ``ElocBatchSize``: Batch size for the local infidelity. Default is -1.
        * ``getCV``: Bool to include control variates. Default is False.
        * ``adaptCV``: Bool to adapt the control variates. Default is False.
        * ``MovingAverageWidth``: Width of the moving average. Default is 1.
    """
    
    def __init__(self, chi, chiSampler, ElocBatchSize=-1, Operator=None, ConjugatedOperator=None, getCV=False, adaptCV=False, MovingAverageWidth=1):

        #########################################################

        # save the target wave funciton internally
        self.chi = chi
        # save the target sampler internally
        self.chiSampler = chiSampler
        # control variates constant
        self.CVc = -0.5
        # control variates flag
        self.getCV = getCV
        # control variates adaptation flag
        self.adaptCV = adaptCV
        # user error check
        if adaptCV and not getCV:
            print("WARNING: adaptCV is set to True, but getCV is set to False. Setting adaptCV to True.")
            self.getCV = True
        # moving average
        self.rmean = MovingAverage(MovingAverageWidth)

        ######## get samples ########
        self.chi_s, self.chi_logChi, self.chi_p = self.chiSampler.sample()
        # get the Infidelity to global scope
        self._InfidelityP = _InfidelityP(ElocBatchSize)
        # needs to be called for initalization
        _, _ = self._InfidelityP.get_s_primes(self.chi_s)
        # check if was called
        self._FP_loc_check = False

        super().__init__(ElocBatchSize=ElocBatchSize)  # Constructor of base class Operator has to be called!

        ##### create the operator wrapper ######
        if Operator is None:
            self._OperatorWrapper = lambda psi: psi
        else:
            self._OperatorWrapper = functools.partial(OperatorWrapper, Operator)

        if ConjugatedOperator is None:
            self._ConjugatedOperatorWrapper = self._OperatorWrapper
        else:
            self._ConjugatedOperatorWrapper = functools.partial(OperatorWrapper, ConjugatedOperator)


    def compile(self):
        """This function computes ratios of neural quantum states :math:`\\frac{\\chi(s)}{\\psi(s)}`. More specifically it implements

            :math:`\sum_{s}\\delta_{ss'}\\frac{\\chi(s)}{\\psi(s')}` 


        Initializer arguments:
            None
        """   
        
        # check if get_FP_loc was called
        if not self._FP_loc_check:
            raise ValueError("get_FP_loc was not called")

        def map_function(s, *args,):

            sp = s.copy()
            configShape = sp.shape

            matEl = jnp.array([1.], dtype=global_defs.tCpx)

            return sp.reshape(1,*configShape), matEl

        return map_function        

    def get_FP_loc(self, psi, sample_chi = True):
        """ Compute :math:`F^\\chi` given the target wave function :math:`\\chi` and trial wave function :math:`\\psi`.

        Arguments:
            * ``psi``: Neural quantum state.

        Returns:
            * :math:`F^\\chi_{\\rm loc}(s)` for each configuration :math:`s`.
            * :math:`F^\\chi`    
        """

        ######## wrap the wave function with the observable ########
        OPpsi = self._ConjugatedOperatorWrapper(psi)

        ######## get samples ########
        if sample_chi:
            self.chi_s, self.chi_logChi, self.chi_p = self.chiSampler.sample()
        # confirm that the function was called
        self._FP_loc_check = True
        # control variates stabilization of the local infidelity
        if self.adaptCV:
            Oloc_set = self._InfidelityP.get_O_loc(self.chi_s,
                                        self.chi_logChi,
                                        OPpsi,
                                        [lambda x: x, lambda x: 2.*x.real, lambda x: 4.*x.real, lambda x: x + 2. * x.real])
            # compute F^\chi_loc
            self.chi_Floc = Oloc_set[0]
            # compute the control variates
            self.chi_FlocCV = Oloc_set[1]
            # square squared fidelity
            self.chi_F2locCV = Oloc_set[2]
            # square infedlity times infidelity
            self.chi_Floc2FlocCV = Oloc_set[3]
            # compute the control variates
            self.Exp_chi_FlocCV = mpi.global_mean(self.chi_FlocCV,self.chi_p)
            # compute F^chi
            self.Exp_chi_Floc = mpi.global_mean(self.chi_Floc,self.chi_p)
        elif self.getCV:
            Oloc_set = self._InfidelityP.get_O_loc(self.chi_s,
                                        self.chi_logChi,
                                        OPpsi,
                                        Observables=[lambda x: x,lambda x: 2. * x. real])
            # compute F^\chi_loc
            self.chi_Floc = Oloc_set[0]
            # compute the control variates
            self.chi_FlocCV = Oloc_set[1]
            # compute the control variates
            self.Exp_chi_FlocCV = mpi.global_mean(self.chi_FlocCV,self.chi_p)
            # compute F^chi
            self.Exp_chi_Floc = mpi.global_mean(self.chi_Floc,self.chi_p)
        else:
            # compute F^\chi_loc
            Oloc_set = self._InfidelityP.get_O_loc(self.chi_s,
                                        self.chi_logChi,
                                        OPpsi,
                                        Observables=[lambda x: x])
            self.chi_Floc = Oloc_set[0]
            # compute F^chi
            self.Exp_chi_Floc = mpi.global_mean(self.chi_Floc,self.chi_p)
        return self.chi_Floc, self.Exp_chi_Floc
    
    
    def get_gradient(self, psi, psi_p, getCVgrad=False, CVscale = 1.):
        """ Compute :math:`\\nabla\\mathcal{I}`

        The gradient is computed from almost exclusively internally stored quantities

            :mat:`2\\mathbb{E}_\\Psi\\big[ \\Re~ O_k(s)F^\\psi_{\\rm loc}(s)\\big] \\, F^\\chi-2\\mathbb{E}_\\Psi\\big[ \\Re~ O_k(\\sigma)\\big] \\, F^\\psi \\, F^\\chi`        

        where :math:`O_k(s)` is the log derivative of the trial wave function.
        Note that the control variate contribution is not included in the gradient.

        Arguments:
            * ``psi``: Neural quantum state.
            * ``psi_p``: Born distribution :math:`\Psi(s)` evaluated at the samples
            * ``getCVgrad``: Bool to include control variate gradient
            * ``corrections``: Bool to include gradient correction
            * ``corrections2``: Bool to include gradient correction for the control variates

        Returns:
            * :math:`F^\\chi_{\\rm loc}(s)` for each configuration :math:`s`.
            * :math:`O_k(s)` for each configuration :math:`s` as a SampledObs object.
        """

        # compute the gradient throught the covariance
        Opsi = psi.gradients(self._flatten_pmapd(self.sp))
        grads = SampledObs(Opsi, psi_p)
        Floc = SampledObs(self.psi_Floc, psi_p)
        grad = (2.*grads.covar(Floc)*self.Exp_chi_Floc).squeeze(-1) 
        return -1. * (grad).real, grads
        

    def get_O_loc(self, samples, psi, logPsiS, psi_p = None, **kwargs):
        """Compute :math:`F^\\psi_{\\rm loc}(s)` in batches.

        Computes :math:`F^\\psi_{\\rm loc}(s)=F^\\chi\\frac{\chi(s)}{\psi(s)}` in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``psi_p``: Born distribution :math:`\Psi(s)` as created by sample funciton
            * ``**kwargs``: Further positional arguments for the operator.

        Returns:
            :math:`1 - F^\\psi_{\\rm loc}(s)F^\\chi` for each configuration :math:`s`.
        """
        ######## wrap the wave function with the observable ########
        OPchi = self._OperatorWrapper(self.chi)

        if self.adaptCV:
            # check if psi_p is None
            if psi_p is None:
                raise ValueError("psi_p is None. Set psi_p to the probabilities")
            # compute the local infidelity
            Oloc_set = self._O_loc_dist(samples,
                                        logPsiS, 
                                        OPchi,
                                        [lambda x: x, lambda x: 2.*x.real, lambda x: 4.*x.real, lambda x: x + 2. * x.real]
                                        )
            self.psi_Floc = Oloc_set[0]
            self.psi_FlocCV = Oloc_set[1]
            self.psi_F2locCV = Oloc_set[2]
            self.psi_Floc2FlocCV = Oloc_set[3]
            # compute the control variates
            self._get_CVc(psi_p)
            if "CVc" in kwargs:
                CVc = kwargs["CVc"]
            else:
                CVc = self.CVc
            # reset the samples
            self.sp = jnp.expand_dims(samples,-2)
            # return the control variates infidelity
            return 1. - self.psi_Floc * self.Exp_chi_Floc - CVc * (self.psi_FlocCV*self.Exp_chi_FlocCV - 1.)
        
        elif self.getCV:
            Oloc_set = self._O_loc_dist(samples,
                                        logPsiS, 
                                        OPchi,
                                        [lambda x: x, lambda x: 2.*x.real]
                                        )
            self.psi_Floc = Oloc_set[0]
            self.psi_FlocCV = Oloc_set[1]
            # reset the samples
            self.sp = jnp.expand_dims(samples,-2)
            
            return 1. - self.psi_Floc * self.Exp_chi_Floc - self.CVc * (self.psi_FlocCV*self.Exp_chi_FlocCV - 1.)
        
        else:
            Oloc_set = self._O_loc_dist(samples,
                                        logPsiS, 
                                        OPchi,
                                        [lambda x: x],)
            self.psi_Floc = Oloc_set[0]
            # reset the samples
            self.sp = jnp.expand_dims(samples,-2)
            return 1. - self.psi_Floc * self.Exp_chi_Floc
        

    def _O_loc_dist(self, samples, logPsiS, chi, Observables, *args):
        """Compute :math:`O_{loc}(s)`.

        If the instance parameter ElocBatchSize is larger than 0 :math:`O_{loc}(s)` is computed in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``*args``: Further positional arguments for the operator.

        Returns:
            :math:`O_{loc}(s)` for each configuration :math:`s`.
        """

        if self.ElocBatchSize > 0:
            return self.get_O_loc_batched(samples, logPsiS, chi, self.ElocBatchSize, Observables, *args)
        else:
            return self.get_O_loc_unbatched(logPsiS, chi(samples), Observables) 
        
    def get_O_loc_unbatched(self, logPsiS, logChiS, Observables=[lambda x: x]):
        """Compute :math:`O_{loc}(s)`.

        This member function assumes that ``get_s_primes(s)`` has been called before, as \
        internally stored matrix elements :math:`O_{s,s'}` are used.

        Computes :math:`O_{loc}(s)=\sum_{s'} O_{s,s'}\\frac{\psi(s')}{\psi(s)}`, given the \
        logarithmic wave function amplitudes of the involved configurations :math:`\\ln(\psi(s))` \
        and :math:`\\ln\psi(s')`

        Arguments:
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``logPsiSP``: Logarithmic amplitudes :math:`\\ln(\psi(s'))`

        Returns:
            :math:`O_{loc}(s)` for each configuration :math:`s`.
        """

        return [self._get_O_loc_pmapd(self.matEl, vmap(ol)(logPsiS), vmap(ol)(logChiS)) for ol in Observables]
    

    def get_O_loc_batched(self, samples, logPsiS, chi, batchSize, Observables, *args):
        """Compute :math:`F^\\psi_{\\rm loc}(s)` in batches.

        Computes :math:`F^\\psi_{\\rm loc}(s)=F^\\chi\\frac{\chi(s)}{\psi(s)}` in a batch-wise manner
        to avoid out-of-memory issues.

        Arguments:
            * ``samples``: Sample of computational basis configurations :math:`s`.
            * ``psi``: Neural quantum state.
            * ``logPsiS``: Logarithmic amplitudes :math:`\\ln(\psi(s))`
            * ``batchSize``: Batch size.
            * ``Observables``: functions modifying the Logarithmic amplitudes.
            * ``*args``: Further positional arguments for the operator.

        Returns:
            :math:`1 - F^\\psi_{\\rm loc}(s)F^\\chi` for each configuration :math:`s`.
        """

        Oloc_set = [None for _ in Observables]

        numSamples = samples.shape[1]
        numBatches = numSamples // batchSize
        remainder = numSamples % batchSize

        # Minimize mismatch
        if remainder > 0:
            batchSize = numSamples // (numBatches+1)
            numBatches = numSamples // batchSize
            remainder = numSamples % batchSize

        for b in range(numBatches):

            batch = self._get_config_batch_pmapd(samples, b * batchSize, batchSize)
            logPsiSbatch = self._get_logPsi_batch_pmapd(logPsiS, b * batchSize, batchSize)

            # internalize the new batch
            sp, _ = self.get_s_primes(batch, *args)
            # get log chi batched
            logChiSbatch = chi(sp)#self._flatten_pmapd(sp))
            # modified the get_O_loc_unbatched function
            OlocBatch_set = self.get_O_loc_unbatched(logPsiSbatch,logChiSbatch,Observables)
            
            for ol_ind,ol in enumerate(Oloc_set):
                if ol is None:
                    if OlocBatch_set[ol_ind].dtype == global_defs.tCpx:
                        Oloc_set[ol_ind] = self._alloc_Oloc_cpx_pmapd(samples)
                    else:
                        Oloc_set[ol_ind] = self._alloc_Oloc_real_pmapd(samples)

            ##############
            # regular infideltiy
            Oloc_set = [self._insert_Oloc_batch_pmapd(Oloc, OlocBatch, b * batchSize) for Oloc, OlocBatch in zip(Oloc_set, OlocBatch_set)]
        
        if remainder > 0:

            batch = self._get_config_batch_pmapd(samples, numBatches * batchSize, remainder)
            batch = global_defs.pmap_for_my_devices(expand_batch, static_broadcasted_argnums=(1,))(batch, batchSize)
            logPsiSbatch = self._get_logPsi_batch_pmapd(logPsiS, numBatches * batchSize, numSamples % batchSize)
            logPsiSbatch = global_defs.pmap_for_my_devices(expand_batch, static_broadcasted_argnums=(1,))(logPsiSbatch, batchSize)

            # internalize the new batch
            sp, _ = self.get_s_primes(batch, *args)
            # get log chi batched
            logChiSbatch = chi(sp)#self._flatten_pmapd(sp))
            ##############
            # regular infideltiy
            # modified the get_O_loc_unbatched function
            OlocBatch_set = self.get_O_loc_unbatched(logPsiSbatch,logChiSbatch,Observables)
        
            OlocBatch_set = [self._get_Oloc_slice_pmapd(OlocBatch, 0, remainder) for OlocBatch in OlocBatch_set]
            # get the second part of the infideltiy
            Oloc_set = [self._insert_Oloc_batch_pmapd(Oloc, OlocBatch, b * batchSize) for Oloc, OlocBatch in zip(Oloc_set, OlocBatch_set)]
                
        # sretrn the local quantity
        return Oloc_set
    
    def _get_CVc(self,psi_p):
        """ 
        We compute the control variates factor :math:`c` for the local infidelity :math:`F^\\chi_{\\rm loc}(s)`, using 
        the square fidelity :math:`|\mathcal{F}|^2`
        :math:` c = -\frac{\text{cov}\big[ \Re \mathcal{F},~|\mathcal{F}|^2 \big]}{\text{var}|\mathcal{F}|^2}`
        This reads written out
        :math:` 
        \text{cov}\big[ \Re \mathcal{F},~|\mathcal{F}|^2 \big] = \\
        \mathbb{E}_\psi 
        \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}} 
        \Bigg| \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}}\Bigg|^2
        \mathbb{E}_\chi 
        \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}} 
        \Bigg| \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}}\Bigg|^2
        -
        \mathbb{E}_\psi 
        \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}} 
        \mathbb{E}_\psi 
        \Bigg| \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}}\Bigg|^2
        \mathbb{E}_\chi 
        \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}} 
        \mathbb{E}_\chi 
        \Bigg| \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}}\Bigg|^2 \\
        \rm var |\mathcal{F}|^2
        =  
        \mathbb{E}_\psi 
        \Bigg| \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}}\Bigg|^4
        \mathbb{E}_\chi 
        \Bigg| \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}}\Bigg|^4
        -
        \Bigg(
        \mathbb{E}_\psi 
        \Bigg| \frac{\bra{\sigma}\ket{\chi}}{\bra{\sigma}\ket{\psi_\theta}}\Bigg|^2
        \mathbb{E}_\chi 
        \Bigg| \frac{\bra{\eta}\ket{\psi_\theta}}{\bra{\eta}\ket{\chi}}\Bigg|^2
        \Bigg)^2`
        """
        # compute the control variates
        # covariance
        covarFF2 = mpi.global_mean(self.chi_Floc2FlocCV,self.chi_p) * mpi.global_mean(self.psi_Floc2FlocCV,psi_p)
        Exp_psi_FlocCV = mpi.global_mean(self.psi_FlocCV,psi_p)
        covarFF2 -= self.Exp_chi_FlocCV * self.Exp_chi_Floc * Exp_psi_FlocCV * mpi.global_mean(self.psi_Floc,psi_p)
        # variance
        varF2 = mpi.global_mean(self.chi_F2locCV,self.chi_p) * mpi.global_mean(self.psi_F2locCV,psi_p)
        varF2 -= (self.Exp_chi_FlocCV * Exp_psi_FlocCV)**2
        self._varF2 = varF2
        # combine
        self.CVc = self.rmean.update( - abs(covarFF2.real / varF2.real))
        # there is no output
        return None


# auxillary function
def expand_batch(batch, batchSize):
    outShape = list(batch.shape)
    outShape[0] = batchSize
    outp = jnp.zeros(tuple(outShape), dtype=batch.dtype)
    return outp.at[:batch.shape[0]].set(batch)
