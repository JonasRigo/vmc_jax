import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import unittest

import jax
from jax.config import config
config.update("jax_enable_x64", True)
import jax.random as random
import flax.nn as nn
import jax.numpy as jnp

import numpy as np

import jVMC
import jVMC.nets as nets
from jVMC.vqs import NQS
import jVMC.sampler as sampler

import jVMC.global_defs as global_defs

import time

def state_to_int(s):

    def for_fun(i,xs):
        return (xs[0] + xs[1][i]*(2**i), xs[1])

    x,_=jax.lax.fori_loop(0,s.shape[-1], for_fun, (0, s))

    return x


class TestMCMC(unittest.TestCase):

    def test_MCMC_sampling(self):
        L=4
        
        weights=jnp.array(
                [ 0.23898957,  0.12614753,  0.19479055,  0.17325271,  0.14619853,  0.21392751,
                  0.19648707,  0.17103704, -0.15457255,  0.10954413,  0.13228065, -0.14935214,
                 -0.09963073,  0.17610707,  0.13386381, -0.14836467]
                )

        # Set up variational wave function
        rbm = nets.CpxRBM(numHidden=2,bias=False)
        params = rbm.init(random.PRNGKey(0),jnp.zeros((L,),dtype=np.int32))
        psi = NQS(rbm,params)
        psi.set_parameters(weights)

        # Set up exact sampler
        exactSampler=sampler.ExactSampler(L)
        
        # Set up MCMC sampler
        mcSampler=sampler.MCMCSampler(random.PRNGKey(0),psi,jVMC.sampler.propose_spin_flip, (L,), numChains=777)

        # Compute exact probabilities
        _, _, pex = exactSampler.sample(psi)

        # Get samples from MCMC sampler
        numSamples=500000
        smc, _, _ = mcSampler.sample(numSamples=numSamples)

        if global_defs.usePmap:
            smc = smc.reshape((smc.shape[0]*smc.shape[1], -1))
        
        self.assertTrue( smc.shape[0] >= numSamples )

        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc,_=np.histogram(smcInt, bins=np.arange(0,17))
        
        # Compare histogram to exact probabilities
        self.assertTrue( jnp.max( jnp.abs( pmc/mcSampler.get_last_number_of_samples() - pex.reshape((-1,))[:16] ) ) < 2e-3 )
    

    def test_autoregressive_sampling(self):

        L=4

        # Set up variational wave function
        rnn = nets.RNN( L=4, hiddenSize=5, depth=2 )
        rnnParams = rnn.init( random.PRNGKey(0), jnp.zeros((L,), dtype=np.int32) )
        rbm = nets.RBM(numHidden=2,bias=False)
        rbmParams = rbm.init(random.PRNGKey(0),jnp.zeros((L,), dtype=np.int32))
        
        psi = NQS((rnn, rbm),(rnnParams,rbmParams))
       
        ps=psi.get_parameters()
        psi.update_parameters(ps)
        # Set up exact sampler
        exactSampler=sampler.ExactSampler(L)
        
        # Set up MCMC sampler
        mcSampler=sampler.MCMCSampler(random.PRNGKey(0),psi,jVMC.sampler.propose_spin_flip, (L,), numChains=777)
        
        # Compute exact probabilities
        _, _, pex = exactSampler.sample(psi)

        numSamples=500000
        smc,p,_=mcSampler.sample(numSamples=numSamples)

        self.assertTrue( jnp.max( jnp.abs( jnp.real(psi(smc)-p)) ) < 1e-12 )
    
        if global_defs.usePmap:
            smc = smc.reshape((smc.shape[0]*smc.shape[1], -1))
       
        self.assertTrue( smc.shape[0] >= numSamples )
        
        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc,_=np.histogram(smcInt, bins=np.arange(0,17))

        self.assertTrue( jnp.max( jnp.abs( pmc/mcSampler.get_last_number_of_samples()-pex.reshape((-1,))[:16] ) ) < 1.1e-3 )
    

    def test_autoregressive_sampling_with_symmetries(self):

        L=4

        # Set up symmetry orbit
        orbit=jnp.array([jnp.roll(jnp.identity(L,dtype=np.int32), l, axis=1) for l in range(L)])

        # Set up variational wave function
        rnn = nets.RNNsym( L=L, hiddenSize=5, orbit=orbit )
        rnnParams = rnn.init( random.PRNGKey(0), jnp.zeros((L,),dtype=np.int32) )
        rbm = nets.RBM(numHidden=2,bias=False)
        rbmParams = rbm.init(random.PRNGKey(0),jnp.zeros((L,),dtype=np.int32))
        
        psi = NQS((rnn, rbm),(rnnParams,rbmParams))
       
        # Set up exact sampler
        exactSampler=sampler.ExactSampler(L)
        
        # Set up MCMC sampler
        mcSampler=sampler.MCMCSampler(random.PRNGKey(0),psi,jVMC.sampler.propose_spin_flip, (L,), numChains=777)
        
        # Compute exact probabilities
        _, logPsi, pex = exactSampler.sample(psi)

        numSamples=1000000
        smc,p,_=mcSampler.sample(numSamples=numSamples)

        self.assertTrue( jnp.max( jnp.abs( jnp.real(psi(smc)-p)) ) < 1e-12 )
    
        if global_defs.usePmap:
            smc = smc.reshape((smc.shape[0]*smc.shape[1], -1))
       
        self.assertTrue( smc.shape[0] >= numSamples )
        
        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc,_=np.histogram(smcInt, bins=np.arange(0,17))

        self.assertTrue( jnp.max( jnp.abs( pmc/mcSampler.get_last_number_of_samples()-pex.reshape((-1,))[:16] ) ) < 1e-3 )
    

    def test_autoregressive_sampling_with_lstm(self):

        L=4

        # Set up symmetry orbit
        orbit=jnp.array([jnp.roll(jnp.identity(L,dtype=np.int32), l, axis=1) for l in range(L)])

        # Set up variational wave function
        rnn = nets.LSTM( L=L, hiddenSize=5 )
        rnnParams = rnn.init( random.PRNGKey(0), jnp.zeros((L,),dtype=np.int32) )
        rbm = nets.RBM(numHidden=2,bias=False)
        rbmParams = rbm.init(random.PRNGKey(0),jnp.zeros((L,),dtype=np.int32))
        
        psi = NQS((rnn, rbm),(rnnParams,rbmParams))
       
        # Set up exact sampler
        exactSampler=sampler.ExactSampler(L)
        
        # Set up MCMC sampler
        mcSampler=sampler.MCMCSampler(random.PRNGKey(0),psi,jVMC.sampler.propose_spin_flip, (L,), numChains=777)
        
        # Compute exact probabilities
        _, logPsi, pex = exactSampler.sample(psi)

        numSamples=1000000
        smc,p,_=mcSampler.sample(numSamples=numSamples)

        self.assertTrue( jnp.max( jnp.abs( jnp.real(psi(smc)-p)) ) < 1e-12 )
    
        if global_defs.usePmap:
            smc = smc.reshape((smc.shape[0]*smc.shape[1], -1))
       
        self.assertTrue( smc.shape[0] >= numSamples )
        
        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc,_=np.histogram(smcInt, bins=np.arange(0,17))

        self.assertTrue( jnp.max( jnp.abs( pmc/mcSampler.get_last_number_of_samples()-pex.reshape((-1,))[:16] ) ) < 1e-3 )
    

    def test_autoregressive_sampling_with_rnn2d(self):

        L=2

        # Set up variational wave function
        rnn = nets.RNN2D( L=L, hiddenSize=5 )
        params = rnn.init( random.PRNGKey(0), jnp.zeros((L,L),dtype=np.int32) )

        psi = NQS((rnn,rnn),(params,params))
        
        # Set up exact sampler
        exactSampler=sampler.ExactSampler((L,L))
        
        # Set up MCMC sampler
        mcSampler=sampler.MCMCSampler(random.PRNGKey(0),psi,jVMC.sampler.propose_spin_flip, (L,), numChains=777)
        
        # Compute exact probabilities
        _, logPsi, pex = exactSampler.sample(psi)

        self.assertTrue(jnp.abs(jnp.sum(pex)-1.) < 1e-12)

        numSamples=1000000
        smc,p,_=mcSampler.sample(numSamples=numSamples)

        self.assertTrue( jnp.max( jnp.abs( jnp.real(psi(smc)-p)) ) < 1e-12 )
    
        if global_defs.usePmap:
            smc = smc.reshape((smc.shape[0]*smc.shape[1], -1))
       
        self.assertTrue( smc.shape[0] >= numSamples )
        
        # Compute histogram of sampled configurations
        smcInt = jax.vmap(state_to_int)(smc)
        pmc,_=np.histogram(smcInt, bins=np.arange(0,17))

        self.assertTrue( jnp.max( jnp.abs( pmc/mcSampler.get_last_number_of_samples()-pex.reshape((-1,))[:16] ) ) < 1e-3 )

if __name__ == "__main__":
    unittest.main()
