import jax
import jax.numpy as jnp
import jax.random as jrnd
from jax.nn import log_softmax
from functools import partial
#import jVMC.global_defs as global_defs
from flax import linen as nn
from typing import Any, List, Optional, Tuple
from jax import Array, vmap, jit


@jit
def sorting_gumble(sample,logits,gumbel,states):

    numSamples = sample.shape[0]
    LocalHilDim = sample.shape[1]
    L = sample.shape[2]

    indexes = jnp.argsort((-gumbel),axis=None) # only take the most likely probabilities --> cutoff after  numSamples
    indexes_states = (indexes // LocalHilDim)[:numSamples]
    sample = sample.reshape(-1,L)[indexes]
    sample = sample.reshape(LocalHilDim,numSamples,L)
    sample = jnp.swapaxes(sample,0,1) # some shenanigans to select the samples correctly 
    
    logits = logits.ravel()[indexes]
    logits = logits.reshape(LocalHilDim,numSamples).T
    
    gumbel = gumbel.ravel()[indexes]
    gumbel = gumbel.reshape(LocalHilDim,numSamples).T
    vals, treedef  = jax.tree_util.tree_flatten(states)

    vals_ordered = [v[indexes_states] for v in vals] # more shenanigans to select the right states for the network...
    states = jax.tree_util.tree_unflatten(treedef,vals_ordered)
    
    return sample,logits,gumbel,states

class gumbel_wrapper(nn.Module):
    """
    Wrapper module for symmetrization.
    This is a wrapper module for the incorporation of gumbel MC sampling (sampling without repetition). 
    
    see https://arxiv.org/pdf/2408.07625v1     
        
    """
    
    net: callable
    is_gumbel = True
    #avgFun: callable = avgFun_Coefficients_Exp
    def setup(self):
        self.L = self.net.L
        self.sequence_length = self.net.PL if hasattr(self.net, "PL") else self.net.L
        self.LocalHilDim = 2 # default spin with local dim equal 2
        if hasattr(self.net, "LocalHilDim"):
            self.LocalHilDim = self.net.LocalHilDim ## == lDim ** patch_size

        if not "sample" in dir(self.net):
            raise NotImplemented('Gumbel requires autoregressive network')

    def __post_init__(self):

        super().__post_init__()
    
    def __call__(self,*args,**kwargs):
        
        return self.net(*args,**kwargs)
        
    def _apply_fun(self, *args,**kwargs):
        return self.net.apply(*args,**kwargs)
    
    def _decode_samples(self, samples):
        if not hasattr(self.net, "patch_states"):
            return samples
        decoded = self.net.patch_states[samples]
        return decoded.reshape(samples.shape[:-1] + (-1,))

    def _gumbel_step(self,sample,logits,gumbel,key,states,cumsum,position):   
        #new samples with (0,..,LocalHilDim-1) at position
        #sample = jnp.array([sample[0].at[position].set(l) for l in jnp.arange(self.LocalHilDim)])
        #right shifted input
        logitnew = jnp.zeros_like(logits)
        sample = jnp.array([sample[0].at[position].set(l) for l in jnp.arange(self.LocalHilDim)])
        #right shifted input
        inputt = jnp.array([jnp.pad(sample[0,:-1],(1,0))])

        #get next conditional probabilities
        call_kwargs = {"block_states": states, "output_state": True}
        if hasattr(self.net, "is_particle") and self.net.is_particle:
            call_kwargs["cumsum"] = cumsum
            call_kwargs["position"] = position
        logitnew, next_states = self(inputt[:,position], **call_kwargs)
        # Canonicalize to the same sampling distribution used by the
        # underlying autoregressive net. RWKV, for example, samples from
        # real(logits) / temperature in RNN mode.
        temperature = getattr(self.net, "temperature", 1.0)
        logitnew = log_softmax(jnp.real(logitnew) / temperature, axis=-1)
        #calculate new updated "total" cond. probabilities
        logitnew = logits[0] + logitnew 
        # gumbel "noise"
        gumbelnew = logitnew + jrnd.gumbel(key[0],shape=(self.LocalHilDim,)) 
        Z = jnp.nanmax(gumbelnew)
        gumbelnew = jnp.nan_to_num(-jnp.log(
            jnp.exp(-gumbel[0])-jnp.exp(-Z)+jnp.exp(-gumbelnew) 
            ),nan=-jnp.inf)

        return sample, logitnew, gumbelnew, next_states
    
    def sample(self, numSamples: int, key) -> Array:
        """Autoregressively sample.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of system configurations.
        """
        
        if numSamples >= (self.LocalHilDim**self.sequence_length):
            raise RuntimeError("number of samples higher than the Hilbert space")
            # Unique samples not possible is the Hilbert space is smaller than the number of samples 
            # This if-condition catches the edge case in order to avoid undefined behavior  
        
        # split the samples
        keys = jrnd.split(key, self.sequence_length)
        ## init step
        shape_samples = (numSamples,self.LocalHilDim,self.sequence_length)
        shape_logits = (numSamples,self.LocalHilDim)
        shape_gumbel = (numSamples,self.LocalHilDim)
        #print(shape_samples,shape_logits)
        working_space_samples = jnp.full(shape_samples,-2,dtype=jnp.int64)
        working_space_logits = jnp.full(shape_logits,-jnp.inf,dtype=jnp.float64)
        working_space_gumbel = jnp.full(shape_gumbel,-jnp.inf,dtype=jnp.float64)
        
        #working_space_samples = working_space_samples.at[0,0,0].set(0)
        working_space_logits = working_space_logits.at[0,0].set(0.)
        working_space_gumbel = working_space_gumbel.at[0,0].set(0.)
        
        states = None
        initial_cumsum = jnp.int64(0)
        init_work = partial(self._gumbel_step, position=0, states=states, cumsum=initial_cumsum)
        key0 = jrnd.split(keys[0],numSamples)
        key0 = jnp.expand_dims(key0,-2)
        samples,logits,gumbel,states  = jax.vmap(init_work)(working_space_samples,working_space_logits,working_space_gumbel,key0)
        
        init_carry = sorting_gumble(samples,logits,gumbel,states)
        first_choices = init_carry[0][:,0,0]
        if hasattr(self.net, "patch_states"):
            cumsum = jax.vmap(lambda t: self.net.patch_states[t].sum())(first_choices)
        else:
            cumsum = first_choices
        init_carry = (*init_carry, cumsum)
        
        res,_ = self._scanning_fn(init_carry,(keys[1:],jnp.arange(1,self.sequence_length)))
        samples, logits,gumbels,_,_ = res
        
        kappa = gumbels[0,1] # first non-chosen configuration used to estimate reweighting

        re_weights = jnp.nan_to_num(jnp.exp(logits[:,0]) /(-jnp.expm1(-jnp.exp(logits[:,0]-kappa))),0)

        samples = self._decode_samples(samples[:,0,:])
        return samples,logits[:,0]*self.net.logProbFactor,re_weights/jnp.sum(re_weights),kappa
        #return samples[:,0,:],logits[:,0],re_weights/jnp.sum(re_weights)

    @partial(nn.scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, carry, key):
        sample = carry[0]
        logits = carry[1]
        gumbel = carry[2]
        states = carry[3]
        cumsum = carry[4]

        position = key[1] # (lattice 0 ... system size)
        keys = jrnd.split(key[0],carry[0].shape[0])
        keys = jnp.expand_dims(keys,-2)

        p_workN = partial(self._gumbel_step,position=position)
        sample,logits,gumbel,states = jax.vmap(p_workN)(sample,logits,gumbel,keys,states,cumsum)

        sorted_carry = sorting_gumble(sample,logits,gumbel,states)
        sample = sorted_carry[0]

        if hasattr(self.net, "is_particle") and self.net.is_particle:
            chosen_path = sample[:, 0, :]
            prefix_mask = (jnp.arange(self.sequence_length) <= position).astype(jnp.int64)
            if hasattr(self.net, "patch_states"):
                token_particles = jnp.sum(self.net.patch_states[chosen_path], axis=-1)
            else:
                token_particles = chosen_path.astype(jnp.int64)
            cumsum = jnp.sum(token_particles * prefix_mask[None, :], axis=-1)
        else:
            chosen = sample[:,0,position]
            if hasattr(self.net, "patch_states"):
                cumsum = cumsum + jax.vmap(lambda t: self.net.patch_states[t].sum())(chosen)
            else:
                cumsum = cumsum + chosen
        return (*sorted_carry, cumsum),None
        