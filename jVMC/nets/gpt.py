"""Simple GPT model for autoregressive encoding of quantum states."""

from functools import partial
from typing import Tuple

from flax.linen import (
    Dense,
    Embed,
    LayerNorm,
    Module,
    MultiHeadDotProductAttention,
    Sequential,
    compact,
    gelu,
    log_softmax,
    make_causal_mask,
    scan,
)
from jax import Array, vmap, debug, jit
from jax.config import config  # type: ignore
from jax.numpy import arange, expand_dims, full, int64, take_along_axis, zeros, roll, log, ones, pi, sin, log
from jax.nn import elu
from jax.random import KeyArray, categorical, split
from jVMC.global_defs import tReal

config.update("jax_enable_x64", True)


class _TransformerBlock(Module):
    """The transformer decoder block."""

    embeddingDim: int
    nHeads: int
    paramDType: type = tReal

    @compact
    def __call__(self, x: Array) -> Array:
        x = x + MultiHeadDotProductAttention(
            self.nHeads, param_dtype=self.paramDType
        )(
            x,
            x,
            mask=make_causal_mask(
                zeros((x.shape[-2]), self.paramDType), dtype=self.paramDType
            ),
        )
        x = LayerNorm(param_dtype=self.paramDType)(x)
        x = x + Sequential(
            [   # the factor 4 is a commen choice of ratio between
                # the hidden layer and the embedding layer
                Dense(self.embeddingDim * 4, param_dtype=self.paramDType),
                gelu,
                Dense(self.embeddingDim, param_dtype=self.paramDType),
            ]
        )(x)
        x = LayerNorm(param_dtype=self.paramDType)(x)
        return x


class ComplexGPT(Module):
    """GPT model for autoregressive decoding of neural quantum states.

    This model outputs the log amplitude of a wave function which in turn is
    a log probability density. It contains a ``sample`` method that peforms
    autorgressive sampling.

    Initialization arguments:
        * ``L``: Length of the spin chain.
        * ``embeddingDim``: Embedding dimension.
        * ``depth``: Number of transformer blocks.
        * ``nHeads``: Number of attention heads.
        * ``logProbFactor``: Factor defining how output and associated sample
                probability are related. 0.5 for pure states and 1.0 for POVMs
                (default: 0.5).
        * ``paramDType``: Data type of the model parameters
                (default: ``jVMC.global_defs.tReal``).
        * ``spinDType``: Data type of the spin configurations
                (default: ``jax.numpy.int64``).
    """

    L: int
    embeddingDim: int
    depth: int
    nHeads: int
    logProbFactor: float = 0.5
    paramDType: type = tReal
    spinDType: type = int64
    # jbr
    localHilDim: int = 2

    @compact
    def __call__(self, s: Array, returnLogAmp: bool = True) -> Array:
        """Forward pass of the model.

        Args:
            * ``s``: A spin configuration.
            * ``returnLogAmp``: Whether to return the log amplitude of the spin
                configuration (default: True).

        Returns:
            The log amplitude of the wave function and the complex phase.
        """

        if not self.embeddingDim % self.nHeads == 0:
            raise AttributeError(
                "The embedding dimension should be divisible by the number of"
                " heads."
            )
        if not s.shape[-1] == self.L:
            raise ValueError(
                "Input length should be equal to context length, L."
            )
        # debug.print("{x}",x=s.shape)
        y = Embed(self.localHilDim, self.embeddingDim, param_dtype=self.paramDType)(s) # [:-1])
        p = self.variable(
            "params",
            "positional_embeddings",
            zeros, # jbr: this is the positional embedding, but it is all zeros
            # (self.L-1, self.embeddingDim),
            (self.L, self.embeddingDim),
            self.paramDType,
        ).value
        # jbr: adding the positional encoding
        y = y + p # jbr: this makes no sense, p is all zeoros
        y = Sequential(
            [
                _TransformerBlock(
                    self.embeddingDim, self.nHeads, self.paramDType
                )
                for _ in range(self.depth)
            ]
        )(y)
        # splice off the last element of the sequence and use it as the phase
        phase = y[-1]
        phase_act = lambda x: 0.9 * sin(2.*pi*x) + 2.*pi*x
        # continue with calulating the log amplitude
        y = Dense(self.localHilDim, param_dtype=self.paramDType)(y[:-1])
        # normalizing the wave function
        y = log_softmax(y) * self.logProbFactor
        # return for wave function
        if returnLogAmp:
            return (
                (take_along_axis(y, expand_dims(roll(s,-1)[:-1], -1), axis=-1)
                .sum(axis=-2)
                .squeeze(-1)-log(2.))
                # adding the phase
            ) + 1.j * ( Dense(1,param_dtype=self.paramDType)(phase) ).squeeze(-1)
        # returns for sampling
        return y

    def sample(self, numSamples: int, key: KeyArray) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """

        def generate_sample(key):
            # we only need 
            keys = split(key, self.L)
            # jax.numpy.full(shape, fill_value, dtype=None, *, device=None)[source]
            s = full((self.L,), -1, self.spinDType)
            # flip 50/50 coin for first spin
            # TODO make this learningable
            choice = categorical(keys[0], 0.5*ones(2))
            # setting the zero choice
            s = s.at[0].set(choice)
            # had to modify because of Jax version?
            s, _ = self._scanning_fn(s, (keys[1:], arange(1,self.L)))
            return s

        keys = split(key, numSamples)
        return vmap(generate_sample)(keys)

    @partial(scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, s: Array, x: Tuple[KeyArray, Array]) -> Tuple[Array, None]:
        logits = self(s, False)
        choice = categorical(x[0], logits[x[1]])
        return s.at[x[1]].set(choice), None


class GPT(Module):
    """GPT model for autoregressive decoding of neural quantum states.

    This model outputs the log amplitude of a wave function which in turn is
    a log probability density. It contains a ``sample`` method that peforms
    autorgressive sampling.

    Initialization arguments:
        * ``L``: Length of the spin chain.
        * ``embeddingDim``: Embedding dimension.
        * ``depth``: Number of transformer blocks.
        * ``nHeads``: Number of attention heads.
        * ``logProbFactor``: Factor defining how output and associated sample
                probability are related. 0.5 for pure states and 1.0 for POVMs
                (default: 0.5).
        * ``paramDType``: Data type of the model parameters
                (default: ``jVMC.global_defs.tReal``).
        * ``spinDType``: Data type of the spin configurations
                (default: ``jax.numpy.int64``).
    """

    L: int
    embeddingDim: int
    depth: int
    nHeads: int
    OutputScale: float = 1.
    logProbFactor: float = 0.5
    paramDType: type = tReal
    spinDType: type = int64
    # jbr
    localHilDim: int = 2

    @compact
    def __call__(self, s: Array, returnLogAmp: bool = True) -> Array:
        """Forward pass of the model.

        Args:
            * ``s``: A spin configuration.
            * ``returnLogAmp``: Whether to return the log amplitude of the spin
                configuration (default: True).

        Returns:
            The log amplitude of the wave function.
        """

        # debug.print("{x}",x=s)

        if not self.embeddingDim % self.nHeads == 0:
            raise AttributeError(
                "The embedding dimension should be divisible by the number of"
                " heads."
            )
        if not s.shape[-1] == self.L:
            raise ValueError(
                "Input length should be equal to context length, L."
            )
        # debug.print("{x}",x=s.shape)
        y = Embed(self.localHilDim, self.embeddingDim, param_dtype=self.paramDType)(s[:-1])
        p = self.variable(
            "params",
            "positional_embeddings",
            zeros, # jbr: this is the positional embedding, but it is all zeros
            (self.L-1, self.embeddingDim),
            self.paramDType,
        ).value
        # jbr: adding the positional encoding
        y = y + p # jbr: this makes no sense, p is all zeoros
        y = Sequential(
            [
                _TransformerBlock(
                    self.embeddingDim, self.nHeads, self.paramDType
                )
                for _ in range(self.depth)
            ]
        )(y)
        y = Dense(2, param_dtype=self.paramDType)(y)
        # normalize the wave function
        y = log_softmax(y) * self.logProbFactor
        if returnLogAmp:
            return self.OutputScale * (
                (take_along_axis(y, expand_dims(roll(s,-1)[:-1], -1), axis=-1)
                .sum(axis=-2)
                .squeeze(-1)-log(self.localHilDim))
            )
        return y

    def sample(self, numSamples: int, key: KeyArray) -> Array:
        """Autoregressively sample a spin configuration.

        Args:
            * ``numSamples``: The number of configurations to generate.
            * ``key``: JAX random key.

        Returns:
            A batch of spin configurations.
        """

        def generate_sample(key):
            # we only need 
            keys = split(key, self.L)
            # jax.numpy.full(shape, fill_value, dtype=None, *, device=None)[source]
            s = full((self.L,), -1, self.spinDType)
            # flip 50/50 coin for first spin
            choice = categorical(keys[0], 0.5*ones(2))
            # setting the zero choice
            s = s.at[0].set(choice)
            # had to modify because of Jax version?
            s, _ = self._scanning_fn(s, (keys[1:], arange(1,self.L)))
            return s

        keys = split(key, numSamples)
        return vmap(generate_sample)(keys)

    @partial(scan,
             variable_broadcast='params',
             split_rngs={'params': False})
    def _scanning_fn(self, s: Array, x: Tuple[KeyArray, Array]) -> Tuple[Array, None]:
        logits = self(s, False)
        choice = categorical(x[0], logits[x[1]].real)
        return s.at[x[1]].set(choice), None
