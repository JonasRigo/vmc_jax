import jax
import jax.numpy as jnp
import flax.linen as nn
from jVMC.util.symmetries import LatticeSymmetry


def avgFun_Coefficients_Exp(coeffs):
    # average (complex) coefficients
    return jnp.log(jnp.mean(jnp.exp(coeffs)))


def avgFun_Coefficients_Log(coeffs):
    return jnp.mean(coeffs)


def avgFun_Coefficients_Sep(coeffs):
    re = jnp.real(coeffs)
    im = jnp.imag(coeffs)
    return 0.5 * jnp.log(jnp.mean(jnp.exp(2 * re))) + 1j * jnp.angle(jnp.mean(jnp.exp(1j * im)))


class SymNet(nn.Module):
    """
    Wrapper module for symmetrization.
    This is a wrapper module for the incorporation of lattice symmetries. 
    The given plain ansatz :math:`\\psi_\\theta` is symmetrized as

        :math:`\\Psi_\\theta(s)=\\frac{1}{|\\mathcal S|}\\sum_{\\tau\\in\\mathcal S}\\psi_\\theta(\\tau(s))`

    where :math:`\\mathcal S` denotes the set of symmetry operations (``orbit`` in our nomenclature).

    Initialization arguments:
        * ``orbit``: orbits which define the symmetry operations (instance of ``util.symmetries.LatticeSymmetry``)
        * ``net``: Flax module defining the plain ansatz.
        * ``avgFun``: Different choices for the details of averaging.

    """
    orbit: LatticeSymmetry
    net: callable
    avgFun: callable = avgFun_Coefficients_Exp

    @nn.compact
    def __call__(self, x):

        inShape = x.shape
        x = 2 * x - 1
        x = jax.vmap(lambda o, s: jnp.dot(o, s.ravel()).reshape(inShape), in_axes=(0, None))(self.orbit.orbit, x)
        x = (x + 1) // 2

        def evaluate(x):
            return self.net(x)

        res = self.avgFun(jax.vmap(evaluate)(x))

        return res

    def sample(self, *args):
        return self.net.sample(*args)

# ** end class SymNet
