import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import flax.nn as nn
import jax.numpy as jnp

import numpy as np

import jVMC
import jVMC.stepper as jVMCstepper
import jVMC.nets as nets
from jVMC.vqs import NQS
import jVMC.operator as op
import jVMC.sampler as sampler
import jVMC.tdvp as tdvp
from jVMC.util import measure, ground_state_search

from functools import partial

L=4
J0=-1.0
hx0=-1.3
J=-1.0
hx=-1.5

numSamples=500

# Set up variational wave function
rbm = nets.CpxRBM.partial(L=L,numHidden=2,bias=False)
_, params = rbm.init_by_shape(random.PRNGKey(0),[(1,L)])
rbmModel = nn.Model(rbm,params)
psi = NQS(rbmModel)

# Set up hamiltonian for ground state search
hamiltonianGS = op.Operator()
for l in range(L):
    hamiltonianGS.add( op.scal_opstr( J0, ( op.Sz(l), op.Sz((l+1)%L) ) ) )
    hamiltonianGS.add( op.scal_opstr( hx0, ( op.Sx(l), ) ) )

# Set up hamiltonian
hamiltonian = op.Operator()
for l in range(L):
    hamiltonian.add( op.scal_opstr( J, ( op.Sz(l), op.Sz((l+1)%L) ) ) )
    hamiltonian.add( op.scal_opstr( hx, ( op.Sx(l), ) ) )

# Set up observables
observables = [hamiltonianGS, op.Operator(), op.Operator(), op.Operator()]
for l in range(L):
    observables[1].add( ( op.Sx(l), ) )
    observables[2].add( ( op.Sz(l), op.Sz((l+1)%L) ) )
    observables[3].add( ( op.Sz(l), op.Sz((l+2)%L) ) )

# Set up MCMC sampler
mcSampler = sampler.MCMCSampler(random.PRNGKey(123), sampler.propose_spin_flip, [L], numChains=10, numSamples=numSamples)

# Set up exact sampler
exactSampler=sampler.ExactSampler(L)

#tdvpEquation = jVMC.tdvp.TDVP(mcSampler, snrTol=1)
delta=5
tdvpEquation = jVMC.tdvp.TDVP(exactSampler, snrTol=1, svdTol=1e-8, rhsPrefactor=1., diagonalShift=delta, makeReal='real')

# Perform ground state search to get initial state
print("** Ground state search")
ground_state_search(psi, hamiltonianGS, tdvpEquation, exactSampler, numSteps=100, stepSize=1e-2, observables=observables)


print("** Time evolution")

observables[0] = hamiltonianGS
tdvpEquation = jVMC.tdvp.TDVP(exactSampler, snrTol=1, svdTol=1e-8, rhsPrefactor=1.j, diagonalShift=0., makeReal='imag')
stepper = jVMCstepper.Euler(timeStep=1e-3)

t=0
tmax=1
obs = measure(observables, psi, exactSampler)
print("{0:.6f} {1:.6f} {2:.6f} {3:.6f} {4:.6f}".format(t, obs[0], obs[1]/L, obs[2]/L, obs[3]/L))
while t<tmax:
    stepperArgs = {'hamiltonian': hamiltonian, 'psi': psi, 'numSamples': numSamples}
    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), stepperArgs)
    psi.set_parameters(dp)
    t += dt

    obs = measure(observables, psi, exactSampler)
    print("{0:.6f} {1:.6f} {2:.6f} {3:.6f} {4:.6f}".format(t, obs[0], obs[1]/L, obs[2]/L, obs[3]/L))
