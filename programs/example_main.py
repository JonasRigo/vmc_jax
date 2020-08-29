import sys
# Find jVMC package
sys.path.append(sys.path[0]+"/..")

import jax
from jax.config import config
config.update("jax_enable_x64", True)

import jax.random as random
import flax
import flax.nn as nn
import jax.numpy as jnp

import numpy as np

import time

import jVMC
import jVMC.operator as op
from jVMC.util import measure, ground_state_search, OutputManager
import jVMC.mpi_wrapper as mpi

from functools import partial

inp = None
if len(sys.argv) > 1:
    # if an input file is given
    with open(sys.argv[1],'r') as f:
        inp = json.load(f)
else:
    # otherwise, set up default input dict
    inp = {}
    inp["general"] = {
        "data_output" : "data.hdf5"
    }
    inp["system"] = {
        "L" : 4,
        "J0" : -1.0,
        "hx0" : -2.5,
        "J" : -1.0,
        "hx" : -0.3
    }

    inp["sampler"] = {
        "type" : "MC",
        "numSamples" : 1000,
        "numChains" : 30
    }

L = inp["system"]["L"]

# Initialize output manager
outp = OutputManager(inp["general"]["data_output"])

# Set up variational wave function
rbm = jVMC.nets.CpxRBM.partial(numHidden=2,bias=False)
_, params = rbm.init_by_shape(random.PRNGKey(0),[(1,inp["system"]["L"])])
rbmModel = nn.Model(rbm,params)

rbm1 = jVMC.nets.RBM.partial(numHidden=6,bias=False)
_, params1 = rbm1.init_by_shape(random.PRNGKey(123),[(1,inp["system"]["L"])])
rbmModel1 = nn.Model(rbm1,params1)
#rbm2 = jVMC.nets.FFN.partial(layers=[5,5],bias=False)
rbm2 = jVMC.nets.RBM.partial(numHidden=6,bias=False)
_, params2 = rbm2.init_by_shape(random.PRNGKey(321),[(1,inp["system"]["L"])])
rbmModel2 = nn.Model(rbm2,params2)

#psi = jVMC.vqs.NQS(rbmModel)
psi = jVMC.vqs.NQS(rbmModel1, rbmModel2)

# Set up hamiltonian for ground state search
hamiltonianGS = op.Operator()
for l in range(L):
    hamiltonianGS.add( op.scal_opstr( inp["system"]["J0"], ( op.Sz(l), op.Sz((l+1)%L) ) ) )
    hamiltonianGS.add( op.scal_opstr( inp["system"]["hx0"], ( op.Sx(l), ) ) )

# Set up hamiltonian
hamiltonian = op.Operator()
for l in range(L):
    hamiltonian.add( op.scal_opstr( inp["system"]["J"], ( op.Sz(l), op.Sz((l+1)%L) ) ) )
    hamiltonian.add( op.scal_opstr( inp["system"]["hx"], ( op.Sx(l), ) ) )

# Set up observables
observables = [hamiltonianGS, op.Operator(), op.Operator(), op.Operator()]
for l in range(L):
    observables[1].add( ( op.Sx(l), ) )
    observables[2].add( ( op.Sz(l), op.Sz((l+1)%L) ) )
    observables[3].add( ( op.Sz(l), op.Sz((l+2)%L) ) )

sampler = None
if inp["sampler"]["type"] == "MC":
    # Set up MCMC sampler
    sampler = jVMC.sampler.MCMCSampler(random.PRNGKey(123), jVMC.sampler.propose_spin_flip, [L], numChains=inp["sampler"]["numChains"], numSamples=inp["sampler"]["numSamples"])
else:
    # Set up exact sampler
    sampler=jVMC.sampler.ExactSampler(L)

#tdvpEquation = jVMC.tdvp.TDVP(mcSampler, snrTol=1)
delta=5
tdvpEquation = jVMC.tdvp.TDVP(sampler, snrTol=1, svdTol=1e-8, rhsPrefactor=1., diagonalShift=delta, makeReal='real')

# Perform ground state search to get initial state
outp.print("** Ground state search")
outp.set_group("ground_state_search")

ground_state_search(psi, hamiltonianGS, tdvpEquation, sampler, numSteps=10, stepSize=1e-2, observables=observables, outp=outp)

# Time evolution
outp.print("** Time evolution")
outp.set_group("time_evolution")

observables[0] = hamiltonian
tdvpEquation = jVMC.tdvp.TDVP(sampler, snrTol=1, svdTol=1e-6, rhsPrefactor=1.j, diagonalShift=0., makeReal='imag')

stepper = jVMC.stepper.AdaptiveHeun(timeStep=1e-3, tol=1e-2)

t=0
tmax=1
outp.start_timing("measure observables")
obs, err = measure(observables, psi, sampler)
outp.stop_timing("measure observables")

outp.write_observables(t, energy=obs[0], X=obs[1]/L, ZZ=obs[2:]/L)

while t<tmax:
    tic = time.perf_counter()
    outp.print( ">  t = %f\n" % (t) )

    # TDVP step
    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=hamiltonian, psi=psi, numSamples=inp["sampler"]["numSamples"], outp=outp)
    psi.set_parameters(dp)
    t += dt
    outp.print( "   Time step size: dt = %f" % (dt) )

    # Measure observables
    outp.start_timing("measure observables")
    obs, err = measure(observables, psi, sampler)
    outp.stop_timing("measure observables")

    # Write observables
    outp.write_observables(t, energy=obs[0], X=obs[1]/L, ZZ=obs[2:]/L)

    outp.print("    Energy = %f +/- %f" % (obs[0], err[0]))

    outp.print_timings(indent="   ")

    toc = time.perf_counter()
    outp.print( "   == Total time for this step: %fs\n" % (toc-tic) )
