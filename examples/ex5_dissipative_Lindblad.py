import matplotlib.pyplot as plt
import numpy as np
import flax
import jax.random as random
from jax.config import config
import jax.numpy as jnp
import jax
import sys
sys.path.append(sys.path[0] + "/..")
# Find jVMC package
import jVMC
from functools import partial
config.update("jax_enable_x64", True)


def copy_dict(a):
    b = {}
    for key, value in a.items():
        if type(value) == type(a):
            b[key] = copy_dict(value)
        else:
            b[key] = value
    return b


def norm_fun(v, df=lambda x: x):
    return jnp.real(jnp.conj(jnp.transpose(v)).dot(df(v)))


L = 5
dim = "1D"
logProbFactor = 1

# Initialize net
orbit = jVMC.util.get_1d_orbit(L)
net = jVMC.nets.RNNsym(inputDim=4, logProbFactor=logProbFactor, hiddenSize=6, L=L, depth=2, orbit=orbit)
params = net.init(jax.random.PRNGKey(1234), jnp.zeros((L,), dtype=jnp.int32))
psi = jVMC.vqs.NQS(net, params)  # Variational wave function
print(f"The variational ansatz has {psi.numParameters} parameters.")

# Set up hamiltonian
system_data = {"dim": dim, "L": L}
povm = jVMC.operator.POVM(system_data)
Lindbladian = jVMC.operator.POVMOperator(povm)
for l in range(L):
    Lindbladian.add({"name": "ZZ", "strength": 1.0, "sites": (l, (l + 1) % L)})
    Lindbladian.add({"name": "X", "strength": 3.0, "sites": (l,)})
    Lindbladian.add({"name": "dephasing", "strength": 1.0, "sites": (l,)})

prob_dist = jVMC.operator.povm.get_1_particle_distributions("y_up", Lindbladian.povm)
biases = jnp.log(prob_dist)
params = copy_dict(psi._param_unflatten_cpx(psi.get_parameters()))

params["params"]["rnn"]["outputDense"]["bias"] = biases
params["params"]["rnn"]["outputDense"]["kernel"] = 1e-15 * params["params"]["rnn"]["outputDense"]["kernel"]
params = jnp.concatenate([p.ravel()
                          for p in jax.tree_util.tree_flatten(params)[0]])
psi.set_parameters(params)

# Set up sampler
sampler = jVMC.sampler.ExactSampler(psi, (L,), lDim=4, logProbFactor=logProbFactor)
# sampler = jVMC.sampler.MCMCSampler(random.PRNGKey(123), psi, jVMC.sampler.propose_POVM_outcome, (L,), numSamples=1000)

# Set up TDVP
tdvpEquation = jVMC.tdvp.TDVP(sampler, rhsPrefactor=-1.,
                              svdTol=1e-6, diagonalShift=0, makeReal='real', crossValidation=False)

stepper = jVMC.stepper.AdaptiveHeun(timeStep=1e-3, tol=1e-3)  # ODE integrator

res = {"X": [], "Y": [], "Z": [], "X_corr_L1": [],
       "Y_corr_L1": [], "Z_corr_L1": []}

times = []
t = 0
while t < 5 * 1e-2:
    times.append(t)
    result = jVMC.operator.povm.measure_povm(Lindbladian.povm, sampler, psi)
    for dim in ["X", "Y", "Z"]:
        res[dim].append(result[dim]["mean"])
        res[dim + "_corr_L1"].append(result[dim + "_corr_L1"]["mean"])

    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=Lindbladian, psi=psi, normFunction=partial(norm_fun, df=tdvpEquation.S_dot))
    t += dt
    psi.set_parameters(dp)
    print(f"t = {t:.3f}, \t dt = {dt:.2e}")
    if tdvpEquation.crossValidation:
        print(f"Cross-Validation-Factor_residual = {tdvpEquation.crossValidationFactor_residual:.3f}")
        print(f"Cross-Validation-Factor_tdvpErr = {tdvpEquation.crossValidationFactor_tdvpErr:.3f}")


plt.plot(times, res["X"], label=r"$\langle X \rangle$")
plt.plot(times, res["Y"], label=r"$\langle Y \rangle$")
plt.plot(times, res["Z"], label=r"$\langle Z \rangle$")
plt.plot(times, res["Z_corr_L1"], label=r"$\langle Z_iZ_{i+1} \rangle$", linestyle="--")
plt.xlabel(r"$Jt$")
plt.legend()
plt.grid()
plt.savefig('Lindblad_evolution.pdf')
plt.show()
