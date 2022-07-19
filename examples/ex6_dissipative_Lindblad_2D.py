import matplotlib.pyplot as plt
import numpy as np
import flax
import jax.random as random
from jax.config import config
import jax.numpy as jnp
import jax
import jVMC
config.update("jax_enable_x64", True)
from functools import partial


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


def xy_to_id(x, y, L):
    return int(x + L * y)


L = 2
inputDim = 4
logProbFactor = 1
dim = "2D"

# Initialize net
sample_shape = (L, L)
psi = jVMC.util.util.init_net({"batch_size": 5000, "net1": {"type": "RNN2D",
                                        "translation": True,
                                        "parameters": {"inputDim": 4, "logProbFactor": 1, "hiddenSize": 5, "L": L, "depth": 2, "cell": "RNN",
                                                       "realValuedOutput": True,
                                                       "realValuedParams": True}}},
                              sample_shape, 1234)
print(f"The variational ansatz has {psi.numParameters} parameters.")

# Set up hamiltonian
system_data = {"dim": dim, "L": L}
povm = jVMC.operator.POVM(system_data)
Lindbladian = jVMC.operator.POVMOperator(povm)
for x in range(L):
    for y in range(L):
        Lindbladian.add({"name": "ZZ", "strength": 1.0, "sites": (xy_to_id(x, y, L), xy_to_id((x + 1) % L, y, L))})
        Lindbladian.add({"name": "ZZ", "strength": 1.0, "sites": (xy_to_id(x, y, L), xy_to_id(x, (y + 1) % L, L))})
        Lindbladian.add({"name": "X", "strength": 3, "sites": (xy_to_id(x, y, L),)})
        Lindbladian.add({"name": "dephasing", "strength": .5, "sites": (xy_to_id(x, y, L),)})

# Set up initial state as product state
prob_dist = jVMC.operator.povm.get_1_particle_distributions("z_up", Lindbladian.povm)
prob_dist /= prob_dist[0]
biases = jnp.log(prob_dist[1:])
params = copy_dict(psi._param_unflatten(psi.get_parameters()))

params["outputDense"]["bias"] = biases
params["outputDense"]["kernel"] = 1e-15 * params["outputDense"]["kernel"]
params = jnp.concatenate([p.ravel()
                          for p in jax.tree_util.tree_flatten(params)[0]])
psi.set_parameters(params)

# Set up sampler
sampler = jVMC.sampler.ExactSampler(psi, sample_shape, lDim=inputDim, logProbFactor=logProbFactor)
# sampler = jVMC.sampler.MCSampler(psi, sample_shape, random.PRNGKey(123), updateProposer=jVMC.sampler.propose_POVM_outcome, numSamples=1000)


# Set up TDVP
tdvpEquation = jVMC.util.tdvp.TDVP(sampler, rhsPrefactor=-1.,
                                   svdTol=1e-6, diagonalShift=0, makeReal='real')

stepper = jVMC.util.stepper.AdaptiveHeun(timeStep=1e-3, tol=1e-2)  # ODE integrator

res = {"X": [], "Y": [], "Z": [], "X_corr_L1": [],
       "Y_corr_L1": [], "Z_corr_L1": []}

times = []
t = 0

while t < 5 * 1e-0:
    times.append(t)
    result = jVMC.operator.povm.measure_povm(Lindbladian.povm, sampler)
    for dim in ["X", "Y", "Z"]:
        res[dim].append(result[dim]["mean"])
        res[dim + "_corr_L1"].append(result[dim + "_corr_L1"]["mean"])

    dp, dt = stepper.step(0, tdvpEquation, psi.get_parameters(), hamiltonian=Lindbladian, psi=psi, normFunction=partial(norm_fun, df=tdvpEquation.S_dot))
    t += dt
    psi.set_parameters(dp)
    print(f"t = {t:.3f}, \t dt = {dt:.2e}")


plt.plot(times, res["X"], label=r"$\langle X \rangle$")
plt.plot(times, res["Y"], label=r"$\langle Y \rangle$")
plt.plot(times, res["Z"], label=r"$\langle Z \rangle$")
plt.plot(times, res["Z_corr_L1"], label=r"$\langle Z_iZ_{i+1} \rangle$", linestyle="--")
plt.xlabel(r"$Jt$")
plt.legend()
plt.grid()
plt.savefig('Lindblad_evolution.pdf')
plt.show()
