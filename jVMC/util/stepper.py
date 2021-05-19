import jax
import jax.numpy as jnp
import numpy as np


class Euler:

    def __init__(self, timeStep=1e-3):

        self.dt = timeStep

    def step(self, t, f, yInitial, **rhsArgs):

        dy = f(yInitial, t, **rhsArgs, intStep=0)

        return yInitial + self.dt * dy, self.dt

# end class Euler


class AdaptiveHeun:

    def __init__(self, timeStep=1e-3, tol=1e-8, maxStep=1):
        self.dt = timeStep
        self.tolerance = tol
        self.maxStep = maxStep

    def step(self, t, f, yInitial, normFunction=jnp.linalg.norm, **rhsArgs):

        fe = 0.5

        dt = self.dt

        while fe < 1.:

            y = yInitial.copy()
            k0 = f(y, t, **rhsArgs, intStep=0)
            y += dt * k0
            k1 = f(y, t + dt, **rhsArgs, intStep=1)
            dy0 = 0.5 * dt * (k0 + k1)

            # now with half step size
            y -= 0.5 * dt * k0
            k10 = f(y, t + 0.5 * dt, **rhsArgs, intStep=2)
            dy1 = 0.25 * dt * (k0 + k10)
            y = yInitial + dy1
            k01 = f(y, t + 0.5 * dt, **rhsArgs, intStep=3)
            y += 0.5 * dt * k01
            k11 = f(y, t + dt, **rhsArgs, intStep=4)
            dy1 += 0.25 * dt * (k01 + k11)

            # compute deviation
            updateDiff = normFunction(dy1 - dy0)
            fe = self.tolerance / updateDiff

            if 0.2 > 0.9 * fe**0.33333:
                tmp = 0.2
            else:
                tmp = 0.9 * fe**0.33333
            if tmp > 2.:
                tmp = 2.

            realDt = dt
            dt *= tmp

            if dt > self.maxStep:
                dt = self.maxStep

        # end while

        self.dt = dt

        return yInitial + dy1, realDt
