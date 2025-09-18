import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import sparse


def generate_factor_list_diagonal(width, height):
    """
    Generate a list of B_factor dependencies where
    each state corresponds to a grid cell in a 2D grid
    with width x height dimensions. Each state has a
    dependency on the previous state of itself and its (max) 8 neighbors.
    """
    n_factors = width * height
    factor_list = []
    for i in range(width):
        for j in range(height):
            factors = []
            factors.append(i * width + j)
            if i > 0:
                factors.append((i - 1) * width + j)
            else:
                factors.append(n_factors)
            if i < width - 1:
                factors.append((i + 1) * width + j)
            else:
                factors.append(n_factors)
            if j > 0:
                factors.append(i * width + j - 1)
            else:
                factors.append(n_factors)
            if j < height - 1:
                factors.append(i * width + j + 1)
            else:
                factors.append(n_factors)
            if i > 0 and j > 0:
                factors.append((i - 1) * width + j - 1)
            else:
                factors.append(n_factors)
            if i > 0 and j < height - 1:
                factors.append((i - 1) * width + j + 1)
            else:
                factors.append(n_factors)
            if i < width - 1 and j > 0:
                factors.append((i + 1) * width + j - 1)
            else:
                factors.append(n_factors)
            if i < width - 1 and j < height - 1:
                factors.append((i + 1) * width + j + 1)
            else:
                factors.append(n_factors)

            factor_list.append(factors)

    return factor_list


@jax.jit
def step_fn(B, deps, action):
    B = B @ action
    for f in reversed(range(9)):
        B = B @ deps[f]
    # make it a one-hot assuming dynamics should be precise
    qs_next = jnp.argmax(B, axis=-1)
    qs_next = jax.nn.one_hot(qs_next, B.shape[-1])
    return qs_next


@jax.jit
def expand_one_hots(one_hots):
    # add an extra zero bin for "out of bounds"
    zeros = jnp.zeros([one_hots.shape[0], 1])
    one_hots = jnp.concatenate([one_hots, zeros], axis=1)
    return one_hots


@jax.jit
def construct_deps(one_hots, B_dependencies):
    # add a last one-hot state for "out of bounds"
    end = jax.nn.one_hot(jnp.array([one_hots.shape[-1] - 1]), one_hots.shape[-1])
    one_hots = jnp.concatenate([one_hots, end], axis=0)

    # collect for each patch, the dependent patches (i.e. self and 8 neighbors)
    deps = []
    for indices in B_dependencies:
        deps.append(one_hots[indices, :])
    deps = jnp.stack(deps, axis=0)
    return deps


@jax.jit
def sparse_multidimensional_outer(arrs):
    x = arrs[0]
    for q in arrs[1:]:
        x = sparse.sparsify(jnp.expand_dims)(x, axis=-1) * q
    return x


class OGM:

    def __init__(self, size, n_bins, n_actions):
        self.width = size[0]
        self.height = size[1]
        self.n_bins = n_bins + 1
        self.n_actions = n_actions

        self.B_dependencies = generate_factor_list_diagonal(self.width, self.height)
        self.pB = sparse.empty([self.n_bins] * 10 + [self.n_actions])
        self.update_B()

    def save(self, filename):
        # save model fit to .npz file
        data = {}
        data["pB"] = sparse.todense(self.pB)
        jnp.savez(filename, **data)

    def load(self, filename):
        # load model fit from .npz file
        data = jnp.load(filename)
        self.pB = sparse.BCOO.fromdense(data["pB"])

    def update_pB(self, obs, next_obs, action, reward):
        s1 = expand_one_hots(next_obs)
        s0 = construct_deps(expand_one_hots(obs), self.B_dependencies)
        a = jnp.repeat(jax.nn.one_hot(jnp.array([action]), self.n_actions), s0.shape[0], axis=0)
        r = jnp.repeat(jax.nn.one_hot(jnp.array([jnp.clip(reward, -1, 1) + 1]), 3), s0.shape[0], axis=0)

        s1 = sparse.BCOO.fromdense(s1, n_batch=1)
        s0 = sparse.BCOO.fromdense(s0, n_batch=2)
        a = sparse.BCOO.fromdense(a, n_batch=1)
        r = sparse.BCOO.fromdense(r, n_batch=1)

        arrs = [s1] + [s0[:, i, :] for i in range(9)] + [a]
        counts = jax.vmap(sparse_multidimensional_outer)(arrs)

        delta = sparse.sparsify(jnp.sum)(counts, axis=0)
        self.pB = sparse.sparsify(jnp.add)(self.pB, delta)

    def update_B(self):
        self.B = sparse.todense(self.pB) + 1e-8
        self.B = self.B / self.B.sum(axis=0, keepdims=True)

    def predict(self, current_obs, action):
        one_hots = expand_one_hots(current_obs)
        a = jax.nn.one_hot(jnp.array(action), self.n_actions)
        deps = construct_deps(one_hots, self.B_dependencies)
        return jax.vmap(lambda d: step_fn(self.B, d, a))(deps)

    def rollout(self, current_obs, actions):
        current_obs = expand_one_hots(current_obs)

        def scan_fn(current_obs, action):
            deps = construct_deps(current_obs, self.B_dependencies)
            a = jax.nn.one_hot(jnp.array(action), self.n_actions)

            next_obs = jax.vmap(lambda d: step_fn(self.B, d, a))(deps)
            return next_obs, next_obs

        last_obs, result = lax.scan(scan_fn, current_obs, actions)
        return result


# class OGMAgent:

#     def __init__(self, ogm: OGM):
#         self.ogm = ogm
#         self.prior = None

#     def act(self, obs):
#         qs = self.ogm.infer_states(obs, self.prior)
#         q_pi = self.ogm.infer_policies(qs)
#         action = self.ogm.sample_action(q_pi)
#         self.prior = self.ogm.update_empirical_prior(action, qs)
#         return action
