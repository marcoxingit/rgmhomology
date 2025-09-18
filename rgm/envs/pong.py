import jax
import jax.numpy as jnp

from equinox import field
from jaxtyping import Array

import math
import matplotlib.pyplot as plt
from pymdp.utils import fig2img

from pymdp.envs.env import Env


def obs2img(observation, num_rows, num_cols, batch_idx=0):
    num_groups = num_rows * num_cols

    img = jnp.zeros([num_rows, num_cols])
    for g in range(num_groups):
        x, y = jnp.unravel_index(g, (num_rows, num_cols))
        img = img.at[x, y].set(observation[g][batch_idx, 0] / 4)

    return img


class Pong(Env):

    num_rows: int = field(static=True)
    num_cols: int = field(static=True)
    cid: Array = field()
    hid: Array = field()

    def __init__(self, num_rows=12, num_cols=9, batch_size=1):
        self.num_rows = num_rows
        self.num_cols = num_cols

        A = []
        A_dependencies = []
        B = []
        B_dependencies = []
        D = []

        num_groups = num_rows * num_cols
        # Karl lets the ball bounce around from start position for num_states steps
        # if the ball return to a same state (=orbit), we're done
        num_states = 4098

        for g in range(num_groups):
            # likelihood tensor yields observations:
            # [ball, paddle (left), paddle (center), paddle (right), background]
            # from two state factors: ball state + paddle state
            A.append(jnp.zeros([5, num_states, num_cols]))
            A[g] = A[g].at[-1, :, :].set(1.0)  # default is background
            A_dependencies.append([0, 1])

        # transitions of the ball state
        B.append(jnp.zeros([num_states, num_states, 1]))
        B_dependencies.append([0])
        # transitions of the paddle state
        B.append(jnp.zeros([num_cols, num_cols, 3]))
        B_dependencies.append([1])

        # start bouncing the ball
        i = 2  # initial location (horizontal)
        j = 2  # initial location (vertical)
        p = 1  # momentum (horizontal)
        q = 1  # momentum (vertical)

        visited = jnp.zeros([num_states, 4])
        for s in range(num_states):
            r = jnp.array([i, j, p, q])
            k = (jnp.abs(visited - r.reshape(1, -1))).sum(axis=1) == 0
            if jnp.any(k):
                r = jnp.argwhere(k)
                s = s - 1

                B[0] = B[0].at[s + 1, s, 0].set(0.0)
                B[0] = B[0].at[r, s, 0].set(1.0)
                B[0] = B[0][: s + 1, : s + 1, :]
                for g in range(num_groups):
                    A[g] = A[g][:, : s + 1, :]
                break
            else:
                visited = visited.at[s, :].set(r)

            n = jnp.ravel_multi_index(jnp.array([i, j]), (num_rows, num_cols))
            A[n] = A[n].at[:, s, :].set(0.0)
            A[n] = A[n].at[0, s, :].set(1.0)
            B[0] = B[0].at[s + 1, s, 0].set(1.0)

            # switch momentum at the boundaries
            if i == 0 or i == num_rows - 1:
                p = -p
            if j == 0 or j == num_cols - 1:
                q = -q

            # transition ball
            i = i + p
            j = j + q

        # paddle transition and likelihood
        for c in range(num_cols):

            # paddle left
            if c > 0:
                n = jnp.ravel_multi_index(jnp.array([0, c - 1]), (num_rows, num_cols))
                A[n] = A[n].at[:, :, c].set(0.0)
                A[n] = A[n].at[1, :, c].set(1.0)

            # paddle center
            n = jnp.ravel_multi_index(jnp.array([0, c]), (num_rows, num_cols))
            A[n] = A[n].at[:, :, c].set(0.0)
            A[n] = A[n].at[2, :, c].set(1.0)

            # paddle right
            if c < num_cols - 1:
                n = jnp.ravel_multi_index(jnp.array([0, c + 1]), (num_rows, num_cols))
                A[n] = A[n].at[:, :, c].set(0.0)
                A[n] = A[n].at[3, :, c].set(1.0)

            # up
            c_next = c + 1 if c < num_cols - 1 else c
            B[1] = B[1].at[c_next, c, 0].set(1.0)

            # noop
            B[1] = B[1].at[c, c, 1].set(1.0)

            # down
            c_next = c - 1 if c > 0 else c
            B[1] = B[1].at[c_next, c, 2].set(1.0)

        D.append(jnp.zeros(num_states))
        D[0] = D[0].at[0].set(1.0)
        D.append(jnp.zeros(num_cols))
        D[0] = D[1].at[0].set(1.0)

        A = [jnp.broadcast_to(a, (batch_size,) + a.shape) for a in A]
        B = [jnp.broadcast_to(b, (batch_size,) + b.shape) for b in B]
        D = [jnp.broadcast_to(d, (batch_size,) + d.shape) for d in D]

        params = {
            "A": A,
            "B": B,
            "D": D,
        }

        dependencies = {
            "A": A_dependencies,
            "B": B_dependencies,
        }

        # hid   - Hidden states corresponding to hits
        # cid   - Hidden states corresponding to misses
        c = []
        h = []
        for s1 in range(num_states):
            for s2 in range(num_cols):
                if visited[s1, 0] == 0 and s2 != visited[s1, 1]:
                    c.append([s1, s2])
                if visited[s1, 0] == 0 and s2 == visited[s1, 1]:
                    h.append([s1, s2])
        cid = jnp.stack([jnp.asarray(x) for x in c])
        hid = jnp.stack([jnp.asarray(x) for x in h])
        self.cid = jnp.broadcast_to(cid, (batch_size,) + cid.shape)
        self.hid = jnp.broadcast_to(hid, (batch_size,) + hid.shape)

        super().__init__(params, dependencies)

    def reset(self, key, state=None):
        new_obs, env = super().reset(key, state)
        return self.append_reward(new_obs), env

    def step(self, rng_key, actions=None):
        new_obs, env = super().step(rng_key, actions)
        return self.append_reward(new_obs), env

    def append_reward(self, new_obs):
        # TODO assume batch size is 1 here?!
        current_state = jnp.concatenate(self.state)
        if jnp.any(jnp.all(self.cid[0] == current_state, axis=1)):
            new_obs.append(-jnp.ones(new_obs[0].shape))
        elif jnp.any(jnp.all(self.hid[0] == current_state, axis=1)):
            new_obs.append(jnp.ones(new_obs[0].shape))
        else:
            new_obs.append(jnp.zeros(new_obs[0].shape))
        return new_obs

    def render(self, mode="human"):
        batch_size = self.params["A"][0].shape[0]

        n = math.ceil(math.sqrt(batch_size))
        fig, axes = plt.subplots(n, n, figsize=(6, 6))

        for i in range(batch_size):
            row = i // n
            col = i % n
            if batch_size == 1:
                ax = axes
            else:
                ax = axes[row, col]

            img = obs2img(self.current_obs, self.num_rows, self.num_cols, i)

            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img, cmap="gray")

        for i in range(batch_size, n * n):
            fig.delaxes(axes.flatten()[i])

        plt.tight_layout()

        if mode == "human":
            plt.show()
        elif mode == "rgb_array":
            return fig2img(fig)

    def _sample_obs(self, key, state):
        return jax.jit(Env._sample_obs)(self, key, state)
