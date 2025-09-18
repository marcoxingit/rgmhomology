import jax.numpy as jnp
from jax import vmap
from jax.experimental import sparse

from functools import partial
from equinox import tree_at

from pymdp.agent import Agent
from pymdp.control import compute_expected_obs, construct_policies, compute_expected_utility


from rgm.fast_structure_learning import *


class RGM:

    def __init__(
        self,
        n_bins=9,
        n_modalities=1,
        width=8,
        height=8,
        max_levels=8,
        dx=2,
    ):
        self.n_bins = n_bins
        self.n_modalities = n_modalities
        self.width = width
        self.height = height
        self.max_levels = max_levels
        self.dx = dx

        self.agents = None
        self.RG = None

    def fit(self, one_hots):
        self.agents, self.RG, _ = spm_mb_structure_learning(
            one_hots,
            None,
            (self.height, self.width, self.n_modalities),
            dx=self.dx,
            num_controls=0,
            max_levels=self.max_levels,
            agents=self.agents,
            RG=self.RG,
        )

    def infer_states(self, one_hots, empirical_prior=None):
        qs = infer(self.agents, one_hots, empirical_prior)
        return qs

    def update_empirical_prior(self, path, qs):
        o, prior = predict(self.agents, qs, path, num_steps=1)

        # only keep the predicted part
        for i in range(len(prior)):
            prior[i] = prior[i][:, prior[i].shape[1] // 2 :, ...]

        return prior

    def reconstruct(self, qs):
        observations, _ = predict(self.agents, qs, None, num_steps=0)
        return observations[0]


@jax.jit
def sparse_multidimensional_outer(arrs):
    x = arrs[0]
    for q in arrs[1:]:
        x = sparse.sparsify(jnp.expand_dims)(x, axis=-1) * q
    return x


class OGM:

    def __init__(self, n_bins=9, n_modalities=1, n_actions=1, shared=False):
        self.n_bins = n_bins
        self.n_modalities = n_modalities
        self.n_actions = n_actions

        self.agent = None
        self.pB = None

        self.shared = shared

    def fit(self, one_hots, actions):
        # generate A/B shared by all patches
        n_frames = one_hots.shape[-2]
        one_hots_per_patch = one_hots.reshape(-1, self.n_modalities, n_frames, self.n_bins).transpose(0, 2, 1, 3)
        n_patches = one_hots_per_patch.shape[0]
        one_hots_per_patch = one_hots_per_patch.reshape(n_patches, n_frames, -1)

        if self.agent is None:
            # get unique modality combinations as hidden states
            if self.shared:
                agg = one_hots_per_patch.reshape(-1, one_hots_per_patch.shape[-1])
                m, n, j = jnp.unique(agg, return_index=True, return_inverse=True, axis=0)
                A_flat = m
                A = jnp.split(A_flat.T, self.n_modalities, axis=0)
                A = jtu.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), n_patches, axis=0), A)
            else:

                def unique_fn(o):
                    return jnp.unique(
                        o, return_inverse=True, return_index=True, axis=0, size=o.shape[0], fill_value=0.0
                    )

                m, n, j = jax.vmap(unique_fn)(one_hots_per_patch)
                n_unique = jnp.max(j) + 1
                patterns = m[:, :n_unique, ...]
                A = jax.vmap(lambda x: jnp.split(x.T, self.n_modalities, axis=0))(patterns)

            n_states = A[0].shape[-1]
            B = [jnp.zeros([n_patches, n_states, n_states, self.n_actions])]

            # construct policies
            policies = construct_policies(
                [n_states],
                [self.n_actions],
                1,
                [0],
            )
            policies = jnp.repeat(policies, n_patches, axis=1)
            self.agent = Agent(A=A, B=B, apply_batch=False, onehot_obs=True, policies=policies)

            if self.shared:
                self.pB = sparse.empty([n_states, n_states, self.n_actions])
            else:
                self.pB = sparse.empty([n_patches, n_states, n_states, self.n_actions], n_batch=1)

            # some util functions
            self.expected_state = partial(compute_expected_state, B_dependencies=self.agent.B_dependencies)
            self.expected_obs = partial(compute_expected_obs, A_dependencies=self.agent.A_dependencies)

            # learn B tensor from observed transitions
            transitions = j.reshape(n_patches, n_frames)
        else:
            # TODO should we expand the state space as well here?

            # create transitions from existing As
            n_states = self.agent.A[0].shape[-1]
            stacked_As = jnp.stack(jtu.tree_map(lambda x: x[0], self.agent.A))
            o = one_hots_per_patch.reshape(-1, self.n_modalities, self.n_bins)
            ell = o[..., None] * log_stable(stacked_As)[None, ...]
            ell = ell.sum(axis=1)
            ell = ell.sum(axis=1)
            transitions = jnp.argmax(ell, axis=-1)
            transitions = transitions.reshape(-1, n_patches).transpose()

        qs = nn.one_hot(transitions, n_states, axis=-1)
        qs_next = qs[:, 1:, :]
        qs = qs[:, :-1, :]
        a = actions[:-1]

        self.update_B(qs, qs_next, a)

    def update_B(self, qs, qs_next, action):
        # do update sparse to avoid memory issues
        a = nn.one_hot(action.squeeze(-1), self.n_actions, axis=-1)
        a = sparse.BCOO.fromdense(a, n_batch=1)
        s1 = sparse.BCOO.fromdense(qs_next, n_batch=2)
        s0 = sparse.BCOO.fromdense(qs, n_batch=2)

        def dirichlet_count(s1, s0):
            counts = vmap(sparse_multidimensional_outer)([s1, s0, a])
            return sparse.sparsify(jnp.sum)(counts, axis=0)

        counts = vmap(dirichlet_count)(s1, s0)

        # update B
        if self.shared:
            delta = sparse.sparsify(jnp.sum)(counts, axis=0)
            self.pB = sparse.sparsify(jnp.add)(self.pB, delta)

            B = sparse.todense(self.pB) + 1e-8
            B = B / B.sum(axis=0, keepdims=True)
            B = jnp.repeat(jnp.expand_dims(B, axis=0), self.agent.B[0].shape[0], axis=0)
        else:
            self.pB = sparse.sparsify(jnp.add)(self.pB, counts)
            B = sparse.todense(self.pB) + 1e-8
            B = B / B.sum(axis=1, keepdims=True)

        self.agent = tree_at(lambda x: (x.B), self.agent, [B])

    def infer_states(self, one_hots, empirical_prior=None):
        # convert one hots [ total_modalities x n_bins] to array of len n_modalities [ n_patches x n_bins]
        # add time dimension
        one_hots = jnp.expand_dims(one_hots, 1)
        obs = self.to_o_list(one_hots)
        if empirical_prior is None:
            empirical_prior = self.agent.D
        qs = self.agent.infer_states(obs, empirical_prior)
        # remove time dimension
        return jtu.tree_map(lambda x: x.squeeze(1), qs)

    def update_empirical_prior(self, action, qs):
        # repeat action for each patch
        action = jnp.repeat(action, self.agent.B[0].shape[0], axis=0)
        # compute expected state given action
        prior = vmap(self.expected_state)(qs, self.agent.B, action)
        return prior

    def reconstruct(self, qs):
        qo = jnp.expand_dims(
            jnp.asarray(jax.vmap(self.expected_obs)(qs, self.agent.A)).transpose(1, 0, 2).reshape(-1, self.n_bins), 1
        )
        return qo

    def to_o_list(self, one_hots):
        o_t = one_hots.reshape(-1, self.n_modalities, one_hots.shape[-2], self.n_bins).transpose(1, 0, 2, 3)
        o_t = jnp.split(o_t, o_t.shape[0], axis=0)
        o_t = jtu.tree_map(lambda x: x.squeeze(0), o_t)
        return o_t

    def infer_policies(self, qs, preferences, t=0):
        c = self.to_o_list(preferences)
        # use t=1 as preference, or rather avg all future?
        c = jtu.tree_map(lambda x: x[:, t, :], c)

        # entropy of next state
        b = jnp.mean(self.agent.B[0], axis=-1)
        qs_next_avg = jax.vmap(lambda b, q: factor_dot(b, q, keep_dims=(0,)))(b, qs)
        H_s = -jnp.sum(qs_next_avg * log_stable(qs_next_avg), axis=-1)

        def eval_policy(policy):
            qs_next = vmap(self.expected_state)(qs, self.agent.B, policy)
            qo = jax.vmap(self.expected_obs)(qs_next, self.agent.A)

            utility = jax.vmap(compute_expected_utility)(qo, c)

            # entropy of next state, conditioned on action
            H_s_a = -jnp.sum(qs_next[0] * log_stable(qs_next[0]), axis=-1)

            return utility, H_s_a

        u, H_s_a = jax.vmap(eval_policy)(self.agent.policies)

        # calculate mutual information between actions and states
        MI = H_s - jnp.mean(H_s_a, axis=0)

        neg_G = jnp.sum(u * MI, axis=-1)
        q_pi = nn.softmax(neg_G)
        return q_pi, u, MI


class RGMAgent:

    def __init__(self, n_bins, n_modalities, n_actions, patches, image_shape, shared=False):
        self.t = 0
        self.shared = shared

        self.rgm = RGM(n_bins, n_modalities, width=patches[0], height=patches[1])
        self.ogm = OGM(n_bins, n_modalities, n_actions, shared=shared)

        self.V = None
        self.projection_bins = None

        self.patch_width = image_shape[1] // patches[1]
        self.patch_height = image_shape[0] // patches[0]

        self.image_shape = (self.patch_height * patches[1], self.patch_width * patches[0])

        # shape of a single "chunk"
        self.observation_shape = None
        self.priors = None
        self.qs = None

    def fit(self, observations, actions):
        one_hots = self.to_one_hot(observations)
        total_modalities, n_frames, n_bins = one_hots.shape

        self.rgm.fit(one_hots)
        self.ogm.fit(one_hots, actions)

        num_steps = 2 ** (len(self.rgm.agents) - 1)
        self.observation_shape = (total_modalities, num_steps, n_bins)

    def to_one_hot(self, observations):
        if self.shared:
            one_hots, self.V, self.projection_bins = map_rgb_patched_2_discrete_shared(
                observations[:, : self.image_shape[0], : self.image_shape[1], :],
                max_n_modes=self.rgm.n_modalities,
                n_bins=self.rgm.n_bins,
                patch_width=self.patch_width,
                patch_height=self.patch_height,
                V=self.V,
                projection_bins=self.projection_bins,
            )
        else:
            one_hots, self.V, self.projection_bins = map_rgb_patched_2_discrete(
                observations[:, : self.image_shape[0], : self.image_shape[1], :],
                max_n_modes=self.rgm.n_modalities,
                n_bins=self.rgm.n_bins,
                patch_width=self.patch_width,
                patch_height=self.patch_height,
                V=self.V,
                projection_bins=self.projection_bins,
            )
        return one_hots

    def to_rgb(self, one_hots):
        if self.shared:
            rgb = map_discrete_2_rgb_patched_shared(
                one_hots,
                self.V,
                self.projection_bins,
                patch_width=self.patch_width,
                patch_height=self.patch_height,
                image_shape=self.image_shape,
            )
        else:
            rgb = map_discrete_2_rgb_patched(
                one_hots,
                self.V,
                self.projection_bins,
                patch_width=self.patch_width,
                patch_height=self.patch_height,
                image_shape=self.image_shape,
            )
        return rgb

    def reset(self):
        self.t = 0

        # reset state / prior / observation window
        self.priors = None
        self.posteriors = None
        self.observations = jnp.ones(jnp.asarray(self.observation_shape)) / self.observation_shape[-1]

        self.qs = self.ogm.agent.D

    def act(self, obs):
        # convert obs to one hot
        one_hots = self.to_one_hot(jnp.expand_dims(obs, axis=0))[:, 0, :]

        # add observation to the ongoing path
        self.observations = self.observations.at[:, self.t, :].set(one_hots)

        # infer RGM state
        self.posteriors = self.rgm.infer_states(self.observations, empirical_prior=self.priors)

        self.t += 1

        # reconstruct RGM path
        if self.t == self.observation_shape[1]:
            # forward the RGM 1 tick
            self.t = 0
            # TODO select best path at highest level instead of just assuming 0
            self.priors = self.rgm.update_empirical_prior(jnp.array([[0]]), self.posteriors)
            self.observations = jnp.ones_like(self.observations) / self.observations.shape[-1]
            preferred_state = [self.priors[-1][0]]
        else:
            preferred_state = self.posteriors

        # generate preferred outcome from RGM

        idx = jnp.argmax(preferred_state[0])
        preferred_sample = jnp.zeros((1, preferred_state[0].shape[-1]))
        preferred_sample = preferred_sample.at[0, idx].set(1.0)
        preferred_state = [preferred_sample]

        preferred_outcomes = self.rgm.reconstruct(preferred_state)

        # infer OGM state
        self.qs = self.ogm.infer_states(one_hots, empirical_prior=self.qs)

        # infer OGM actions given RGM preference
        q_pi, u, mi = self.ogm.infer_policies(self.qs, preferred_outcomes, t=self.t)
        action = jnp.argmax(q_pi)

        info = {
            "observation": obs,
            "action": action,
            "qs": self.qs,
            "preferred_outcomes": preferred_outcomes,
            "posteriors": self.posteriors,
            "priors": self.priors,
            "one_hots": one_hots,
            "mi": mi,
            "t": self.t,
        }

        # update OGM empirical prior
        self.qs = self.ogm.update_empirical_prior(jnp.array([[action]]), self.qs)

        return action, info
