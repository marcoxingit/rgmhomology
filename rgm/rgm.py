import jax.numpy as jnp

from jax import vmap
from functools import partial
from pymdp.agent import Agent
from equinox import tree_at

from rgm.fast_structure_learning import *


class RGM:

    def __init__(
        self,
        max_levels=8,
        dx=2,
        n_bins=9,
        size=None,
        action_range=(-1.0, 1.0),
        svd=True,
        tile_diameter=16,
        n_eigen=16,
    ):
        self.max_levels = max_levels
        self.dx = dx
        self.n_bins = n_bins
        self.action_range = action_range
        self.num_controls = 0

        self.svd = svd
        self.locations_matrix = None
        self.tile_diameter = tile_diameter
        self.n_eigen = n_eigen
        self.action_range = action_range

        self.Vs_padded_and_stacked = None
        self.sv_discrete_axis = None
        self.V_per_patch_dimensions = None

        self.patch_indices = None
        self.patch_centroids = None
        self.patch_weights = None
        self.valid_counts = None

        self.spatial_shape = size
        self.agents = None
        self.RG = None

    def learn_structure(self, observations, actions):
        one_hots = self.to_one_hot(observations, actions)
        self.agents, self.RG, self.LB = spm_mb_structure_learning(
            one_hots,
            self.locations_matrix,
            self.spatial_shape,
            dx=self.dx,
            num_controls=self.num_controls,
            max_levels=self.max_levels,
            agents=self.agents,
            RG=self.RG,
        )
        # shape of one_hot observations that yield one top-level qs
        num_steps = 2 ** (len(self.agents) - 1)
        self.observation_shape = jnp.asarray(
            [one_hots.shape[0], num_steps, one_hots.shape[2]]
        )

    def learn_B(self, observations, actions):
        # TODO this only updates top-level B, should we update all levels?
        one_hots = self.to_one_hot(observations, actions)
        qs = infer(self.agents, one_hots)
        qB, E_qB = learn_transitions(qs, pB=self.agents[-1].B)
        self.agents[-1] = tree_at(lambda x: (x.B), self.agents[-1], (E_qB))

    def save(self, filename):
        # save model fit to .npz file
        data = {}
        # info from SVD to discretize RGB
        if self.svd:
            data[f"Vs_padded_and_stacked"] = self.Vs_padded_and_stacked
            data[f"V_per_patch_dimensions"] = self.V_per_patch_dimensions
            for i in range(len(self.sv_discrete_axis)):
                data[f"sv_discrete_axis_{i}"] = self.sv_discrete_axis[i]
            data["locations_matrix"] = self.locations_matrix
            data[f"patch_indices"] = self.patch_indices
            data["group_indices"] = jnp.asarray(self.group_indices)

        # info to discretize actions
        data["action_bins"] = self.action_bins

        # A and B tensors for agents
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i].A)):
                data[f"A_{i}_{j}"] = self.agents[i].A[j]
                data[f"A_dependencies_{i}_{j}"] = jnp.asarray(
                    self.agents[i].A_dependencies[j]
                )
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i].B)):
                data[f"B_{i}_{j}"] = self.agents[i].B[j]
                data[f"B_dependencies_{i}_{j}"] = jnp.asarray(
                    self.agents[i].B_dependencies[j]
                )

        # generic shapes and controls
        data["observation_shape"] = self.observation_shape
        data["spatial_shape"] = self.spatial_shape
        data["num_controls"] = jnp.array([self.num_controls])
        jnp.savez(filename, **data)

    def load(self, filename):
        # load model fit from .npz file
        data = jnp.load(filename)

        if self.svd:
            self.locations_matrix = jnp.asarray(data["locations_matrix"])
            self.patch_indices = data["patch_indices"]
            self.group_indices = data["group_indices"].tolist()
            self.Vs_padded_and_stacked = jnp.asarray(data["Vs_padded_and_stacked"])
            self.V_per_patch_dimensions = jnp.asarray(data["V_per_patch_dimensions"])
            self.sv_discrete_axis = []
            i = 0
            while f"sv_discrete_axis_{i}" in data:
                self.sv_discrete_axis.append(jnp.asarray(data[f"sv_discrete_axis_{i}"]))
                i += 1

        self.action_bins = jnp.asarray(data["action_bins"])

        agents = []
        for i in range(self.max_levels):
            if f"A_{i}_0" not in data:
                break

            A = []
            A_dependencies = []
            j = 0
            while f"A_{i}_{j}" in data:
                A.append(data[f"A_{i}_{j}"])
                A_dependencies.append(data[f"A_dependencies_{i}_{j}"].tolist())
                j += 1

            B = []
            B_dependencies = []
            j = 0
            while f"B_{i}_{j}" in data:
                B.append(data[f"B_{i}_{j}"])
                B_dependencies.append(data[f"B_dependencies_{i}_{j}"].tolist())
                j += 1

            agents.append(
                Agent(
                    A,
                    B,
                    A_dependencies=A_dependencies,
                    B_dependencies=B_dependencies,
                    apply_batch=False,
                )
            )
        self.agents = agents

        self.observation_shape = data["observation_shape"]
        self.spatial_shape = data["spatial_shape"]
        self.num_controls = data["num_controls"][0]

    def to_one_hot(self, observations, actions=None, mask_indices=None):
        if self.svd:
            self.spatial_shape = jnp.asarray(observations.shape[-3:])
            (
                (
                    observation_one_hots,
                    self.locations_matrix,
                    self.group_indices,
                    self.sv_discrete_axis,
                    self.Vs_padded_and_stacked,
                    self.V_per_patch_dimensions,
                ),
                self.patch_indices,
                self.patch_centroids,
                self.patch_weights,
                self.valid_counts,
            ) = map_rgb_2_discrete(
                observations,
                tile_diameter=self.tile_diameter,
                n_bins=self.n_bins,
                sv_thr=0.0,  # use fixed number of eigenvectors to speed up unique jit
                max_n_modes=self.n_eigen,  # use fixed number of eigenvectors to speed up unique jit
                t_resampling=1,  # use timestep of 1 for SVD
                Vs_padded_and_stacked=self.Vs_padded_and_stacked,
                V_per_patch_dimensions=self.V_per_patch_dimensions,
                sv_discrete_axis=self.sv_discrete_axis,
                patch_indices=self.patch_indices,
                patch_centroids=self.patch_centroids,
                patch_weights=self.patch_weights,
                valid_counts=self.valid_counts,
            )
        else:
            if observations.shape[-1] == 1:
                observation_one_hots = nn.one_hot(
                    jnp.squeeze(observations, -1), self.n_bins
                )
            else:
                observation_one_hots = observations

        if actions is None:
            action_one_hots = (
                jnp.ones(
                    [self.num_controls, observation_one_hots.shape[1], self.n_bins]
                )
                / self.n_bins
            )
        else:
            action_one_hots, self.action_bins = map_action_2_discrete(
                actions,
                n_bins=self.n_bins,
                min_val=self.action_range[0],
                max_val=self.action_range[1],
            )
            self.num_controls = action_one_hots.shape[0]

        mask_val = 1.0 / observation_one_hots.shape[-1]
        if mask_indices is not None:
            if self.svd:
                # mask observation one_hots per group
                for idx in range(len(self.group_indices)):
                    if self.group_indices[idx] in mask_indices:
                        observation_one_hots = observation_one_hots.at[idx, ...].set(
                            mask_val
                        )
            else:
                # mask observation one_hots per channel
                for idx in mask_indices:
                    observation_one_hots = observation_one_hots.at[idx, ...].set(
                        mask_val
                    )

        # prepend action one_hots to first group
        if self.svd:
            self.group_indices = [0] * self.num_controls + self.group_indices

        one_hots = jnp.concatenate((action_one_hots, observation_one_hots), axis=0)
        return one_hots

    def infer_states(
        self, observations, actions=None, priors=None, one_hot_obs=False, version="new"
    ):
        if not one_hot_obs:
            one_hots = self.to_one_hot(observations, actions)
        else:
            one_hots = observations
        qs = (
            infer(self.agents, one_hots, priors)
            if version == "new"
            else infer_old(self.agents, one_hots, priors)
        )
        return qs

    def update_empirical_prior(self, path, qs):
        o, prior = predict(self.agents, qs, path, num_steps=1)

        # only keep the predicted part
        for i in range(len(prior)):
            prior[i] = prior[i][:, prior[i].shape[1] // 2 :, ...]

        return prior

    def predict(self, qs):
        observations, _ = predict(self.agents, qs, None, num_steps=0)
        action_one_hots = observations[0][: self.num_controls]
        actions = map_discrete_2_action(action_one_hots, self.action_bins)

        observation_one_hots = observations[0][self.num_controls :]
        return observation_one_hots, actions

    def reconstruct(self, qs):
        observations, _ = predict(self.agents, qs, None, num_steps=0)
        return self.discrete_2_rgb_action(observations[0])

    def discrete_2_rgb_action(self, one_hots):
        action_one_hots = one_hots[: self.num_controls]
        actions = map_discrete_2_action(action_one_hots, self.action_bins)

        observation_one_hots = one_hots[self.num_controls :]

        if not self.svd:
            return observation_one_hots, actions

        imgs = map_discrete_2_rgb(
            observation_one_hots,
            self.sv_discrete_axis,
            self.patch_indices,
            self.Vs_padded_and_stacked,
            self.V_per_patch_dimensions,
            self.valid_counts,
            self.spatial_shape,
            t_resampling=1,
        )
        imgs = imgs.reshape(
            (
                imgs.shape[0] * imgs.shape[1],
                imgs.shape[-3],
                imgs.shape[-2],
                imgs.shape[-1],
            )
        )

        # convert to plot-able format
        imgs = jnp.transpose(imgs, (0, 2, 3, 1))
        imgs /= 255
        imgs = jnp.clip(imgs, 0, 1)
        imgs = (255 * imgs).astype(jnp.uint8)
        return imgs, actions

    def infer_policies(self, qs):
        pass

    def sample_action(self, q_pi):
        pass


class RGMAgent:

    def __init__(self, rgm: RGM):
        self.rgm = rgm
        self.t = 0

    def reset(self):
        if self.rgm.agents is not None:
            self.t = 0
            self.priors = None
            self.observations = (
                jnp.ones(jnp.asarray(self.rgm.observation_shape))
                / self.rgm.observation_shape[-1]
            )
            self.posterior = jnp.ones([1, self.rgm.agents[-1].A[0].shape[-1]])
            self.posterior = [self.posterior / jnp.sum(self.posterior)]

    def act(self, obs):
        if self.rgm.agents is None:
            # nothing learnt yet, return -1
            return [-1]

        # add latest observations to a sliding window and infer/predict
        one_hots = self.rgm.to_one_hot(jnp.expand_dims(obs, axis=0))
        self.observations = self.observations.at[:, self.t, :].set(one_hots[:, 0, :])
        # print("update given observations")
        self.posterior = self.rgm.infer_states(
            self.observations, priors=self.priors, one_hot_obs=True
        )

        print(
            "top level state idx",
            jnp.argmax(self.posterior[0]),
            self.posterior[0].max(),
            jnp.argwhere(self.posterior[0] > 0.1),
        )

        # TODO reconstruct from posterior, from argmax or from sample?
        idx = jnp.argmax(self.posterior[0])
        qs = jnp.zeros((1, self.posterior[0].shape[-1]))
        qs = qs.at[0, idx].set(1.0)
        pred, actions = self.rgm.predict([qs])
        # pred, actions = self.rgm.predict(self.posterior)
        a = actions[self.t]

        # TODO should we clamp action observations to selected actions?

        self.t += 1

        # if not "confident", do random action instead?!
        # how to determine whether to act or not - use entropy of q or likelihood of observations?
        # entropy = -jnp.sum(self.posterior[0] * jnp.log(self.posterior[0] + 1e-8))
        likelihood = (
            pred[:, : self.t, :]
            * self.observations[self.rgm.num_controls :, : self.t, :]
        ).sum() / (pred.shape[0] * self.t)
        print("likelihood", likelihood)

        if self.t == self.observations.shape[1]:
            # forward 1 tick at the higest level
            self.t = 0
            # TODO "infer" the path at the highest level
            self.priors = self.rgm.update_empirical_prior(
                jnp.array([[0]]), self.posterior
            )
            self.posterior = [self.priors[len(self.rgm.agents) - 1][0]]
            self.observations = (
                jnp.ones_like(self.observations) / self.observations.shape[-1]
            )
            # print("update given prior")

        # if likelihood < 0.99:
        #     # TODO shift time in self.observations?!
        #     return [-1]

        return a.astype(jnp.int32)
