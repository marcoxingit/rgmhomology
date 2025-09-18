import time
import cv2
import numpy as onp
import jax
from jax import numpy as jnp
from jax import vmap, nn, jit
from jax import tree_util as jtu
from jaxtyping import Array
from opt_einsum import contract

from functools import partial
from typing import List

from pymdp.agent import Agent
from pymdp.control import compute_expected_obs, compute_expected_state
from pymdp.maths import factor_dot, log_stable
from pymdp.algos import run_factorized_fpi
from pymdp.learning import update_state_transition_dirichlet

from functools import partial

import math
from math import prod


def read_frames_from_npz(file_path: str, num_frames: int = 32, rollout: int = 0):
    """read frames from a npz file from atari expert trajectories"""
    # shape is [num_rollouts, num_frames, 1, height, width, channels]
    res = onp.load(file_path)
    frames = res["arr_0"][rollout, 0:num_frames, 0, ...]
    return frames


def read_frames_from_mp4(file_path: str, num_frames: int = 32, size: tuple[int] = (128, 128)):
    """ " read frames from an mp4 file"""
    cap = cv2.VideoCapture(file_path)

    width, height = size
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    x_center = video_width // 2
    y_center = video_height // 2
    # x_start = max(0, x_center - width // 2)
    # y_start = max(0, y_center - height // 2)
    x_start = max(0, x_center - width)
    y_start = max(0, y_center - height)

    frame_indices = jnp.linspace(0, total_frames - 1, num_frames, dtype=int)
    frame_indices = jnp.concatenate((frame_indices, frame_indices), axis=0)
    frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx.item()))

        ret, frame = cap.read()
        if not ret:
            break

        # cropped_frame = frame[y_start:y_start+height, x_start:x_start+width]
        cropped_frame = frame[y_start : y_start + 2 * height, x_start : x_start + 2 * width]
        resized_frame = cv2.resize(cropped_frame, size)
        frames.append(resized_frame)

    cap.release()
    return jnp.array(frames)


def map_rgb_2_discrete(
    image_data: Array,
    tile_diameter=32,
    n_bins=9,
    max_n_modes=32,
    sv_thr=(1.0 / 32.0),
    t_resampling=2,
    Vs_padded_and_stacked=None,
    V_per_patch_dimensions=None,
    sv_discrete_axis=None,
    patch_indices=None,
    patch_centroids=None,
    patch_weights=None,
    valid_counts=None,
):
    """Re-implementation of `spm_rgb2O.m` in Python
    Maps an RGB image format to discrete outcomes

    Args:
    image_data: Array
        The image data to be mapped to discrete outcomes. Shape (num_frames, width, height, channels)
    tile_diameter: int
        Diameter of the tiles (`nd` in the original code)
    n_bins: int
        Number of variates (`nb` in the original code)
    max_n_modes: int
        Maximum number of modes (`nm` in the original code)
    sv_thr: int
        Threshold for singular values (`su` in the original code)
    t_resampling: int
        Threshold for temporal resampling (`R` in the original code)
    Vs_padded_and_stacked: Array
        Batch of eigenvectors per patch from a previous SVD
    V_per_patch_dimensions: List[int]
        Number of rows in each V_per_patch from a previous SVD
    sv_discrete_axis: List[Array]
        List of quantization bins for singular variates from a previous SVD
    """
    # ensure number of bins is odd (for symmetry)
    n_bins = int(2 * jnp.trunc(n_bins / 2) + 1)

    n_frames, width, height, n_channels = image_data.shape
    T = int(t_resampling * jnp.trunc(n_frames / t_resampling))  # length of time partition

    # transpose to [T x C x W x H]
    image_data = jnp.transpose(
        image_data[:T, ...], (0, 3, 1, 2)
    )  # truncate the time series and transpose the axes to the right place

    # concat each t_resampling frames
    image_data = image_data.reshape((T // t_resampling, -1, width, height))

    # shape of the data excluding the time dimension ((t_resampling * C) x W x H)
    shape_no_time = image_data.shape[1:]

    # patch_indices, patch_centroids, patch_weights = spm_tile(
    #     width=shape_no_time[1],
    #     height=shape_no_time[2],
    #     n_copies=shape_no_time[0],
    #     tile_diameter=tile_diameter,
    # )

    # only run the tiling if its outputs are not already provided / in existence
    if patch_indices is None or patch_centroids is None or patch_weights is None or valid_counts is None:
        patch_indices, patch_centroids, patch_weights, valid_counts = spm_tile_fast(
            width=shape_no_time[1],
            height=shape_no_time[2],
            tile_diameter=tile_diameter,
        )

    return (
        patch_svd(
            image_data,
            patch_indices,
            patch_centroids,
            patch_weights,
            valid_counts,
            sv_thr=sv_thr,
            max_n_modes=max_n_modes,
            n_bins=n_bins,
            Vs_padded_and_stacked=Vs_padded_and_stacked,
            V_per_patch_dimensions=V_per_patch_dimensions,
            sv_discrete_axis=sv_discrete_axis,
        ),
        patch_indices,
        patch_centroids,
        patch_weights,
        valid_counts,
    )


@partial(jit, static_argnames=["max_patch_dimension"])
def batched_svd_projection(
    image_data: Array,
    patch_indices: Array,
    patch_weights: Array,
    padded_Vs: Array,
    valid_counts: Array,
    max_patch_dimension: int = 100,
):
    n_frames, channels_x_duplicates, width, height = image_data.shape

    Y_all_patches = (
        jnp.take(image_data.reshape(n_frames, channels_x_duplicates, width * height), patch_indices, axis=2)
        * patch_weights
    )
    Y_per_patch = [jnp.empty((n_frames, max_patch_dimension))] * len(patch_indices)
    for g_i, patch_g_indices in enumerate(patch_indices):
        # the line below doesn't jit
        """IndexError: Array slice indices must have static start/stop/step to be used with NumPy indexing syntax.
        Found slice(None, Traced<ShapedArray(int32[])>with<DynamicJaxprTrace(level=1/0)>, None).
        To index a statically sized array at a dynamic position, try lax.dynamic_slice/dynamic_update_slice
        (JAX does not support dynamically sized arrays within JIT compiled functions).
        """
        y_g = Y_all_patches[:, :, g_i, : valid_counts[g_i]].reshape(n_frames, channels_x_duplicates * valid_counts[g_i])
        # now pad the columns with zeros to make it equal to max_patch_dimension = channels_x_duplicates*max(valid_counts)
        Y_per_patch[g_i] = jnp.pad(
            y_g, ((0, 0), (0, max_patch_dimension - y_g.shape[1])), mode="constant", constant_values=0
        )

    Y_all_patches = jnp.stack(Y_per_patch)

    # then do a batched matrix multiplication where Y_all_patches = (num_patches, n_frames, channels_x_duplicates*max(valid_counts)) and padded_Vs = (num_patches, channels_x_duplicates*max(valid_counts), num_eigenvectors)
    # this will give us the projections for each patch
    u_all_patches = Y_all_patches @ padded_Vs

    return u_all_patches


def patch_svd(
    image_data: Array,
    patch_indices: List[Array],
    patch_centroids,
    patch_weights: List[Array],
    valid_counts: Array,
    sv_thr: float = 1e-6,
    max_n_modes: int = 32,
    n_bins: int = 9,
    Vs_padded_and_stacked=None,
    V_per_patch_dimensions=None,
    sv_discrete_axis=None,
):
    """
    image_data: [time, channel, width, height]
    patch_indices: [[indicies_for_patch] for num_patches]
    patch_weights: [[indices_for_patch, num_patches] for num_patches]
    """
    n_frames, channels_x_duplicates, width, height = image_data.shape
    o_idx = 0
    observations = []
    locations_matrix = []
    group_indices = []

    # if V_per_patch and sv_discrete_axis are not provided, we need to do SVD
    do_svd = False
    if Vs_padded_and_stacked is None or sv_discrete_axis is None:
        sv_discrete_axis = []
        V_per_patch = [None] * len(patch_indices)
        do_svd = True

    if not do_svd:

        # current doesn't work, due to 'non-Hashable static arguments'error...?
        # u_all_patches = batched_svd_projection(image_data, patch_indices, patch_weights, Vs_padded_and_stacked, valid_counts, max_patch_dimension=channels_x_duplicates * int(max(valid_counts).item()))

        max_patch_dimension = channels_x_duplicates * max(valid_counts)
        Y_all_patches = (
            jnp.take(image_data.reshape(n_frames, channels_x_duplicates, width * height), patch_indices, axis=2)
            * patch_weights
        )
        Y_per_patch = [jnp.empty((n_frames, max_patch_dimension))] * len(patch_indices)
        for g_i, patch_g_indices in enumerate(patch_indices):
            y_g = Y_all_patches[:, :, g_i, : valid_counts[g_i]].reshape(
                n_frames, channels_x_duplicates * valid_counts[g_i]
            )
            # now pad the columns with zeros to make it equal to max_patch_dimension = channels_x_duplicates*max(valid_counts)
            Y_per_patch[g_i] = jnp.pad(
                y_g, ((0, 0), (0, max_patch_dimension - y_g.shape[1])), mode="constant", constant_values=0
            )

        Y_all_patches = jnp.stack(Y_per_patch)

        # then do a batched matrix multiplication where Y_all_patches = (num_patches, n_frames, channels_x_duplicates*max(valid_counts)) and Vs_padded_and_stacked = (num_patches, channels_x_duplicates*max(valid_counts), num_eigenvectors)
        # this will give us the projections for each patch
        # (num_patches, n_frames, num_eigenvectors a.k.a. num_modalities)
        u_all_patches = Y_all_patches @ Vs_padded_and_stacked

        # V = V_per_patch[g_i]
        # u = Y @ V

        num_groups, n_frames, num_modalities = u_all_patches.shape

        # this will have shape (num_groups, num_modalities, n_bins)
        sv_discrete_axes_reshaped = jnp.stack(sv_discrete_axis).reshape(num_groups, num_modalities, -1)

        # (num_groups, n_frames, num_modalities)
        min_indices = jnp.argmin(
            jnp.abs(u_all_patches[..., None] - jnp.expand_dims(sv_discrete_axes_reshaped, 1)), axis=-1
        )

        # (num_groups, n_frames, num_modalities, n_bins)
        observations = nn.one_hot(min_indices, n_bins)

        # reshape observations to (num_groups*num_modalities, n_frames, n_bins)
        observations = jnp.transpose(observations, (0, 2, 1, 3)).reshape(-1, n_frames, n_bins)

        group_indices = jnp.repeat(jnp.arange(num_groups), num_modalities).tolist()

        locations_matrix = jnp.repeat(patch_centroids, num_modalities, axis=0)

        # for m in range(num_modalities):
        #     observations.append([])
        #     for t in range(u.shape[0]):
        #         min_indices = jnp.argmin(jnp.abs(u[t, m] - sv_discrete_axis[o_idx]))
        #         observations[o_idx].append(nn.one_hot(min_indices, n_bins))

        #     locations_matrix.append(patch_centroids[g_i, :])
        #     group_indices.append(g_i)
        #     o_idx += 1
    else:
        # iterate over all patches
        for g_i, patch_g_indices in enumerate(patch_indices):

            # get the pixels of each patch, weighted by their distance to the centroid
            # Y = (
            #     image_data.reshape(n_frames, channels_x_duplicates * width * height)[:, patch_g_indices]
            #     * patch_weights[g_i]
            # )

            # using the new version of patch_inddices, which are in terms of wdith * height, not in terms of channel_x_duplicates * width * height
            Y = (
                image_data.reshape(n_frames, channels_x_duplicates, width * height)[
                    :, :, patch_g_indices[: valid_counts[g_i]]
                ]
                * patch_weights[g_i][: valid_counts[g_i]]
            )
            # now reshape into n_frames, channel_x_duplicates * num_pixels_in_current_patch
            Y = Y.reshape(n_frames, channels_x_duplicates * valid_counts[g_i])

            # new_image = jnp.zeros(channels_x_duplicates*width*height)
            # new_image = new_image.at[patch_g_indices].set(patch_weights[g_i])
            # new_image = new_image.reshape(channels_x_duplicates, width, height)
            # plt.imshow(new_image[0])
            # plt.title(f'Patch {g_i}')
            # plt.show()

            # (n_frames x n_frames), (n_frames,), (n_frames x n_frames)
            U, svals, V = jnp.linalg.svd(Y @ Y.T, full_matrices=True)

            # plt.imshow(Y@Y.T)
            # plt.title(f'Patch {g_i}')
            # plt.show()

            normalized_svals = svals * (len(svals) / svals.sum())
            topK_svals = normalized_svals >= sv_thr  # equivalent of `j` in spm_svd.m
            topK_s_vectors = U[:, topK_svals]

            projections = Y.T @ topK_s_vectors  # do equivalent of spm_en on this one
            projections_normed = projections / jnp.linalg.norm(projections, axis=0, keepdims=True)

            svals = jnp.sqrt(svals[topK_svals])

            num_modalities = min(len(svals), max_n_modes)

            if num_modalities > 0:
                V_per_patch[g_i] = projections_normed[:, :num_modalities]
                weighted_topk_s_vectors = topK_s_vectors[:, :num_modalities] * svals[:num_modalities]

            # generate (probability over discrete) outcomes
            for m in range(num_modalities):

                # discretise singular variates
                d = jnp.max(jnp.abs(weighted_topk_s_vectors[:, m]))

                # this determines the number of bins
                projection_bins = jnp.linspace(-d, d, n_bins)

                observations.append([])
                for t in range(n_frames):

                    # finds the index of of the projection at time t, for singular vector m, in the projection bins -- this will determine how it gets discretized
                    min_indices = jnp.argmin(jnp.absolute(weighted_topk_s_vectors[t, m] - projection_bins))

                    # observations are a one-hot vector reflecting the quantization of each singular variate into one of the projection bins
                    observations[o_idx].append(nn.one_hot(min_indices, n_bins))

                # record locations and group for this outcome
                locations_matrix.append(patch_centroids[g_i, :])
                group_indices.append(g_i)
                sv_discrete_axis.append(projection_bins)
                o_idx += 1

        max_patch_dimension = channels_x_duplicates * max(valid_counts)
        # pad the V's per patch along their rows to make them equal to channels_x_duplicates*max(valid_counts)
        # store the number of rows in each V_per_patch
        V_per_patch_dimensions = [V_per_patch[g_i].shape[0] for g_i in range(len(V_per_patch))]
        Vs_padded_and_stacked = jnp.stack(
            jtu.tree_map(
                lambda x: jnp.pad(
                    x, ((0, max_patch_dimension - x.shape[0]), (0, 0)), mode="constant", constant_values=0
                ),
                V_per_patch,
            )
        )

        locations_matrix = jnp.stack(locations_matrix)
        observations = jnp.asarray(observations)

    return (
        observations,
        locations_matrix,
        group_indices,
        sv_discrete_axis,
        Vs_padded_and_stacked,
        V_per_patch_dimensions,
    )


def map_discrete_2_rgb_old(
    observations,
    group_indices,
    sv_discrete_axis,
    V_per_patch,
    # Vs_padded_and_stacked,
    # V_per_patch_dimensions,
    patch_indices,
    # valid_counts,
    image_shape,
    t_resampling=2,
):
    n_groups = len(patch_indices)

    # image_shape given as [W H C]
    # shape = [t_resampling, image_shape[-1], image_shape[-3], image_shape[-2]]

    width, height, n_channels = image_shape

    # recons_image = jnp.zeros(prod(shape))
    recons_image = jnp.zeros((t_resampling * n_channels, width * height))

    for group_idx in range(n_groups):

        # V_patch_i = Vs_padded_and_stacked[group_idx,:V_per_patch_dimensions]

        modality_idx_in_patch = [modality_idx for modality_idx, g_i in enumerate(group_indices) if g_i == group_idx]
        num_modalities_in_patch = len(modality_idx_in_patch)

        matched_bin_values = []
        for m in range(num_modalities_in_patch):
            m_idx = modality_idx_in_patch[m]
            matched_bin_values.append(sv_discrete_axis[m_idx].dot(observations[m_idx]))

        matched_bin_values = jnp.array(matched_bin_values)

        if len(matched_bin_values) > 0:

            reconstructed_image_patch = V_per_patch[group_idx].dot(matched_bin_values)
            reconstructed_image_patch_reshaped = reconstructed_image_patch.reshape(t_resampling * n_channels, -1)

            recons_image = recons_image.at[:, patch_indices[group_idx]].set(
                recons_image[:, patch_indices[group_idx]] + reconstructed_image_patch_reshaped
            )
            # recons_image = recons_image.at[patch_indices[group_idx]].set(
            #     recons_image[patch_indices[group_idx]] + V_per_patch[group_idx].dot(matched_bin_values)
            # )

    recons_image = recons_image.reshape((t_resampling, n_channels, width, height))
    # recons_image = jnp.transpose(recons_image, (0, 2, 3, 1))
    return recons_image


def map_discrete_2_rgb(
    observations,
    sv_discrete_axis,
    patch_indices,
    Vs_padded_and_stacked,
    V_per_patch_dimensions,
    valid_counts,
    image_shape,
    t_resampling=2,
):

    n_frames = observations.shape[1]
    num_groups = patch_indices.shape[0]
    num_modalities = Vs_padded_and_stacked.shape[-1]

    # image_shape given as [W H C]
    width, height, n_channels = image_shape

    # shape = [n_frames, t_resampling, n_channels, width, height]

    recons_image = jnp.zeros((n_frames, t_resampling * n_channels, width * height))

    # jnp.stack(sv_discrete_axis).shape = (num_groups * num_modalities, n_bins)
    # jnp.moveaxis(observations, 1, 0).shape = (n_frames, num_groups * num_modalities, n_bins)
    # total output shape = (n_frames, num_groups * num_modalities)
    matched_bin_values = (jnp.stack(sv_discrete_axis) * jnp.moveaxis(observations, 1, 0)).sum(-1)

    # (num_groups, t_resampling*n_channels*max(valid_counts), num_modalities) x (n_frames, num_groups, num_modalities) --> (n_frames, num_groups, t_resampling*n_channels*max(valid_counts))
    reconstructed_images_just_valid_pixels = contract(
        "gpm,fgm->fgp", Vs_padded_and_stacked, matched_bin_values.reshape(n_frames, num_groups, num_modalities)
    )

    # now we need to fill out the full image with the appropriate pixels
    # reconstructed_images_just_valid_pixels= reconstructed_images_just_valid_pixels.reshape(n_frames, num_groups, t_resampling*n_channels, max(valid_counts))

    for g_i, patch_indices_i in enumerate(patch_indices):
        reconstructed_patch_valid = reconstructed_images_just_valid_pixels[:, g_i, : V_per_patch_dimensions[g_i]]
        reconstructed_patch_reshaped = reconstructed_patch_valid.reshape(n_frames, t_resampling * n_channels, -1)
        recons_image = recons_image.at[:, :, patch_indices_i[: valid_counts[g_i]]].set(
            recons_image[:, :, patch_indices_i[: valid_counts[g_i]]]
            + reconstructed_patch_reshaped[:, :, : valid_counts[g_i]]
        )

    return recons_image.reshape((n_frames, t_resampling, n_channels, width, height))


def map_action_2_discrete(actions, n_bins=9, dt=1, min_val=-1, max_val=1):
    """
    Maps continuous actions to discrete outcomes
    Args:
        actions: Array
            The actions to be mapped to discrete outcomes. Shape (num_frames, num_actions)
        n_bins: int
            Number of bins to quantize the actions into
    """

    action_bins = jnp.linspace(min_val, max_val, n_bins)
    min_indices = jnp.argmin(jnp.absolute(actions[:, :, jnp.newaxis] - action_bins), axis=-1)
    one_hots = nn.one_hot(min_indices, n_bins)

    # when svd-ing, we lump together 2 frames/actions per timestep
    reshaped = one_hots.reshape(-1, one_hots.shape[1] * dt, n_bins)
    transposed = jnp.transpose(reshaped, (1, 0, 2))
    return transposed, action_bins


def map_discrete_2_action(action_one_hots, action_bins, dt=1):
    """
    Maps discrete outcomes to continuous actions
    Args:
        action_one_hots: Array
            The one-hot encoded actions. Shape (num_frames, num_actions, num_bins)
        action_bins: Array
            The bins to quantize the actions into. Shape (num_bins,)
    """

    transposed = jnp.transpose(action_one_hots, (1, 0, 2))
    actions = jnp.sum(transposed * action_bins, axis=-1)
    actions = actions.reshape(-1, actions.shape[1] // dt)
    return actions


def frames_2_patches(observations, patch_width=8, patch_height=8):
    n_frames, height, width, channels = observations.shape

    # Calculate the number of blocks
    num_p_h, num_p_w = height // patch_height, width // patch_width
    n_patches = num_p_w * num_p_h

    # Reshape and transpose to get the patches in the desired format
    reshaped = observations.reshape(n_frames, num_p_h, patch_height, num_p_w, patch_width, channels)
    patches = reshaped.transpose(0, 1, 3, 2, 4, 5).reshape(n_frames, n_patches, patch_height, patch_width, channels)

    return patches


def map_rgb_patched_2_discrete_shared(
    observations, max_n_modes=8, n_bins=9, patch_width=8, patch_height=8, V=None, projection_bins=None
):
    """
    Convert a batch of RGB frames to a batch of discrete outcomes, by projecting on the principle eigenvectors.
    If no V and projection_bins are provided, they are calculated by doing SVD on the input data.
    In this case, all patches share the same V and projection_bins.
    """
    observations = observations / 255.0
    patches = frames_2_patches(observations, patch_width, patch_height)

    n_frames, n_patches, _, _, _ = patches.shape

    # should be (num_frames x num_patches , patch_height x patch_width x channels)
    Y = patches.reshape(n_frames * n_patches, -1)
    if V is None:
        _, svals, V = jnp.linalg.svd(Y.T @ Y, full_matrices=True)
        V = V[:max_n_modes, :].T

    Z = Y @ V

    if projection_bins is None:
        d = jnp.max(jnp.abs(Z), axis=0)
        projection_bins = jnp.linspace(-d, d, n_bins)

    def get_bin(value, bins):
        return nn.one_hot(jnp.argmin(jnp.absolute(value - bins)), n_bins)

    # (num_frames x num_patches, n_modes, n_bins)
    one_hots = jax.vmap(lambda x: jax.vmap(get_bin)(x, projection_bins.T))(Z)

    # reshape to (n_modalities, n_frames, n_bins)
    # with n_modalities = n_modes x n_patches per frame
    one_hots = one_hots.reshape(n_frames, n_patches, max_n_modes, n_bins)
    one_hots = one_hots.transpose(1, 2, 0, 3)
    one_hots = one_hots.reshape(n_patches * max_n_modes, n_frames, n_bins)
    return one_hots, V, projection_bins


def map_rgb_patched_2_discrete(
    observations, max_n_modes=8, n_bins=9, patch_width=8, patch_height=8, V=None, projection_bins=None
):
    """
    Convert a batch of RGB frames to a batch of discrete outcomes, by projecting on the principle eigenvectors.
    If no V and projection_bins are provided, they are calculated by doing SVD on the input data.
    In this case, each patch has own V and projection_bins.
    """
    observations = observations / 255.0
    patches = frames_2_patches(observations, patch_width, patch_height)

    n_frames, n_patches, _, _, _ = patches.shape

    # should be (num_patches, num_frames, patch_height * patch_width * channels)
    Y = patches.reshape(n_frames, n_patches, -1).transpose(1, 0, 2)
    if V is None:

        def calc_V_fn(y):
            _, _, v = jnp.linalg.svd(y.T @ y, full_matrices=True)
            return v[:max_n_modes, :].T

        V = jax.vmap(calc_V_fn)(Y)

    Z = jax.vmap(jnp.matmul)(Y, V)

    if projection_bins is None:

        def calc_bins_fn(Z):
            d = jnp.max(jnp.abs(Z), axis=0)
            return jnp.linspace(-d, d, n_bins)

        projection_bins = jax.vmap(calc_bins_fn)(Z)

    def get_bin(value, bins):
        return nn.one_hot(jnp.argmin(jnp.absolute(value - bins)), n_bins)

    # (num_frames x num_patches, n_modes, n_bins)
    def to_one_hot(Z, bins):
        return jax.vmap(lambda x: jax.vmap(get_bin)(x, bins.T))(Z)

    one_hots = jax.vmap(to_one_hot)(Z, projection_bins)

    # reshape to (n_modalities, n_frames, n_bins)
    # with n_modalities = n_modes x n_patches per frame
    one_hots = one_hots.transpose(0, 2, 1, 3)
    one_hots = one_hots.reshape(n_patches * max_n_modes, n_frames, n_bins)
    return one_hots, V, projection_bins


def patches_2_frames(patches, image_shape=(64, 64)):
    height = image_shape[0]
    width = image_shape[1]
    n_frames, n_patches, patch_height, patch_width, channels = patches.shape
    num_p_h, num_p_w = height // patch_height, width // patch_width

    patches = patches.reshape(n_frames, num_p_h, num_p_w, patch_height, patch_width, 3)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    return patches.reshape(n_frames, height, width, 3)


def map_discrete_2_rgb_patched(one_hots, V, projection_bins, patch_width=8, patch_height=8, image_shape=(64, 64)):
    n_patches, n_bins, n_modes = projection_bins.shape
    n_modalities, n_frames, _ = one_hots.shape

    one_hots = one_hots.reshape(n_patches, n_modes, n_frames, n_bins)

    def reconstruct_patch(patch, p, v):
        vals = (patch.transpose(1, 0, 2) * p.T).sum(axis=-1)
        return vals @ v.T

    patches = jax.vmap(reconstruct_patch)(one_hots, projection_bins, V)
    patches = patches.transpose(1, 0, 2).reshape(n_frames, -1, patch_height, patch_width, 3)
    return patches_2_frames(patches, image_shape)


def map_discrete_2_rgb_patched_shared(
    one_hots, V, projection_bins, patch_width=8, patch_height=8, image_shape=(64, 64)
):
    n_bins, n_modes = projection_bins.shape
    n_modalities, n_frames, _ = one_hots.shape

    one_hots = one_hots.reshape(-1, n_modes, n_frames, n_bins)

    def reconstruct_patch(patch):
        vals = (patch.transpose(1, 0, 2) * projection_bins.T).sum(axis=-1)
        return vals @ V.T

    patches = jax.vmap(reconstruct_patch)(one_hots)
    patches = patches.transpose(1, 0, 2).reshape(n_frames, -1, patch_height, patch_width, 3)
    return patches_2_frames(patches, image_shape)


def spm_dir_norm(a):
    """
    Normalisation of a (Dirichlet) conditional probability matrix
    Args:
        A: (Dirichlet) parameters of a conditional probability matrix
    Returns:
        A: normalised conditional probability matrix
    """
    a0 = jnp.sum(a, axis=0)
    i = a0 > 0
    a = jnp.where(i, a / a0, a)
    a = a.at[:, ~i].set(1 / a.shape[0])
    return a


def spm_tile_fast(width: int, height: int, tile_diameter: int = 32):

    # distance threshold definition
    D = 2 * tile_diameter

    # Centroid locations
    n_rows = int((width + 1) / tile_diameter)
    n_columns = int((height + 1) / tile_diameter)

    # Compute patch centers
    x_patch_centers = jnp.linspace(tile_diameter / 2 - 1, width - tile_diameter / 2, n_rows)
    y_patch_centers = jnp.linspace(tile_diameter / 2 - 1, height - tile_diameter / 2, n_columns)

    # Meshgrid to create all combinations of x and y patch centers
    x_centers, y_centers = jnp.meshgrid(x_patch_centers, y_patch_centers)
    patch_centers = jnp.stack([x_centers.ravel(), y_centers.ravel()], axis=1)  # shape: (n_patches, 2)

    # Function to compute the Euclidean distance for a given patch center
    def compute_distances(patch_center, points):
        return jnp.linalg.norm(points - patch_center, axis=1)

    pixel_xy_coords = jnp.array(jnp.meshgrid(jnp.arange(width), jnp.arange(height))).T.reshape(-1, 2)

    # Use vmap to vectorize over the patch centers
    batched_distances = jax.vmap(compute_distances, in_axes=(0, None))(patch_centers, pixel_xy_coords)

    # Option 1: Using the circle area (more precise)
    # max_pixels_per_patch = int(jnp.ceil(jnp.pi * D**2))
    # circle area might be too restrictive, so we use a more relaxed version
    max_pixels_per_patch = int(jnp.ceil(3.2 * D**2))

    # Option 2: Using the bounding square (more conservative)
    # max_pixels_per_patch = int((jnp.floor(2 * D) + 1) ** 2)

    # Define function to get indices
    def get_patch_indices(distances, threshold):
        return jnp.argwhere(distances < threshold, size=max_pixels_per_patch, fill_value=-1).flatten()

    get_patch_indices_vmap = jax.vmap(get_patch_indices, in_axes=(0, None))

    # Apply the function
    pixel_indices_per_patch = get_patch_indices_vmap(batched_distances, D)  # shape: (n_patches, max_pixels_per_patch)

    # pixel-weights per group
    h = jnp.where(batched_distances < D, jnp.exp(-jnp.square(batched_distances) / (2 * (tile_diameter / 2) ** 2)), 0.0)
    H_weights = spm_dir_norm(h)  # normalize across groups

    # Filter the weights to only include the valid pixels per patch
    H_weights = vmap(lambda h_p, indices_p: h_p[indices_p], in_axes=(0, 0))(H_weights, pixel_indices_per_patch)

    # Get number of valid pixel indices per patch, which can be used to ignore the placeholder -1 values when indexing from the data later on
    valid_counts = jnp.sum(batched_distances < D, axis=1)

    # Create mask_matrix
    indices = jnp.arange(max_pixels_per_patch)
    valid_counts_expanded = valid_counts[:, None]
    mask_per_patch = indices < valid_counts_expanded  # shape: (num_patches, max_pixels_per_patch)

    G = pixel_indices_per_patch
    # create M, which should be a (num_patches, 2) matrix of the centroid of all the patches, calculated using the pixel indices that belong to that patch
    x_y_coords_per_patch = pixel_xy_coords[pixel_indices_per_patch]  # Shape: (num_patches, max_pixels_per_patch, 2)

    # Apply mask
    mask_matrix_expanded = mask_per_patch[..., None]  # Shape: (num_patches, max_pixels_per_patch, 1)
    x_y_coords_per_patch_masked = x_y_coords_per_patch * mask_matrix_expanded  # Invalid coords set to zero

    # Sum coordinates
    sum_coords_per_patch = jnp.sum(x_y_coords_per_patch_masked, axis=1)  # Shape: (num_patches, 2)

    # Compute mean coordinates
    valid_counts_safe = jnp.where(valid_counts_expanded > 0, valid_counts_expanded, 1)
    M = sum_coords_per_patch / valid_counts_safe  # Shape: (num_patches, 2)

    return G, M, H_weights, valid_counts


def spm_tile(width: int, height: int, n_copies: int, tile_diameter: int = 32):
    """
    Grouping into a partition of non-overlapping outcome tiles
    This routine identifies overlapping groups of pixels, returning their
    mean locations and a cell array of weights (based upon radial Gaussian
    basis functions) for each group. In other words, the grouping is based
    upon the location of pixels; in the spirit of a receptive field afforded
    by a sensory epithelium.Effectively, this leverages the conditional
    independencies that inherit from local interactions; of the kind found in
    metric spaces that preclude action at a distance.

    Args:
        L: list of indices
        width: width of the image
        height: height of the image
        tile_diameter: diameter of the tiles
    Returns:
        G: outcome indices
        M: (mean) outcome location
        H: outcome weights
    """

    def distance(x, y):
        return jnp.sqrt(((x - y) ** 2).sum())

    def flatten(l):
        return [item for sublist in l for item in sublist]

    # Centroid locations
    n_rows = int((width + 1) / tile_diameter)
    n_columns = int((height + 1) / tile_diameter)
    x = jnp.linspace(tile_diameter / 2 - 1, width - tile_diameter / 2, n_rows)
    y = jnp.linspace(tile_diameter / 2 - 1, height - tile_diameter / 2, n_columns)

    pixel_indices = n_copies * [jnp.array(jnp.meshgrid(jnp.arange(width), jnp.arange(height))).T.reshape(-1, 2)]
    pixel_indices = jnp.concatenate(pixel_indices, axis=0)

    h = [[[] for _ in range(n_columns)] for _ in range(n_rows)]
    g = [[None for _ in range(n_columns)] for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_columns):
            pos = jnp.array([x[i], y[j]])
            distance_evals = vmap(lambda x: distance(x, pos))(pixel_indices)

            ij = jnp.argwhere(distance_evals < (2 * tile_diameter)).squeeze()
            h[i][j] = jnp.exp(-jnp.square(distance_evals) / (2 * (tile_diameter / 2) ** 2))
            g[i][j] = ij

    G = flatten(g)
    h_flat = flatten(h)

    num_groups = n_rows * n_columns

    # weighting of groups
    h_matrix = jnp.stack(h_flat)  # [num_groups, n_pixels_per_group)
    h = spm_dir_norm(h_matrix)  # normalize across groups

    H_weights = [h[g_i, G[g_i]] for g_i in range(num_groups)]

    M = jnp.zeros((num_groups, 2))
    for g_i in range(num_groups):
        M = M.at[g_i, :].set(pixel_indices[G[g_i], :].mean(0))

    return G, M, H_weights


def spm_space(L: Array, dx: int = 2):
    """
    This function takes a set of modalities and their
    spatial coordinates and decimates over space into a compressed
    set of modalities, and assigns the previous modalities
    to the new set of modsalities.

    Args:
        L (Array): num_modalities x 2
        dx (int): spatial decimation factor
    Returns:
        G (List[Array[int]]):
            outcome indices mapping new modalities indices to
            previous modality indices
    """

    # this is the second case (skipping if isvector(L))
    # locations
    Nl = L.shape[0]
    unique_locs = jnp.unique(L, axis=0)
    Ng = unique_locs.shape[0]
    Ng = jnp.ceil(jnp.sqrt(Ng / (dx * dx)))
    if Ng == 1:
        G = jnp.arange(Nl)
        return [G]

    # decimate locations
    x = jnp.linspace(jnp.min(L[:, 0]), jnp.max(L[:, 0]), int(Ng))
    y = jnp.linspace(jnp.min(L[:, 1]), jnp.max(L[:, 1]), int(Ng))
    R = jnp.fliplr(jnp.array(jnp.meshgrid(x, y)).T.reshape(-1, 2))

    # nearest (reduced) location
    closest_loc = lambda loc: jnp.argmin(jnp.linalg.norm(R - loc, axis=1))
    g = vmap(closest_loc)(L)

    # grouping partition
    G = []

    # these two lines do the equivalent of u = unique(g, 'stable') in MATLAB
    _, unique_idx = jnp.unique(g, return_index=True)
    u = g[jnp.sort(unique_idx)]
    for i in range(len(u)):
        G.append(jnp.argwhere(g == u[i]).squeeze())

    return G


def spm_group(S, n=2, num_modalities_per_pixel=1):
    """
    Grouping into a partition of non-overlapping outcome tiles.
        S = shape as [num_rows, num_cols]
        n = block size
    """

    # Function to calculate the flattened indices for a block starting at (i, j)
    def block_flattened_indices(i, j, block_height, block_width):
        rows = jnp.arange(block_height) + i  # Handle any block height
        cols = jnp.arange(block_width) + j  # Handle any block width
        # Create a grid of row and column indices
        row_col_pairs = jnp.array(jnp.meshgrid(rows, cols, indexing="ij")).reshape(2, -1).T
        # Flatten the row and column indices to 1D indices
        flat_indices = row_col_pairs[:, 0] * S[1] + row_col_pairs[:, 1]
        return flat_indices

    # Regular nxn blocks (fully fitting)
    block_start_rows = jnp.arange(0, S[0] - n + 1, n)
    block_start_cols = jnp.arange(0, S[1] - n + 1, n)
    # Create a grid of block starting positions for regular blocks
    block_positions = jnp.array(jnp.meshgrid(block_start_rows, block_start_cols, indexing="ij")).reshape(2, -1).T
    # Vectorize the flattened index calculation for each regular nxn block
    regular_block_indices = vmap(lambda pos: block_flattened_indices(pos[0], pos[1], n, n))(block_positions)
    G = [regular_block_indices[i] for i in range(len(regular_block_indices))]
    # Handle remainder along the bottom (rows)
    if S[0] % n != 0:
        bottom_start_row = S[0] - (S[0] % n)
        for j in range(0, S[1] - n + 1, n):
            G.append(block_flattened_indices(bottom_start_row, j, S[0] % n, n))
    # Handle remainder along the right (columns)
    if S[1] % n != 0:
        right_start_col = S[1] - (S[1] % n)
        for i in range(0, S[0] - n + 1, n):
            G.append(block_flattened_indices(i, right_start_col, n, S[1] % n))
    # Handle the bottom-right corner block (if both S[0] and S[1] are not divisible by n)
    if S[0] % n != 0 and S[1] % n != 0:
        G.append(block_flattened_indices(S[0] - (S[0] % n), S[1] - (S[1] % n), S[0] % n, S[1] % n))
    if num_modalities_per_pixel > 1:
        nmpx = num_modalities_per_pixel
        G_ext = []
        for g in G:
            a = []
            for gg in g:
                a.append(jnp.arange(gg * nmpx, gg * nmpx + nmpx, 1))
            G_ext.append(jnp.concatenate(a))
        G = G_ext
    return G


def spm_time(T, d):
    """
    Grouping into a partition of non-overlapping sequences
    Args:
    T (int): total number of the timesteps
    d (int): number timesteps per partition

    Returns:
    list: A list of partitions with non-overlapping sequences
    """
    t = []
    for i in range(T // d):
        t.append(jnp.arange(d) + (i * d))
    return t


def spm_unique(a):
    """
    Fast approximation by simply identifying unique locations in a
    multinomial statistical manifold, after discretising to probabilities of
    zero, half and one (using jax.numpy's unique and fix operators).

    Args:
        a: array (n, x)
    Returns:
        indices of unique x'es
    """

    # Discretize to probabilities of zero, half, and one
    # 0 to 0.5 -> 0, 0.5 to 1 -> 1, 1 -> 2
    o_discretized = jnp.fix(2 * a)

    # Find unique rows -- this however needs to be changed to mimic the behavior of unique(o_discretized, 'stable')
    _, j = jnp.unique(o_discretized, return_inverse=True, axis=0)

    # suddenly j no longer has trailing 1 dimension?!
    if j.shape[-1] == 1:
        j = j.squeeze(axis=1)
    return j


def create_group_partitioning(locations_matrix=None, size=None, num_controls=0, dx: int = 2, max_levels: int = 8):
    """
    Create group partitioning across levels
    Args:
        locations_matrix (array): (num_modalities, 2) (optional)
        size (tuple): (width, height, [num_modalities]) of the image (optional)
        num_controls (int): number of control modalities
        dx (int): spatial decimation factor
        max_levels (int): maximum number of levels in the hierarchy
    Returns:
        RG (list): group partitioning across levels
        LG (list): locations matrix across levels
    """

    RG, LG = [], []

    num_modalities_per_pixel = 1
    if size is not None and len(size) > 2:
        num_modalities_per_pixel = size[2]
        size = size[:2]
    for n in range(max_levels):

        if locations_matrix is None:
            if size is None:
                raise ValueError("Either locations_matrix or size must be provided")
            G = spm_group(size, dx, num_modalities_per_pixel)
        else:
            G = spm_space(locations_matrix, dx=dx)
        if n == 0 and num_controls > 0:
            # prepend action one_hots to first group
            G = [g + num_controls for g in G]
            G[0] = jnp.concatenate((jnp.arange(num_controls), G[0]))

        RG.append(G)
        LG.append(locations_matrix)

        # coarse grain locations
        if locations_matrix is not None:
            coarse_locations = []

            for g in range(len(G)):
                # append twice (for initial state and path)
                coarse_locations.append(jnp.mean(locations_matrix[G[g]], axis=0))
                coarse_locations.append(jnp.mean(locations_matrix[G[g]], axis=0))

            locations_matrix = jnp.stack(coarse_locations)
        else:
            size = [math.ceil(size[0] / dx), math.ceil(size[1] / dx)]
            num_modalities_per_pixel = 2

        # Terminate the hierarchical spatial coarse-graining if there is only one group at the current level
        if len(G) == 1:
            break

    return RG, LG


def spm_structure_learn_or_merge(observations, dt: int = 2, A: List[Array] = None, B: Array = None):
    """
    Either learns As and B tensor from a sequence of observations for a particular group,
    or updates As and B tensor from a sequence of observations for a particular group
    Args:
        observations (array): (num_modalities, num_steps, num_obs)
        dt (int): number of timesteps to coarse grain
        A (list): likelihood tensors (optional)
        B (array): transition tensor (optional)
        merge (bool): whether to merge or learn
    Returns:
        a (list): likelihood tensors
        b (array): transition tensor
        init_states : initial states for coarse sequence of observations
        paths: path for each coarse sequence of observations
    """

    # get the observations for the current group of patch-specific indices, and then reshape so that time dimension is first, and lagging dimension is (num_modalities x num_outcomes_per_modality).
    # So for each row, this last axis correponds to a potentially new state, which is simply a configuration of observation levels across modalities
    num_modalities, num_steps, num_obs = observations.shape
    o = jnp.moveaxis(observations, 1, 0).reshape(num_steps, -1)

    if A is not None:
        # Stack each existing modality's A tensor along the observation dimension to recover the 'known' unique configurations
        # Transpose to put the time dimension first

        # The old A might have less state bins as the new observations
        # coming from the lower level already expanded
        # zero pad the existing patterns before concatenating
        padded = jtu.tree_map(lambda x: jnp.pad(x, ((0, num_obs - x.shape[0]), (0, 0))), A)
        existing_patterns = jnp.concatenate(padded, axis=0).T

        # Find the number of unique latent states that were already present in the existing structure

        Ns_existing = A[0].shape[1]

        # Concatenate the existing likelihood mappings with the new observations along the time dimension to identify new configurations
        o_concat = jnp.concatenate([existing_patterns, o], axis=0)
    else:
        Ns_existing = 0
        o_concat = o

    # print("Patterns shape", o_concat.shape)
    num_patterns, pattern_size = o_concat.shape

    # # convert one hots to integers for more efficient unique
    o_concat_int = jnp.argmax(o_concat.reshape(num_patterns, num_modalities, num_obs), axis=-1)
    # print("Patterns as int shape", o_concat_int.shape)

    # Pad with -1 to have more efficient unique - we pad to the nearest multiple of 1024
    # Add 1 in the front so we know we can cut out the first found pattern
    max_size = int(jnp.ceil((num_patterns + 1) / 1024) * 1024)
    o_concat_padded = jnp.pad(
        o_concat_int, ((1, max_size - 1 - o_concat_int.shape[0]), (0, 0)), mode="constant", constant_values=-1
    )

    # Check which configurations are unique using jnp.unique (TODO jit with fixed return size?)
    m, n, j = jnp.unique(o_concat_padded, return_index=True, return_inverse=True, axis=0)

    # remove -1-ed patterns
    # there is one at the front
    j = j[1 : num_patterns + 1] - 1
    n = n[1:] - 1

    # Define total number of states (old and new)
    Ns = len(jnp.unique(j))

    # Only keep new ones to expand B later on
    j = j[Ns_existing:].squeeze()

    # Now create new A tensors
    n_sort_idx = jnp.argsort(n)
    A_expanded_flat = o_concat[n[n_sort_idx]]
    A_expanded = jnp.split(A_expanded_flat.T, num_modalities, axis=0)

    # Learn or update transition tensor B to include transitions to/from the new states

    # Old B dimensions: (Ns_existing, Ns_existing, num_paths)
    num_existing_paths = 1 if B is None else B.shape[2]

    # Create new B tensor for the updated states (new rows/cols for new states)
    B_expanded = jnp.zeros((Ns, Ns, num_existing_paths))

    if B is not None:
        # Fill in the existing transitions, if they exist
        B_expanded = B_expanded.at[:Ns_existing, :Ns_existing, :].set(B)

    # convert `j` which is a timeseries of codebook indices (relative to `m`) to a timeseries of indices in A_expanded_flat,
    # which are the indices of the unique states in the new sttate space (columns of A, rows/columns of B)
    # convert each element `t` of `j[t]` to the index of that state in n_sort_idx
    j = vmap(lambda jt: jnp.argwhere(n_sort_idx == jt, size=1).squeeze())(j)

    # TODO take into account dt here as well?
    for t in range(len(j) - 1):
        if not jnp.any(B_expanded[j[t + 1], j[t], :]):
            u = jnp.where(~jnp.any(B_expanded[:, j[t], :], axis=0))[0]
            if len(u) == 0:
                B_expanded = jnp.concatenate((B_expanded, jnp.zeros((Ns, Ns, 1))), axis=2)
                B_expanded = B_expanded.at[j[t + 1], j[t], -1].set(1)
            else:
                B_expanded = B_expanded.at[j[t + 1], j[t], u[0]].set(1)

    # Update initial states and paths
    init_states = nn.one_hot(j, num_classes=Ns, axis=-1)

    # paths
    paths = vmap(lambda t: B_expanded[j[t + 1], j[t], :])(jnp.arange(len(j) - 1))

    return A_expanded, B_expanded, init_states, paths


def spm_mb_structure_learning(
    observations,
    locations_matrix=None,
    size=None,
    num_controls=0,
    dx: int = 2,
    dt: int = 2,
    max_levels: int = 8,
    agents: List[Agent] = None,
    RG: List[Array] = None,
):
    """

    Args:
        observations (array): (num_modalities, time, num_obs)
        locations_matrix (array): (num_modalities, 2) (optional)
        size (tuple): (width, height) of the image (optional)
        num_controls (int): number of control modalities
        dx (int): spatial decimation factor
        dt (int): temporal resampling factor
        max_levels (int): maximum number of levels in the hierarchy
        agents (list): list of Agent objects (optional)
        RG (list): group partitioning across levels (optional)
    """

    """First, create group partitioning (RG) across levels if not already present """

    # If there's no RG, create both it and LG (the list of hierarchical location-matrices)
    LG = None
    if RG is None:
        RG, LG = create_group_partitioning(locations_matrix, size, num_controls, dx=dx, max_levels=max_levels)

    num_identified_levels = len(RG)

    if agents is None:
        agents = []
        merge = False
    else:
        merge = True

    observations = [observations]

    for n in range(num_identified_levels):

        T = spm_time(observations[n].shape[1], dt)

        # A tensors for each group
        A = [None] * observations[n].shape[0]
        A_dependencies = [None] * observations[n].shape[0]

        # B tensors for each group
        B = []
        # initial states per group
        X = []
        # initial paths per group
        P = []

        G_n = RG[n]
        for g in range(len(G_n)):
            t1 = time.time()
            # The 0-indexing on each likelihood or transition tensor is to account for the (1,) batch dim of jax-pymdp
            A_existing = [agents[n].A[m_i][0] for m_i in G_n[g]] if merge else None
            B_existing = agents[n].B[g][0] if merge else None
            a, b, init_states, paths = spm_structure_learn_or_merge(
                observations[n][G_n[g]], dt, A=A_existing, B=B_existing
            )

            # TODO should we normalize b + small constant to make
            # uniform transition on unseen s0, path combos?
            # this yields non-one-hots in the A tensors though :-/
            # b = b + 1e-8
            # b = b / jnp.sum(b, axis=0, keepdims=True)

            # a, b, init_states, paths = spm_structure_fast(observations[n][G_n[g]], dt)

            for m_relative, m_idx in enumerate(G_n[g]):
                A[m_idx] = a[m_relative]
                A_dependencies[m_idx] = [g]

            B.append(b)
            X.append(init_states)
            P.append(paths)
            t2 = time.time()
            print("structure learn", "level", n, "group", g, "time", t2 - t1)

        # TODO update existing Agent objects or just replace them with new ones?
        # TODO right now an Agent's policy would be a combination of paths over all groups, which blows up
        # currently avoiding this by passing empty policies but needs to revisit if we want to do planning
        pdp = Agent(A=A, B=B, A_dependencies=A_dependencies, apply_batch=True, onehot_obs=True, policies=jnp.empty([0]))
        if merge:
            agents[n] = pdp
        else:
            agents.append(pdp)

        # observation dim size for next level
        ndim = max(max(pdp.num_states), max(pdp.num_controls))
        # we have to gather initial state and path as observations for the next level
        observations.append(jnp.zeros((len(G_n) * 2, len(T), ndim)))

        # set initial states and paths for the next level
        for t in range(len(T)):
            for g in range(len(G_n)):
                # initial states (even indices)
                observations[n + 1] = observations[n + 1].at[2 * g, t, : X[g].shape[-1]].set(X[g][T[t][0], :])
                # paths (odd indices)
                observations[n + 1] = observations[n + 1].at[2 * g + 1, t, : P[g].shape[-1]].set(P[g][T[t][0], :])

    return agents, RG, LG


def pad_to_same_size(arrays: list):
    """
    Pad arrays to the same size along the last dimension
    """
    max_size = max([a.shape[-1] for a in arrays])
    padded = [jnp.pad(a, ((0, 0), (0, max_size - a.shape[-1]))) for a in arrays]
    return padded


def infer_old(agents, observations, priors=None):
    """
    Infer the top level state given the observations and priors.
    Some observations can be masked out with uniform vectors if not yet fully observed.
    When priors is None, we use the (uniform) priors in the agent's D tensors.

    Args:
        agents (list): list of n agents, n the number of levels in the hieararchy
        observations (array): (num_modalities, n**T, num_observation_bins): observations of the lowest level
        priors (list): list of n priors, n the number of levels in the hierarchy

    Returns:
        Inferred top level state
    """
    if priors is None:
        # TODO broadcasting priors based on the number of expected timesteps required for inferring 1 top level state
        # currently assuming T=2
        # TODO not every agent has the same number of states, so we need to pad the priors to the same size
        priors = []
        for n, agent in enumerate(agents):
            Ds = jnp.stack(pad_to_same_size(agent.D))
            priors.append(
                jnp.broadcast_to(
                    Ds,
                    (len(agent.D), 2 ** (abs(n - len(agents) + 1)), Ds.shape[-1]),
                )
            )

    for n in range(len(agents)):
        # infer states for each observation

        # convert observations array to a list or arrays (modalities), and make time the batch dimension to vmap over
        # TODO not considering actual batch dimension here
        o = [observations[i, :, :] for i in range(observations.shape[0])]

        priors_n = [priors[n][i, :, : agents[n].D[i].shape[-1]] for i in range(priors[n].shape[0])]

        # doesn't auto-broadcast to batch, call inference method vmapped ourselves
        infer = partial(
            run_factorized_fpi,
            A_dependencies=agents[n].A_dependencies,
            num_iter=agents[n].num_iter,
        )

        if priors_n[0].shape[0] != o[0].shape[0]:
            # longer timesequence of observations is given, need to broadcast priors
            k = o[0].shape[0] // priors_n[0].shape[0]
            priors_n = jtu.tree_map(
                lambda x: jnp.broadcast_to(x, (k, x.shape[0], x.shape[1])).reshape(o[0].shape[0], x.shape[1]),
                priors_n,
            )

        qs = vmap(infer, in_axes=(None, 0, 0))(jtu.tree_map(lambda x: x[0], agents[n].A), o, priors_n)

        if n == len(agents) - 1:
            # reached the top level, no more paths to infer?, return this? (and only this?)
            return qs

        # now infer paths for each subsequence of T (= 2)
        q0 = jtu.tree_map(lambda x: x[::2, :], qs)
        q1 = jtu.tree_map(lambda x: x[1::2, :], qs)

        D = q0
        E = []
        # TODO make this a method instead of this loop?
        action_marginal_fn = lambda b, qs: factor_dot(b, qs, keep_dims=(2,))
        for g in range(len(agents[n].B)):
            action_marginal = vmap(action_marginal_fn, in_axes=(None, 0))(b[0], [q1[g], q0[g]])
            # needs to be normalize, as B is only normalized for the to_state dim
            action_marginal = action_marginal + 1e-6
            action_marginal = action_marginal / jnp.sum(action_marginal, axis=-1)[:, jnp.newaxis]
            E.append(action_marginal)

        ndim = max([d.shape[-1] for d in D] + [e.shape[-1] for e in E])

        # pad D and E to be same trailing dim
        D = jtu.tree_map(lambda x: jnp.pad(x, ((0, 0), (0, ndim - x.shape[-1]))), D)
        E = jtu.tree_map(lambda x: jnp.pad(x, ((0, 0), (0, ndim - x.shape[-1]))), E)

        interleaved = E + D
        interleaved[::2] = D
        interleaved[1::2] = E
        observations = jnp.asarray(interleaved)


def infer(agents, observations, priors=None):
    """
    Infer the top level state given the observations and priors.
    Some observations can be masked out with uniform vectors if not yet fully observed.
    When priors is None, we use the (uniform) priors in the agent's D tensors.

    Args:
        agents (list): list of n agents, n the number of levels in the hieararchy
        observations (array): (num_modalities, n**T, num_observation_bins): observations of the lowest level
        priors (list): list of n priors, n the number of levels in the hierarchy

    Returns:
        Inferred top level state
    """
    if priors is None:
        # TODO broadcasting priors based on the number of expected timesteps required for inferring 1 top level state
        # currently assuming T=2

        # TODO not every agent has the same number of states, so we need to pad the priors to the same size
        priors = []
        for n, agent in enumerate(agents):
            Ds = jnp.stack(pad_to_same_size(agent.D))
            priors.append(
                jnp.broadcast_to(
                    Ds,
                    (len(agent.D), 2 ** (abs(n - len(agents) + 1)), Ds.shape[-1]),
                )
            )

    for n, agent in enumerate(agents):
        # infer states for each observation

        # Create a vectorized version of A_dependencies
        vectorized_dependencies = jnp.array(agent.A_dependencies).squeeze()

        # create a list of indices, one for each factor, that can be used to index into the log-likelihoods for each factor across modalities
        f_to_m_idx = jtu.tree_map(lambda f: jnp.where(vectorized_dependencies == f)[0], list(range(agent.num_factors)))

        loq_qs_all = update_posterior_over_states_optimized(observations, agent.A, priors[n], f_to_m_idx)

        # Split the result into a list of log_qs, one for each factor
        log_qs_padded = jnp.split(loq_qs_all, agent.num_factors, axis=0)

        # Now convert the log qs back to qs through softmaxing and remembering to truncate the state dimension before doing so
        qs = jtu.tree_map(lambda x, ns: nn.softmax(x[0, ..., :ns], axis=-1), log_qs_padded, agent.num_states)

        if n == len(agents) - 1:
            # reached the top level, no more paths to infer?, return this? (and only this?)
            return qs

        # now infer paths for each subsequence of T (= 2)
        q0 = jtu.tree_map(lambda x: x[::2, :], qs)
        q1 = jtu.tree_map(lambda x: x[1::2, :], qs)

        D = q0
        E = []
        # TODO make this a method instead of this loop?
        action_marginal_fn = lambda b, qs: factor_dot(b, qs, keep_dims=(2,))
        for g, b in enumerate(agent.B):
            action_marginal = vmap(action_marginal_fn, in_axes=(None, 0))(b[0], [q1[g], q0[g]])
            # needs to be normalize, as B is only normalized for the to_state dim
            action_marginal = action_marginal + 1e-6
            action_marginal = action_marginal / jnp.sum(action_marginal, axis=-1)[:, jnp.newaxis]
            E.append(action_marginal)

        ndim = max([d.shape[-1] for d in D] + [e.shape[-1] for e in E])

        # pad D and E to be same trailing dim
        D = jtu.tree_map(lambda x: jnp.pad(x, ((0, 0), (0, ndim - x.shape[-1]))), D)
        E = jtu.tree_map(lambda x: jnp.pad(x, ((0, 0), (0, ndim - x.shape[-1]))), E)

        # concatenate D and E
        interleaved = E + D
        interleaved[::2] = D
        interleaved[1::2] = E
        observations = jnp.asarray(interleaved)


@jit
def update_posterior_over_states_optimized(observations, A, priors, f_to_m_idx):

    # Stack all the A matrices across modalities, ensuring to pad their lagging dimensions to the same size
    stacked_As = jnp.stack(pad_to_same_size(jtu.tree_map(lambda x: x[0], A)))

    # move the time dimension to the front so everything is broadcasted across time
    o = jnp.moveaxis(observations, 1, 0)

    # Compute the log likelihood of all observations, for all factors AND modalities at the same time
    ell = (o[..., None] * log_stable(stacked_As)).sum(axis=2)

    # Sum the log likelihoods for each factor across the modalities driven by the same factor
    ell_all_factors = jtu.tree_map(lambda idx: ell[:, idx, :].sum(axis=1), f_to_m_idx)

    # Add the log of the priors to the log likelihoods (and recall that each log_qs has extra state dimensions)
    loq_qs_all = jnp.stack(ell_all_factors) + log_stable(priors)
    return loq_qs_all


def predict(agents, D=None, E=None, num_steps=1):
    """
    Infer the top level state given the observations and priors.
    Some observations can be masked out with uniform vectors if not yet fully observed.
    When priors is None, we use the (uniform) priors in the agent's D tensors.

    Args:
        agents (list): list of n agents, n the number of levels in the hieararchy
        D (list): list of initial state factors at the top level
        E (list): list of path at the top level

    Returns:
        Predicted states and observations for all levels in the hierarchy
    """

    n = len(agents) - 1

    beliefs = [
        None,
    ] * (len(agents))
    observations = [
        None,
    ] * (len(agents))

    # add time dimension, so qs[f] has shape (batch_dim, 1, num_states)
    if D is None:
        D = agents[n].D
    qs = jtu.tree_map(lambda x: jnp.expand_dims(x, 1), D)

    # unroll highest level
    expected_state = partial(compute_expected_state, B_dependencies=agents[n].B_dependencies)

    for _ in range(num_steps):
        # extract the last timestep, such tthat qs_last[f] has shape (batch_dim, num_states)
        qs_last = jtu.tree_map(lambda x: x[:, -1, ...], qs)
        # this computation of the predictive prior is correct only for fully factorised Bs.
        pred = vmap(expected_state)(qs_last, agents[n].B, E)
        # pred, qs  = agents[n].update_empirical_prior(E, qs)
        # stack in time dimension
        qs = jtu.tree_map(
            lambda x, y: jnp.concatenate([x, jnp.expand_dims(y, 1)], 1),
            qs,
            pred,
        )
    # qs[f] will have shape (batch_dim, num_steps+1, num_states)
    qs_stacked = jtu.tree_map(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2])), qs)
    beliefs[n] = qs_stacked

    # generate outcomes of highest level
    qo = compute_expected_obs_fast(qs_stacked, agents[n].A, agents[n].A_dependencies)
    qo = jtu.tree_map(lambda x: x.squeeze(1), jnp.split(qo, agents[n].num_modalities, axis=1))

    observations[n] = qo

    while n > 0:
        qo = observations[n]

        n -= 1
        agent = agents[n]

        # split this in initial state "D" and path "E"
        DD = qo[::2]
        for i in range(len(agent.B)):
            DD[i] = DD[i][:, : agent.B[i].shape[1]]
        EE = jtu.tree_map(lambda x: jnp.argmax(x, axis=1), qo[1::2])

        # unroll path and get beliefs qs at level n
        # TODO repeat if dt > 2
        expected_state = partial(compute_expected_state, B_dependencies=agents[n].B_dependencies)
        B_stacked = jtu.tree_map(
            lambda x: jnp.broadcast_to(x, (DD[0].shape[0], x.shape[1], x.shape[2], x.shape[3])),
            agents[n].B,
        )
        pred = vmap(expected_state)(DD, B_stacked, EE)

        # stack in time dimension
        qs = jtu.tree_map(
            lambda x, y: jnp.concatenate([jnp.expand_dims(x, 1), jnp.expand_dims(y, 1)], 1),
            DD,
            pred,
        )

        qs_stacked = jtu.tree_map(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2])), qs)
        beliefs[n] = qs_stacked

        # now generate outcomes of level n
        qo = compute_expected_obs_fast(qs_stacked, agents[n].A, agents[n].A_dependencies)
        qo = jtu.tree_map(lambda x: x.squeeze(1), jnp.split(qo, agents[n].num_modalities, axis=1))

        observations[n] = qo

    observations = [jnp.asarray(o) for o in observations]
    beliefs = [jnp.stack(pad_to_same_size(b)) for b in beliefs]
    return observations, beliefs


@jit
def compute_expected_obs_fast(qs, A, A_dependencies):
    """
    Compute the expected observations given the beliefs at a certain level
    """

    vectorized_dependencies = jnp.array(A_dependencies).squeeze()
    stacked_As = jnp.stack(pad_to_same_size(jtu.tree_map(lambda x: x[0], A)))
    qs_stacked_across_factors = jnp.moveaxis(jnp.stack(pad_to_same_size(qs)), 1, 0)
    qs_selected = qs_stacked_across_factors[:, vectorized_dependencies, :]
    qo = contract("MOS,TMS->TMO", stacked_As, qs_selected, backend="jax")

    return qo


def learn_transitions(qs, actions=None, B_dependencies=None, pB=None):
    """
    qs: a list of jax.numpy arrays of shape [(n_time, n_states) for f in factors]
    """

    n_states = qs[0].shape[-1]
    n_time = qs[0].shape[0] - 1
    print(f"Fitting for {n_states} states and {n_time} timesteps.")

    if pB is None:
        pB = [1e-3 * jnp.ones((n_states, n_states, 1)) for _ in range(len(qs))]

    if actions is None:
        # No actions = the zero action at each timestep
        actions = jnp.zeros((n_time, len(qs)))

    if B_dependencies is None:
        B_dependencies = [[i] for i in range(len(qs))]

    # Map the qs's to something that can be used for learning.
    beliefs = []

    # take all the states but the last one
    qs_f_prev = jtu.tree_map(lambda x: x[:-1], qs)
    # take all the states but the first one
    qs_f = jtu.tree_map(lambda x: x[1:], qs)

    for f in range(len(pB)):

        # Extract factor
        q_f = jnp.array(qs_f[f].tolist())
        q_prev_f = [jnp.array(qs_f_prev[fi].tolist()) for fi in B_dependencies[f]]
        beliefs.append([q_f, *q_prev_f])

    qB, E_qB = update_state_transition_dirichlet(pB, beliefs, actions, num_controls=[b.shape[-1] for b in pB], lr=1)

    norm = lambda x: jnp.divide(
        jnp.clip(x, a_min=1e-8),
        jnp.clip(x, a_min=1e-8).sum(axis=1, keepdims=True),
    )

    E_qB = jtu.tree_map(norm, qB)

    return qB, E_qB
