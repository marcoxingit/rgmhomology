import mediapy
import seaborn as sns
import jax
import jax.numpy as jnp


def show_gameplay(infos, size=(160, 160), fps=20, codec="gif"):
    observations = [info["observation"] for info in infos]

    mediapy.show_videos(
        {"play": observations},
        width=size[1],
        height=size[0],
        fps=fps,
        codec=codec,
    )


def analyze_gameplay(infos, rgm_agent, size=(160, 160), fps=5, codec="gif"):
    observations = []
    reconstructions = []
    preferences = []
    attention = []

    cmap = sns.color_palette("coolwarm", as_cmap=True)

    for info in infos:
        observation = info["observation"]
        observations.append(observation)

        reconstructions.append(rgm_agent.to_rgb(rgm_agent.ogm.reconstruct(info["qs"])))
        t = info["t"]
        preferences.append(rgm_agent.to_rgb(info["preferred_outcomes"])[t : t + 1])

        mask = info["mi"].reshape(rgm_agent.rgm.height, rgm_agent.rgm.width)
        mask = jax.image.resize(mask, (observation.shape[:2]), method="nearest")
        mask = cmap(mask)
        attention.append(mask[:, :, :3])

    reconstructions = jnp.concatenate(reconstructions, axis=0)
    preferences = jnp.concatenate(preferences, axis=0)

    mediapy.show_videos(
        {"play": observations, "reconstructions": reconstructions, "preferences": preferences, "attention": attention},
        width=size[1],
        height=size[0],
        fps=fps,
        codec=codec,
    )


def compare_ogm_reconstruction(observation, rgm_agent, prior=None):
    one_hots = rgm_agent.to_one_hot(jnp.expand_dims(observation, axis=0))[:, 0, :]
    if prior is None:
        prior = rgm_agent.ogm.agent.D
    qs = rgm_agent.ogm.infer_states(one_hots, empirical_prior=prior)
    qo = rgm_agent.ogm.reconstruct(qs)
    rgb = rgm_agent.to_rgb(qo)
    mediapy.compare_images([observation, rgb[0]])


def show_rgm_generation(state_idx, rgm_agent, size=(160, 160), fps=2, codec="gif"):
    videos = {}
    if not isinstance(state_idx, list):
        state_idx = [state_idx]
    for idx in state_idx:
        one_hot = jnp.zeros([1, rgm_agent.rgm.agents[-1].B[0].shape[1]])
        one_hot = one_hot.at[0, idx].set(1.0)
        qs = [one_hot]
        qo_idx = rgm_agent.rgm.reconstruct(qs)
        rgb = rgm_agent.to_rgb(qo_idx)
        videos[str(idx)] = rgb
    mediapy.show_videos(videos, width=size[1], height=size[0], fps=fps, codec=codec)


def show_rgm_inferred_state(observations, rgm_agent, size=(160, 160), fps=2, codec="gif"):
    one_hots = rgm_agent.to_one_hot(observations)
    qs = rgm_agent.rgm.infer_states(one_hots)
    idx = jnp.argmax(qs[0], axis=-1)[0]
    val = qs[0][0, idx]
    key = f"{idx} ({val})"
    qo = rgm_agent.rgm.reconstruct(qs)
    rgb = rgm_agent.to_rgb(qo)
    mediapy.show_videos({key: rgb}, width=size[1], height=size[0], fps=fps, codec=codec)


def analyze_planning(info, rgm_agent, size=(160, 160), fps=1, codec="gif"):
    qs = info["qs"]
    t = info["t"]
    c = info["preferred_outcomes"][:, t : t + 1, :]
    q_pi, u, mi = rgm_agent.ogm.infer_policies(qs, c)

    rgb1 = rgm_agent.to_rgb(rgm_agent.ogm.reconstruct(qs))

    predictions = {}
    prefs = rgm_agent.to_rgb(c)
    predictions["preference"] = prefs

    for act in range(rgm_agent.ogm.n_actions):
        qs_next = rgm_agent.ogm.update_empirical_prior(jnp.array([[act]]), qs)
        rgb2 = rgm_agent.to_rgb(rgm_agent.ogm.reconstruct(qs_next))
        predictions[f"action {act} ({q_pi[act]:.2f})"] = jnp.concatenate([rgb1, rgb2])

    mediapy.show_videos(predictions, width=size[1], height=size[0], fps=fps, codec=codec)
    return q_pi, u, mi


def analyze_svd_reconstruction(observations, rgm_agent, size=(160, 160), fps=20, codec="gif"):
    one_hots = rgm_agent.to_one_hot(observations)
    rgb = rgm_agent.to_rgb(one_hots)
    mediapy.show_videos(
        {"ground truth": observations, "reconstruction": rgb}, width=size[1], height=size[0], fps=fps, codec=codec
    )
