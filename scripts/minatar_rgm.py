import mediapy
import jax.numpy as jnp

from rgm.rgm import RGM, RGMAgent
from atari.common import *

from argparse import ArgumentParser

import seaborn as sns
from matplotlib import colors


def render(observation):
    observation = observation[:, :, :4]
    cmap = sns.color_palette("cubehelix", observation.shape[-1])
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(observation.shape[-1] + 2)]
    norm = colors.BoundaryNorm(bounds, observation.shape[-1] + 1)
    numerical_state = jnp.amax(observation * jnp.reshape(jnp.arange(observation.shape[-1]) + 1, (1, 1, -1)), 2) + 0.5
    img = cmap(norm(numerical_state))
    return img[:, :, :3]


def to_one_hot(o):
    zero_mask = jnp.all(o == 0, axis=-1)  # shape of batch x time x 10 x 10
    arr = jnp.zeros((o.shape[0], o.shape[1], o.shape[2], o.shape[3], o.shape[4] + 1), dtype=jnp.float32)
    arr = arr.at[..., :4].set(o)
    arr = arr.at[zero_mask, 4].set(1)
    return arr


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-g", "--game", default="breakout")
    args = parser.parse_args()

    game = args.game
    env = make_game(game_config(args.game.capitalize(), ObservationType.MINI))

    steps = 0
    train_sequences = 0
    train_steps = 0

    observations = []
    actions = []
    rewards = []

    num_episodes = 128
    horizon = 128

    rgm = RGM(max_levels=8, n_bins=5, dx=2, size=(10, 10), action_range=(0, 4), svd=False)
    rgm_agent = RGMAgent(rgm)

    # for n in tqdm(range(num_episodes)):
    while steps < 1000:
        print(steps)

        acs = []
        os = []
        rs = []

        rgm_agent.reset()
        obs, info = env.reset()
        for i in range(horizon):
            o = to_one_hot(jnp.asarray([[obs]]))[0].reshape(-1, 100, 5).transpose(1, 0, 2)
            action = rgm_agent.act(o)
            action = int(action[0])
            if action == -1:
                # if rgm_agent returns -1, randomly sample an action
                action = env.action_space.sample()
            next_obs, reward, done, trunc, info = env.step(action)

            steps += 1

            acs.append([action])
            os.append(obs)
            rs.append(reward)
            obs = next_obs

            if done:
                achieved_reward = jnp.asarray(rs)
                r = jnp.where(achieved_reward > 0)[0]
                if len(r) > 0:
                    last_reward_idx = r[-1]
                    size = 8 * (last_reward_idx // 8) + 8
                    if len(os) >= size:
                        train_sequences += 1
                        train_steps += size
                        # we have a trajectory with some rewards, add to RGM?
                        o = to_one_hot(jnp.asarray([os]))[0].reshape(-1, 100, 5).transpose(1, 0, 2)
                        a = jnp.asarray(acs)
                        rgm.learn_structure(o, a)
                break

        # only train on 1 sequence for debugging
        # if rgm.agents is not None:
        #     break

        observations.append(os)
        actions.append(acs)
        rewards.append(rs)

    rgm.save("data/rgms/mini_" + args.game + "_rgm.npz")

    imgs = []
    for i in range(len(observations)):
        for j in range(len(observations[i])):
            imgs.append(render(observations[i][j]))

    with mediapy.set_show_save_dir("data/rgms/"):
        mediapy.show_videos({game: imgs}, width=320, height=320, fps=20, codec="gif")

    print("Interacted with the environment for", steps, "steps")

    totals = []
    for r in rewards:
        totals.append(jnp.sum(jnp.asarray(r)))
    print("Got reward (avg/max)", jnp.mean(jnp.asarray(totals)), jnp.max(jnp.asarray(totals)))
    print("Trained on", train_sequences, "sequences, totalling", train_steps, "steps")
