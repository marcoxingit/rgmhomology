import jax
import numpy as np

from atari.common import *
from rgm import *

import mediapy
from argparse import ArgumentParser


def evaluate(game, rgm, n=1):
    game = game.capitalize()
    if game == "Battlezone":
        game = "BattleZone"

    env = make_game(game_config(game, ObservationType.RGB))

    imgs = []
    recs = []

    rgm_agent = RGMAgent(rgm)

    rewards = 0.0

    for _ in range(n):
        terminated = False
        obs, _ = env.reset()
        imgs.append(obs)
        while not terminated:
            obs = jax.image.resize(obs, (64, 64, 3), "bilinear")
            action, im, _, _ = rgm_agent.act(obs)
            action = int(action[0])
            obs, reward, terminated, trunc, info = env.step(action)
            imgs.append(obs)
            recs.append(im)
            rewards += reward

        print(rewards)

    with mediapy.set_show_save_dir("/tmp"):
        mediapy.show_videos({game: imgs, game + "_imagined": recs}, width=160, height=210, fps=20, codec="h264")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-g", "--game", default="freeway")
    args = parser.parse_args()

    game = args.game

    if game == "freeway":
        action_range = [0, 8]
    elif game == "boxing":
        action_range = [3, 17]
    elif game == "breakout":
        action_range = [0, 8]
    elif game == "battlezone":
        action_range = [3, 17]
    elif game == "demonattack":
        action_range = [0, 8]
    elif game == "frostbite":
        action_range = [0, 18]
    elif game == "hero":
        action_range = [3, 17]
    elif game == "pacman":
        action_range = [0, 8]
    elif game == "pong":
        action_range = [0, 8]
    elif game == "seaquest":
        action_range = [0, 18]

    data_dir = "data/atari"
    observations = np.load(data_dir + "/" + game + "/frames.npz")["arr_0"][0, :1024, 0, :, :, :]  # 64x64
    actions = np.load(data_dir + "/" + game + "/actions.npz")["arr_0"][0][:1024]

    print("~~" + game + "~~")

    rgm = RGM(
        sv_thr=1.0 / 256, action_range=action_range, n_bins=action_range[1] - action_range[0] + 1, tile_diameter=16
    )
    # this will trigger SVD
    rgm.to_one_hot(observations, actions)

    # now iteratively add data and evaluate
    episode_length = 128

    for b in range(20):
        for e in range(8):
            end = (e + 1) * episode_length
            start = end - episode_length - 4
            if start < 0:
                start = 0

            print(b, start, end)

            observations = np.load(data_dir + "/" + game + "/frames.npz")["arr_0"][b, start:end, 0, :, :, :]  # 64x64
            actions = np.load(data_dir + "/" + game + "/actions.npz")["arr_0"][b, start:end]

            rgm.learn_structure(observations, actions)

            print(rgm.agents[-1].B[0].shape)
            rgm.save("data/rgms/" + game + "_rgm_" + str(b * 8 + e) + ".npz")

            jax.clear_caches()

    # print("evaluate ")
    # evaluate(game, rgm, 5)
