import gym
import sys
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
from mpi4py import MPI

from iirabm_env_V2 import Iirabm_Environment

sys.path.insert(1,"./DRL_Files")
from callbacks import CustomCallback

SAVE_LOCATION="IIRABM_DRL_Experiments/limited_control"
print(SAVE_LOCATION)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


ENV_STEPS = 4200                # Total number of environment frames allowed
action_L1= .1
potential_difference_mult=200
phi_mult = 100


env = Iirabm_Environment(rendering=None, ENV_MAX_STEPS=ENV_STEPS, action_L1=action_L1, potential_difference_mult=potential_difference_mult, phi_mult = phi_mult)
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

print("env created")
try:
    model = DDPG.load(SAVE_LOCATION + "/" + str(rank) + "_Best_Agent", env=env)
except:
    try:
        model = DDPG.load(SAVE_LOCATION + "/" + str(rank) + "_Agent", env=env)
    except:
        model = DDPG('MlpPolicy', env, train_freq=(1,"episode"), learning_starts=1000, action_noise = action_noise, device="cpu")

print("model created")
cb = CustomCallback(SAVE_FILE_NAME=SAVE_LOCATION, rendering=False)
print("CB created")
model.learn(total_timesteps=432000, callback=cb)
model.save(SAVE_LOCATION + "/Agent")
