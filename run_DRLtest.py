import gym
import sys
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import numpy as np
from mpi4py import MPI
import tensorflow as tf


# OH: 0.08, IS: 2, NRI: 0, NIR: 2, injNum: 20  <-- training case (test0)
# OH: 0.12, IS: 1, NRI: 0, NIR: 1, injNum: 32  <-- test 8
# OH: 0.1, IS: 1, NRI: 0, NIR: 3, injNum: 20   <-- test 11
# OH: 0.08, IS: 2, NRI: 0, NIR: 1, injNum: 23  <-- test 43
# OH: 0.12, IS: 2, NRI: 0, NIR: 1, injNum: 28  <-- test 44

NO_ACTION=False

_OH=0.08
_IS=2
_NRI=0
_NIR=2
_injNum=20

test_num = "test0"
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


#print("env created")
try:
    model = DDPG.load(SAVE_LOCATION + "/" + str(rank) + "_Best_Agent", env=env)
except:
    try:
        model = DDPG.load(SAVE_LOCATION + "/" + str(rank) + "_Agent", env=env)
    except:
        print("Agent not found. Using untrained agent instead")
        model = DDPG('MlpPolicy', env, train_freq=(1,"episode"), learning_starts=1000, action_noise = action_noise, device="cpu")

#print("model created")
cb = CustomCallback(SAVE_FILE_NAME=SAVE_LOCATION, rendering=False, testing=True)
#print("CB created")

death_count =0
live_count = 0
for test_ep in range(100):
    #print(rank, test_ep)

    obs = env.reset(OH=_OH, IS=_IS, NRI=_NRI, NIR=_NIR, injNum=_injNum, seed=test_ep)
    done = False
    step = 0
    if NO_ACTION:
        action = tf.expand_dims(np.array([0,0,0,0,0,0,0,0,0,0,0]),0)
    while not done:
        if not NO_ACTION:
            action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
    if env.oxydef_history[env.current_step] > 600:
        death_count += 1
    else:
        live_count += 1
    try:
        cyto_data = np.load(SAVE_LOCATION+"/" + str(rank) + "_cyto_" + test_num + ".npy")
        cyto_data = np.concatenate((cyto_data, np.expand_dims(env.full_history, axis=0)), axis=0)
        np.save(SAVE_LOCATION+"/" + str(rank) + "_cyto_" + test_num + ".npy", cyto_data)
    except FileNotFoundError:
        np.save(SAVE_LOCATION+"/" + str(rank) + "_cyto_" + test_num +".npy", np.expand_dims(env.full_history, axis=0))
    try:
        cyto_data = np.load(SAVE_LOCATION+"/" + str(rank) + "_action_"+test_num+".npy")
        cyto_data = np.concatenate((cyto_data, np.expand_dims(env.action_history, axis=0)), axis=0)
        np.save(SAVE_LOCATION+"/" + str(rank) + "_action_"+test_num+".npy", cyto_data)
    except FileNotFoundError:
        np.save(SAVE_LOCATION+"/" + str(rank) + "_action_"+test_num+".npy", np.expand_dims(env.action_history, axis=0))

print("\n\nOH: {}, IS: {}, NRI: {}, NIR: {}, injNum: {}\n\n".format(_OH, _IS, _NRI, _NIR, _injNum))
print(rank, live_count, death_count)
