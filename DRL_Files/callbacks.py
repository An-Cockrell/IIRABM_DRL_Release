from stable_baselines3.common.callbacks import BaseCallback
import h5py
import os
import shutil
import numpy as np
from mpi4py import MPI

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0, SAVE_FILE_NAME="training_data", rendering = False, testing = False):
        super(CustomCallback, self).__init__(verbose)
        self.created_spatial_dataset = False
        self.save_path = SAVE_FILE_NAME
        self.render = rendering
        self.testing = testing
        self.episode_num = 0
        self.cum_reward = 0
        self.best_reward = -10000
        print("callback Created")
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if rank==0:
            # try:
            #     shutil.rmtree(self.save_path)
            #     os.remove(self.save_path+"/training_data.csv")
            #     os.remove(self.save_path+"/spatial_data.npy")
            # except Exception as e:
            #     print(e)
            try:
                os.mkdir(self.save_path)
            except Exception as e:
                print(e)
        print("starting training")

        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.model.save(self.save_path + "/" + str(rank) + "_Agent")

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.cum_reward += self.training_env.get_attr("reward")[0]
        if self.training_env.get_attr("done")[0]:
            print("\n\n\n Done inside callback\n\n")
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()

            if self.episode_num%20 ==0:
                outcome_hist = self.training_env.get_attr("outcome_hist")[0][-20:]
                print("Last 20 episode outcomes: {.2d} live,  {.2d} timeout,  {.2d} die".format(outcome_hist.count(1), outcome_hist.count(3), outcome_hist.count(2)))
        if self.render:
            self.training_env.render()
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if not self.testing:
            with open(self.save_path+"/" + str(rank)+"_cyto_output.csv", "ab") as f:
               np.savetxt(f, self.training_env.get_attr("full_history")[0], delimiter=",")
            with open(self.save_path+"/" + str(rank)+"_action_taken.csv", "ab") as f:
               np.savetxt(f, self.training_env.get_attr("action_history")[0], delimiter=",")

            try:
                cyto_data = np.load(self.save_path+"/" + str(rank) + "_cyto.npy")
                cyto_data = np.concatenate((cyto_data, np.expand_dims(self.training_env.get_attr("full_history")[0], axis=0)), axis=0)
                np.save(self.save_path+"/" + str(rank) + "_cyto.npy", cyto_data)
            except FileNotFoundError:
                np.save(self.save_path+"/" + str(rank) + "_cyto.npy", np.expand_dims(self.training_env.get_attr("full_history")[0], axis=0))
            try:
                cyto_data = np.load(self.save_path+"/" + str(rank) + "_action.npy")
                cyto_data = np.concatenate((cyto_data, np.expand_dims(self.training_env.get_attr("action_history")[0], axis=0)), axis=0)
                np.save(self.save_path+"/" + str(rank) + "_action.npy", cyto_data)
            except FileNotFoundError:
                np.save(self.save_path+"/" + str(rank) + "_action.npy", np.expand_dims(self.training_env.get_attr("action_history")[0], axis=0))


            if os.path.isfile(self.save_path+"/" + str(rank) + "_reward.csv"):
               with open(self.save_path+"/" + str(rank) + "_reward.csv", "ab") as f:
                   np.savetxt(f, [self.cum_reward], delimiter=",")
            else:
               np.savetxt(self.save_path+"/" + str(rank) + "_reward.csv", [self.cum_reward], delimiter=",")
        if self.cum_reward >= self.best_reward:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            self.model.save(self.save_path + "/" + str(rank) + "_Best_Agent")
            self.best_reward = self.cum_reward
        self.cum_reward = 0
        self.episode_num+=1

        if self.episode_num%20 ==0:
            outcome_hist = self.training_env.get_attr("outcomes")[0][-20:]
            print("rank {} finished episode {}.   Last 20 episode outcomes: {} live,  {} timeout,  {} die                                                  ".format(rank, self.episode_num, outcome_hist.count(1), outcome_hist.count(3), outcome_hist.count(2)))

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
