# IIRABM_DRL_Release
A repository for code for training a DRL agent on the IIRABM DRL environment
---

In the experiments we ran, a batch of 20 DDPG agents created using the stable_baselines3 package were trained simultaneously using MPI on the IIRABM environment. The environment is set up such that the agent selects an action, and that action is held for a number of timesteps that correlates to 6 hours of real time, then the agent is free to select a new action based on its observation.

The observation space available to the agents were aggregate values of all 11 cytokines in the simulation [TNF, sTNFr, IL10, GCSF, IFNg, PAF, IL1, IL4, IL8, IL12, sIL1r]. The values of the observation returned are the mean value at each step over the period of a particular action of the aggregate cytokines.

The action space for the agent in the limited control scheme is to either add up to 10 units of a cytokine to each grid square of the simulation, or to diminish the amount of that cytokine at each grid square by a factor of up to 10^-3. In the limited control version, the cytokines available to be acted on are [TNF, IL1, IL2, IL4, IL8, IL12, IFNg].

The simulation ends when oxygen deficit (a measure of total system damage in the IIRABM) either reaches a value below 100 (indicating a fully healed run) or above 8160 (indicating damage so great that the run is considered dead). If the simulation does not reach a terminal conclusion by 4200 simulation steps (~28 days of simulated time), the episode is cut off and a new one is started.

The reward at each step is the change in oxygen deficit compared to the previous step, with a decreasing value giving a positive reward. A terminal reward of 1000 is given for a run ending with a successful heal, and a penalty of -1000 is given for a run ending with a death.

Each agent was allowed to take 432,000 actions during the learning period, which equates to at least 6000 episodes, with that amount increasing slightly depending on how many episodes end before the time limit  of 4200 steps.



---


To run the simulation, compile the C++ code into a shared object file using the following command:
`g++ -fPIC -Wall -shared -o Simulation.so ./CPP_Hypo/*.cpp -std=c++11 -O3`

A new agent is trained by executing `run_DRL.py` and that agent is saved in a location specified inside the file called `SAVE_LOCATION`.
It can also be run using mpi using the command `mpiexec -n 20 python3 -u run_DRL.py` and each mpi process will train and save an individual agent.

Those agents can be tested using `run_DRLtest.py` by executing the file in the same way as `run_DRL.py`

The test data can be visualized using `plot_training_data.py` and correctly specifying the desired files to visualize.
