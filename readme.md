# Reinforcement Learning Based Spiking Neural Network for a Quadruped Robot Simulation

This repo is for a simulation of a RL-based SNN which controls the locomotion of a quadruped robot. This repo is following the process used by the Learning to Walk paper.

**Note: No local environment setup is required.**

Follow the steps to get the repo working on your local machine:

1. Clone the repo on your local machine via git. Open the directory in which you would like to clone the repo via your terminal, and then paste the following command: `git clone https://github.com/samchopra2003/Reinforcement-Learning-Based-Spiking-Neural-Network.git`.

2. Please download the following dependencies if you do not have them downloaded: `numpy`, `matplotlib`, `tqdm`. A simple pip install should be sufficient.

3. Now you are all ready to run the program. Simply run `python main.py` from command prompt or whatever your preferred choice of CLI is.

---

## Some notes about the code

You can tune the parameters of the reward calculation at the top of `main.py`, namely the parameters: `gamma` and `T1`. `gamma` represents the learning rate and `T1` is the coefficient by which we divide the reward from the camera.

If you want to increase the time that the code runs, simply change the `time` variable at the top of the `main.py` and `spiking.py` files. The times must be the same. In future, I will change the repo such that there is only one `time` variable.

Most of the spiking neural network parameters are in `spiking.py`, except the randomly initialized weight matrix. This weight matrix is initialized in `network.py`, and it is a 6x6 matrix, 2 sensory neurons (camera and gyroscope) and 4 CPG neurons. 

`gyro.py` is simulating the gyroscope reward for the robot by setting up a very simple physics environment. The comments in the file are self-explanatory with regards to the function of the variables and methods.

Similarly, `camera.py` is simulating the camera reward for the robot by setting up a very simple physics environment. Please read comments in the file to learn more.

`testing.py` is a file for unit testing for some functions, not relevant to the working of the code.

Lastly, although an episode lasts for 10000 time steps in the current code, the episode may terminate sooner, if the robot falls down or when the robot makes 50 bad moves ina row, and thus no forward translation is achieved. In that case, the code will terminate then and there, and display a message in the terminal.

## TODO

- Obtain the desired gait consistently, the `desired_gait_time` list shows the time stamps of when the desired gait pattern has occurred.

- Might have to improve the complexity of the physics environment.

- Might have to improve the synaptic weight update calculation, and do some error calculations and corrections (not implement in original paper).