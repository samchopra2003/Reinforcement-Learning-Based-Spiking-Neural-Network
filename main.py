"""Main file which will combine all the components."""
from network import Network
from spiking import evolve_network, update_weights, update_input_neuron_spike, plot
from gyro import Gyro
from camera import Camera
# from testing import forecast

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

""" Global variables """
# time limit we will run the main loop till
time = 10000
# camera reward coefficient
T1 = 20
# gamma = learning rate
gamma = 0.15

"""Main loop (following master algorithm)"""
def main():
    # Initializing synaptic weights randomly in network.py
    # Initialized CPG neuron voltages in spiking.py

    # Define camera and gyro instances
    cam = Camera()
    gyro = Gyro()
    reward = np.zeros(time)

    # forecast()

    t_break = 0

    for t in tqdm(range(2, time)):

        neu_spiked = evolve_network(t)

        # let network evolve for 100 time steps 
        if t < 100:
            continue

        # limbs correspond to neurons 2 through 5
        fell, balanced = gyro.step(neu_spiked)

        # if robot has fallen, terminate this episode
        if fell: 
            t_break = t
            break

        # if balanced set the gyro spike
        if balanced: update_input_neuron_spike("gyro")

        # bad_streak means terminate episode
        # bad_translation means bad forward translation for current time step
        bad_streak, good_translation = cam.step(neu_spiked)

        # if robot not showing forward translation, terminate this episode
        if bad_streak: 
            t_break = t
            break

        # if good forward translation, set camera spike
        if good_translation: update_input_neuron_spike("camera")
        
        # calculate reward
        reward[t] = gyro.reward + cam.reward * (1/T1)

        # update synaptic weights
        update_weights(reward=reward[t], learning_rate=gamma,
                       spiked_neu=neu_spiked)

        
        
    # Plot results
    cumu_reward = np.cumsum(reward)
    plt.plot(np.linspace(1, time, num=time), cumu_reward)
    plt.xlabel("Time (a.u.)")
    plt.ylabel("Cumulative reward")
    plt.show()

    # Plot Vm evolution at the end
    plot()

    # see first 100 rewards
    # print(reward[t_break-20:t_break])
    # print(gyro.current_balance)



# program entrypoint

if __name__ == "__main__":
    main()