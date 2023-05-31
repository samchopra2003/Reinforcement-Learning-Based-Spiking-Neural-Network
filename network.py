"""Fully connected neural network, with 4 CPG neurons and two input neurons
    from the camera and the gyroscope."""

import numpy as np

N_neurons = 6
init_max_weight = 0.1

# Network dimensions are 6x6 (fully_connected)
# Initializing random weights for all synapses.
class Network:

    def __init__(self):
        # Neurons 1 and 2 are for gyro and camera, rest are CPG neurons
        self.W_Synapse = np.random.random((N_neurons, N_neurons)) * init_max_weight
        # make sure no self connections
        for n in range(N_neurons): self.W_Synapse[n][n] = 0

        # spiked neurons (camera spike starts off as 1)
        self.spiked = [0, 1, 0, 0, 0, 0]
        # spiked neurons in the previous time step
        # used for updating synaptic weights
        self.pre_spiked = [0, 1, 0, 0, 0, 0]
        # clip the weights to a max and min value
        self.max_W = 2
        self.min_W = -2

    def update_camera_neu(self):
        """Spike if good, ideal or course correction forward translation"""
        self.spiked[1] = 1

    def update_gyro_neu(self):
        """Spike if balance is lost"""
        self.spiked[0] = 1

    def reset(self):
        """Reset self.spiked"""
        self.spiked = [0, 0, 0, 0, 0, 0]


# init_weights()
# print(type(W_Synapse))
# print(W_Synapse)