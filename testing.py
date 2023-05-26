from spiking import (sim_evolve_normal, plot, 
                     evolve_network, print_spiked_limbs, weight_update_combination,
                     update_weights)


# Vm evolution forecast at start before RL 
def test():

    for t in range(2, 30000):
        _ = evolve_network(t)
        # if t < 20:
        #     print_spiked_limbs()

    plot()



test()

# Vm evolition normal test without RL
# sim_evolve_normal()
# plot()

# forecast for every time step
def forecast():
    for t in range(2, 30000):
        _ = evolve_network(t)
        if t >= 3 and t < 13:
            print_spiked_limbs()
        
    plot()



def test_weight_update_combination():
    combos = weight_update_combination([0, 1, 2])
    print(combos)
    

def test_updated_weights():
    update_weights(reward=5, learning_rate=0.15, spiked_neu=[0, 1, 2])


