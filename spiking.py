import numpy as np
import matplotlib.pyplot as plt
from network import Network


R = 1 # 200 ohms
k = 0 #W
time = 10000
N_neurons = 4


def pwl(x,alpha,beta,delta):
    """PWL tanh function"""
    ub=(alpha/beta)+delta
    lb=(-alpha/beta)+delta
    if isinstance(x,np.ndarray):
        sz=len(x)
        y=np.zeros(sz)
        for i in range(sz):
            if beta>=0:
                if x[i]<=min(lb,ub):
                    y[i]=-abs(alpha)
                elif min(lb,ub)<x[i]<=max(lb,ub):
                    y[i]=abs(beta)*(x[i]-delta)
                elif x[i]>=max(ub,lb):
                    y[i]=abs(alpha)
            else:
                if x[i]<=min(lb,ub):
                    y[i]=abs(alpha)
                elif min(lb,ub)<x[i]<=max(lb,ub):
                    y[i]=-abs(beta)*(x[i]-delta)
                elif x[i]>=max(ub,lb):
                    y[i]=-abs(alpha)
    else:
        y=[]
        if beta>=0:
            if x<=min(lb,ub):
                y=-abs(alpha)
            elif min(lb,ub)<x<=max(ub,lb):
                y=abs(beta)*(x-delta)
            elif x>=max(ub,lb):
                y=abs(alpha)
        else:
            if x<=min(lb,ub):
                y=abs(alpha)
            elif min(lb,ub)<x<=max(ub,lb):
                y=-abs(beta)*(x-delta)
            elif x>=max(ub,lb):
                y=-abs(alpha)
    return y

alpha = [-2, 2, -1.5, 2] # In the order of alpha_f_neg, alpha_s_pos, alpha_s_neg, alpha_us_neg
delta = [0, 0, -0.88, 0] # In the order of delta_f_N, delta_s_P, delta_s_N, delta_us_N
beta = [-2, 2, -1.5, 35]


Cm =  1   # 5 micro Farad

Iapp=[-0.2, -1.8, -1.8, -1.8]

I = np.zeros((N_neurons,time))
for i in range(N_neurons):
    I[i][:] = Iapp[i]


Vm = np.zeros((N_neurons,time)) # milli volts
Vf = np.zeros((N_neurons,time))
Vs = np.zeros((N_neurons,time))
Vus = np.zeros((N_neurons,time))
z = np.zeros((N_neurons,time))


tauf = 1
taus = 50
tauus = 50 * 50
dT = 1 # 1 milli second

Erev=0.48
Vthresh=1.9

eventp = []#-1*np.ones(N_neurons)
for i in range(N_neurons):
    eventp.append([])

net = Network()
# only use CPG neurons 
# print(net.W_Synapse)
W = net.W_Synapse[2:]
# print("W=", W)
W = np.delete(W, [0, 1], axis=1)
# print("W=", W)


def update_input_neuron_spike(input: str):
    """Update the input neuron spike"""
    if input == "gyro":
        net.update_gyro_neu()
    else:
        net.update_camera_neu()


spiked_neu = []
# print(net.W_Synapse)

def evolve_network(t : int) -> list:
    """ Function to evolve the membrane voltage of neurons"""
    for cur_neu in range(N_neurons):
        dVfdT = (Vm[cur_neu][t - 1] - Vf[cur_neu][t - 1]) / tauf
        dVsdT = (Vm[cur_neu][t - 1] - Vs[cur_neu][t - 1]) / taus
        dVusdT = (Vm[cur_neu][t - 1] - Vus[cur_neu][t - 1]) / tauus

        Vf[cur_neu][t] = Vf[cur_neu][t - 1] + (dVfdT * dT)
        Vs[cur_neu][t] = Vs[cur_neu][t - 1] + (dVsdT * dT)
        Vus[cur_neu][t] = Vus[cur_neu][t - 1] + (dVusdT * dT)

        F_N = pwl(Vf[cur_neu][t - 1], alpha[0], beta[0], delta[0])
        S_P = pwl(Vs[cur_neu][t - 1], alpha[1], beta[1], delta[1])
        S_N = pwl(Vs[cur_neu][t - 1], alpha[2], beta[2], delta[2])
        US_N = pwl(Vus[cur_neu][t - 1], alpha[3], beta[3], delta[3])

        Isum = 0
        for conn in range(N_neurons):
            Isum += (W[cur_neu][conn] * (Erev - Vm[conn][t-1]) + 
                        net.W_Synapse[0][conn+2] * net.spiked[0] + 
                            net.W_Synapse[1][conn+2] * net.spiked[1])


        I_x = F_N + S_P + S_N + US_N
        I_P = (Vm[cur_neu][t - 1] / R)


        dVmdT = (- I_P - I_x + I[cur_neu][t] + Isum) / Cm
        Vm[cur_neu][t] = Vm[cur_neu][t - 1] + (dVmdT * dT)
        # print(Isum, I_x + I_P, cur_neu, t, Vm[cur_neu][t])

        if Vm[cur_neu][t] > Vthresh and Vm[cur_neu][t-1] < Vthresh:
            net.spiked[cur_neu+2] = 1
            eventp[cur_neu].append(t)

    spiked_neu = []
    # care about CPG neuron spikes only
    for neu, spike in enumerate(net.spiked[2:], start=2):
        if spike == 1: spiked_neu.append(neu - 2)

    # reset spiked
    net.reset()

    return spiked_neu


desired_gait_time = []
def weight_update_combination(spiked_neu, t) -> list:
    """Determines the combinations of neurons for synaptic
    weight updates."""
    # dictionary with all neuron combos used
    neuron_combos = {num: [] for num in range(0, 6)}

    # list of tuples of weights that have to be updated
    # this order is important since we have directed edges! 
    # pre neuron first, post/cur neuron second
    to_be_updated_weights = []

    # pre_spiked neurons list
    pre_spiked_neu = []
    for neu, spike in enumerate(net.pre_spiked[2:], start=2):
        if spike == 1: pre_spiked_neu.append(neu - 2)

    # desired gait
    if (0 and 2 in pre_spiked_neu and 1 and 3 in spiked_neu) or \
    (1 and 3 in pre_spiked_neu and 0 and 2 in spiked_neu):
        desired_gait_time.append(t)
        

    for pre in pre_spiked_neu:
        for cur in spiked_neu:
            # make sure combo not used and no self connections
            if pre in neuron_combos[cur] or pre == cur:
                continue
            # else register in neuron_combos dict
            # and list of tuples of weights
            neuron_combos[cur].append(pre)
            neuron_combos[pre].append(cur)
            to_be_updated_weights.append((pre, cur))

    return to_be_updated_weights


def update_weights(reward, learning_rate, spiked_neu, t) -> None:
    """Updating CPG neuron weights"""
    combos = weight_update_combination(spiked_neu, t)
    # print("combos=", combos)
    for i in range(len(combos)):
        # apply synaptic weight update calculation on CPG neurons
        # random number between 0 and 1
        pre, post = combos[i]

        # print(f"net.W_Synapse[{pre}][{post}] before = ", net.W_Synapse[pre][post])
        net.W_Synapse[pre][post] += \
            learning_rate * reward * np.random.random()
        # print(f"net.W_Synapse[{pre}][{post}] after = ", net.W_Synapse[pre][post])
        
        # Clip the weights to maximum and minimum
        if net.W_Synapse[pre][post] > net.max_W:
            net.W_Synapse[pre][post] = net.max_W

        if net.W_Synapse[pre][post] < net.min_W:
            net.W_Synapse[pre][post] = net.min_W

    # update the pre_spiked array
    net.pre_spiked = net.spiked


def plot():
    plt.subplot(3,2,1)
    plt.title('Neuron 1')
    plt.plot(1* np.linspace(1, time, num=time), Vm[0],'b')
    plt.subplot(3,2,2)
    plt.title('Neuron 2')
    plt.plot(1* np.linspace(1,time, num=time),Vm[1],'r')
    plt.subplot(3,2,3)
    plt.title('Neuron 3')
    plt.plot(1* np.linspace(1,time, num=time),Vm[2],'g')
    plt.subplot(3,2,4)
    plt.title('Neuron 4')
    plt.plot(1* np.linspace(1,time, num=time),Vm[3],'y')
    plt.subplot(3,2,5)

    plt.plot(1* np.linspace(1, time, num=time), Vm[0],'b')
    plt.plot(1* np.linspace(1, time, num=time), Vm[1],'r')

    plt.subplot(3,2,6)
    plt.plot(1* np.linspace(1, time, num=time), Vm[2],'g')
    plt.plot(1* np.linspace(1, time, num=time), Vm[3],'y')
    #plt.xlim([0.1,35000])
    #plt.ylim([-5,5])
    plt.show()
    plt.plot(1* np.linspace(1, time, num=time), Vm[0],'b')
    plt.plot(1* np.linspace(1, time, num=time), Vm[1],'r')
    plt.plot(1* np.linspace(1, time, num=time), Vm[2],'g')
    plt.plot(1* np.linspace(1, time, num=time), Vm[3],'y')
    plt.show()
    plt.subplot(2,2,1)
    plt.eventplot(eventp,color='b')
    plt.subplot(2,2,2)
    plt.eventplot(eventp[1],color='r')

    plt.subplot(2,2,3)
    plt.eventplot(eventp[2],color='g')

    plt.subplot(2,2,4)
    plt.eventplot(eventp[3],color='y')
    #plt.show()
    plt.eventplot([[200,100,300,400],[400],[600],[800]])
    #plt.eventplot(eventp[1],color='r')
    #plt.eventplot(eventp[2],color='g')
    #plt.eventplot(eventp[3],color='y')
    #plt.show()

    # plt.plot(1e-3 * np.linspace(1,time, num=time),I[0])
    # plt.plot(1e-3 * np.linspace(1,time, num=time),I[1])
    # plt.plot(1e-3 * np.linspace(1,time, num=time),I[2])
    # plt.plot(1e-3 * np.linspace(1,time, num=time),I[3])

    # plt.show()


def print_spiked_limbs():
    """Print spiked limbs test."""
    print(f"Spiked neurons (limbs): {spiked_neu}")



def show_desired_gait_time() -> list:
    """Show desired gait times as a list."""
    return desired_gait_time



# variables redefined for testing purposes
# Iapp=[-2, -1.8, -1.8, -1.8]
# W = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0.9, 0, 0, 0]])
# # print("test_W=", W)

# alpha = [-2, 2, -1.5, 2] # In the order of alpha_f_neg, alpha_s_pos, alpha_s_neg, alpha_us_neg
# delta = [0, 0, -0.88, 0] # In the order of delta_f_N, delta_s_P, delta_s_N, delta_us_N
# beta = [-2, 2, -1.5, 2]


def sim_evolve_normal():
    """Evolves without RL for testing purposes."""
    for t in range(2, time):
        for cur_neu in range(N_neurons):
            dVfdT = (Vm[cur_neu][t - 1] - Vf[cur_neu][t - 1]) / tauf
            dVsdT = (Vm[cur_neu][t - 1] - Vs[cur_neu][t - 1]) / taus
            dVusdT = (Vm[cur_neu][t - 1] - Vus[cur_neu][t - 1]) / tauus

            Vf[cur_neu][t] = Vf[cur_neu][t - 1] + (dVfdT * dT)
            Vs[cur_neu][t] = Vs[cur_neu][t - 1] + (dVsdT * dT)
            Vus[cur_neu][t] = Vus[cur_neu][t - 1] + (dVusdT * dT)

            F_N = pwl(Vf[cur_neu][t - 1], alpha[0], beta[0], delta[0])
            S_P = pwl(Vs[cur_neu][t - 1], alpha[1], beta[1], delta[1])
            S_N = pwl(Vs[cur_neu][t - 1], alpha[2], beta[2], delta[2])
            US_N = pwl(Vus[cur_neu][t - 1], alpha[3], beta[3], delta[3])

            Isum = 0
            for conn in range(N_neurons):
                Isum += W[cur_neu][conn] * (Erev - Vm[conn][t])
            # Isum=0


            I_x = F_N + S_P + S_N + US_N
            I_P = (Vm[cur_neu][t - 1] / R)



            dVmdT = (- I_P - I_x + I[cur_neu][t] + Isum) / Cm
            Vm[cur_neu][t] = Vm[cur_neu][t - 1] + (dVmdT * dT)
            # print(Isum, I_x + I_P, cur_neu, t, Vm[cur_neu][t])
            if Vm[cur_neu][t] > Vthresh and Vm[cur_neu][t-1] < Vthresh:
                eventp[cur_neu].append(t)