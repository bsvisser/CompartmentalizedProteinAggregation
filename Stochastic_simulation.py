import numpy as np
import math
import numpy
from datetime import date
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy.integrate import odeint


#   Function used for finding t50 values
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#   Function used for calculating propensities
def Prop(s, k):
    prop = [s[0]*(s[0]-1)/2*k[0],
            2*s[0]*s[1]*k[1],
            s[0]*(s[0]-1)*s[2]/2*k[2],
            s[3]*(s[3]-1)/2*k[3]*R,
            2*s[3]*s[4]*k[4]*R,
            s[3]*(s[3]-1)*s[5]/2*k[5]*R**2,
            s[0]*k[6],
            s[3]*k[7]*R
           ]
    return prop

simulations = 1000 #    Number of simulations


#   Parameters
R = 10 #    Volume fraction of compartment
Kp = 10 #   Partitioning coefficient
cS = 5E-4 # Concentration of monomer
S = (R+1)*cS/(Kp+R) #C  oncentration outside
P = 0 # Concentration of aggregates t0
M = 0 # Concentration of monomers in aggregates t0

S_in = Kp*S #   Concentration of monomers inside
P_in = 0 #  Concentration of aggregates inside t0
M_in = 0 #  Concentration of monomers in aggregates t0
S_in_seq = 0 #  Concentration of sequestered monomers inside

S0 = [S,P,M,S_in,P_in,M_in]


kin = 0.3
kout = kin/Kp
kmod = 1
kn_o = 0.1
kplus_o = 500
k2_o = 8000

n1 = 2
n2 = 2

state_index = [0,1,2,3,4,5,6]
state_label = ["S","P","M", "S_in","P_in","M_in", "M_tot"]

#   Gillespie model

#    reactions and fluxvector
#      S   P   M S_in P_in M_in
F = [[-2, +1, +2,  0,  0,  0], #    0) s + s > p + m + m , kn;#    outer compartment
     [-1,  0, +1,  0,  0,  0], #    1) s + p > p + m , kplus;
     [-2, +1, +2,  0,  0,  0], #    2) s + s + m > p + m + m + m , k2;
     [ 0,  0,  0, -2, +1, +2], #    3) s_in + s_in > p_in + m_in + m_in , kn_in;#    inner compartment
     [ 0,  0,  0, -1,  0, +1],#    4) s_in + p_in > p_in + m_in , kplus_in;
     [ 0,  0,  0, -2, +1, +2],#    5) s_in + s_in + m_in > p_in + m_in + m_in + m_in , k2_in;    
     [-1,  0,  0, +1,  0,  0], #    6) s > s_in , kin;#    transport
     [+1,  0,  0, -1,  0,  0] #    7) s_in > s , kout = kin/Kp;
    ]
Fa = np.array(F)
Ft = Fa.transpose()

S_tot = 5000
S = round((R*S_tot)/(R+Kp))
P = 0
M = 0

S_in = (S_tot-S)
P_in = 0
M_in = 0

Na = 6.02E23
file = "%d_%d_%f_%d_%d_%f_%d_%d_%f_%d_%s.txt" % (S_tot,simulations,cS,R,Kp,kn,kplus,k2,kin,kmod,date)
with open(file, 'w') as f:
    f.write("M_in/(M_out+M_in)\tt50 (in)\tt50 (out)\tt50 (all)\t(M)/(P) \t(M_in)/(P_in) \t(M+M_in)/(P+P_in)")

simrun=0
averaget50=0
for i in range(simulations):
    simrun+=1
    print(simrun)     

    #    initial conditions
    x=S_tot/cS*R/(R+1)

    state = [S , P, M, S_in, P_in, M_in,]
    
    t_max = 500
    
    #    reaction rates
    #    [kn,kplus,k2,kn_in,kplus_in,k2_in,kin,kout]
    
    kn = kn_o
    kplus = kplus_o
    k2 = k2_o
    
    kn_in = kn*kmod
    kplus_in = kplus*kmod
    k2_in = k2*kmod
    
    kn /= (x/2)
    kplus /= x
    k2 /= (x**2/2)
    
    kn_in /= (x/2)
    kplus_in /= x
    k2_in /= (x**2/2)
    
    k = [kn,kplus,k2,kn_in,kplus_in,k2_in,kin,kout]
    
    #    simulation

    t = 0
    state_history = state
    time_history = np.array(0)
    
    prop = Prop(state,k)
    while t<t_max:
        if t >= 0.99*t_max:
            if (M+M_in) < 0.95*(S+S_in):
                t_max +=1
        p_cumsum = np.cumsum(prop)
        p_sum = p_cumsum[-1]
        if p_sum == 0:
            break
        p_cumsum /= p_sum
        #    pick reaction
        
        rand = np.random.random()
        r = np.searchsorted(p_cumsum, rand)
        
        #    update state
        
        dstate = Ft[:, r]
        state += dstate
        state = numpy.where(state<0, 0, state)
        state = state.astype('int64')
        
        #    update propensities
        
        prop = Prop(state,k)
        t += 1.0 / p_sum * np.log(1.0 / rand)
        state_history = np.vstack((state_history,state))
        time_history = np.vstack((time_history,t))
        
    #    plot
    
    state_index = [0,1,2,3,4,5,6]
    state_label = ["S","P","M", "S_in","P_in","M_in","M_tot"]
    
    M_tot=[]
    for a in state_history:
        M_tot.append(a[2]+a[5])
        
    M_tot = np.array(M_tot)
    M_tot = M_tot.reshape(len(M_tot),1)
    state_history = np.column_stack((state_history,M_tot))
    
    for j in range(len(state_index)):
        plt.plot(time_history, state_history[:, state_index[j]], alpha=0.8,label = state_label[j])
    
    plt.legend(fancybox = True)
    plt.xlabel('t')
    plt.ylabel('molecules')
    plt.show()
    
    
    #   Printing of results and writing to file
    results=[]
    AggF = state_history[-1, state_index[5]]/(state_history[-1, state_index[2]]+state_history[-1, state_index[5]])
    print("M_in/(M_out+M_in): ", AggF)
    t50in= time_history[np.where(state_history[:, state_index[5]]==find_nearest(state_history[:, state_index[5]],0.5*max(state_history[:, state_index[5]])))[0][0]][0]
    print("t50 (In):", time_history[np.where(state_history[:, state_index[5]]==find_nearest(state_history[:, state_index[5]],0.5*max(state_history[:, state_index[5]])))[0][0]][0])
    t50out= time_history[np.where(state_history[:, state_index[2]]==find_nearest(state_history[:, state_index[2]],0.5*max(state_history[:, state_index[2]])))[0][0]][0]
    
    print("t50 (Out) :", time_history[np.where(state_history[:, state_index[2]]==find_nearest(state_history[:, state_index[2]],0.5*max(state_history[:, state_index[2]])))[0][0]][0])
    
    t50_all = time_history[np.where(state_history[:, state_index[6]]==find_nearest(state_history[:, state_index[6]],0.5*max(state_history[:, state_index[6]])))[0][0]][0]
    print("t50 (All) :", time_history[np.where(state_history[:, state_index[6]]==find_nearest(state_history[:, state_index[6]],0.5*max(state_history[:, state_index[6]])))[0][0]][0])
    
    averaget50 = ((averaget50*(simrun-1))+t50_all)/simrun
    print(averaget50)
        
    MP = (state_history[:,2][-1]/state_history[:,1][-1])
    print("M/P :", MP)
    
    MinPin = (state_history[:,5][-1]/state_history[:,4][-1])
    print("M_in/P_in :", MinPin)
    
    MtotPtot = (state_history[:,5][-1] + state_history[:,2][-1]) / (state_history[:,4][-1]+state_history[:,1][-1])
    print("M_tot/P_tot :", MtotPtot)
    
    results.extend([AggF,t50in,t50out,t50_all,MP,MinPin,MtotPtot])
    results = [1 if math.isnan(x) else x for x in results]
    
    with open(file, 'a') as f:
        f.write("\n")
        for row in results:
            f.write("%s\t" % str(row))
