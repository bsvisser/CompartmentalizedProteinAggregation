import numpy as np
import numpy
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Function used for finding t50 values
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#Parameters
R = 10 #Volume fraction of compartment
Kp = 10 #Partitioning coefficient
cS = 5E-4 #Concentration of monomer
S = (R+1)*cS/(Kp+R) #Concentration outside
P = 0 #Concentration of aggregates t0
M = 0 #Concentration of monomers in aggregates t0

S_in = Kp*S #Concentration of monomers inside
P_in = 0 #Concentration of aggregates inside t0
M_in = 0 #Concentration of monomers in aggregates t0
S_in_seq = 0 #Concentration of sequestered monomers inside

S0 = [S,P,M,S_in,P_in,M_in,S_in_seq]

t_max = 500

kin = 1

kout = kin/Kp

kmod = 1

kn = 0.1
kplus = 500
k2 = 8000
kseq_in = 1
kseq_out = 1

kn_in = kn * kmod
kplus_in = kplus* kmod
k2_in = k2* kmod

n1 = 2
n2 = 2

kn_o, kplus_o, k2_o = kn, kplus, k2

def prim(S, kn, n1):
    return kn*S**n1

def elon(S, P, kplus):
    return 2*kplus*S*P
    
def sec(S, M, k2, n2):
    return k2*M*S**n2 

def trans(S, kin):
    return kin*S

def partitioning(y, t, kn, kplus, k2, kn_in, kplus_in, k2_in, kin, kout, kseq_in, kseq_out, n1, n2):
    S, P, M, S_in, P_in, M_in, S_in_seq = y
    dydt = [-2*prim(S,kn,n1)-elon(S,P,kplus)-2*sec(S,M,k2,n2)-trans(S,kin)+trans(S_in,kout)-trans(S,kseq_in)+trans(S_in_seq,kseq_out),
            +prim(S,kn,n1)+sec(S,M,k2,n2),
            +2*prim(S,kn,n1)+elon(S,P,kplus)+2*sec(S,M,k2,n2),
            -2*prim(S_in,kn_in,n1)-elon(S_in,P_in,kplus_in)-2*sec(S_in,M_in,k2_in,n2)+trans(S,kin)*R-trans(S_in,kout)*R,
            +prim(S_in,kn_in,n1)+sec(S_in,M_in,k2_in,n2),
            +2*prim(S_in,kn_in,n1)+elon(S_in,P_in,kplus_in)+2*sec(S_in,M_in,k2_in,n2),
            +trans(S,kseq_in)-trans(S_in_seq,kseq_out)
    ]
    return dydt

t = np.linspace(0, t_max, 2000)

sol = odeint(partitioning, S0, t, args=(kn, kplus, k2, kn_in, kplus_in, k2_in, kin, kout, kseq_in, kseq_out, n1, n2))



state_index = [0,1,2,3,4,5,6,7]
state_label = ["S","P","M", "S_in","P_in","M_in", "M_tot","S_in_seq"]

M_tot=[]
for a in sol:
    M_tot.append(a[2]+a[5]/R)
    
M_tot = np.array(M_tot)
M_tot = M_tot.reshape(len(sol[:,0]),1)
sol = np.column_stack((sol,M_tot))
#np.concatenate((sol,np.concatenate(M_tot)[:,None]),axis=1)
for j in range(len(state_index)):
    plt.plot(t, sol[:, j], alpha=0.5,label = state_label[j])
plt.legend(fancybox = True)
plt.xlabel('t')
plt.ylabel('c')
plt.show()


print("t50 (All) :", t[np.where(sol[:,6]==find_nearest(sol[:,6],np.max(sol[:,6])*0.5))[0][0]])
print("t50 (In) :", t[np.where(sol[:,5]==find_nearest(sol[:,5],np.max(sol[:,5])*0.5))[0][0]])
print("t50 (Out) :", t[np.where(sol[:,2]==find_nearest(sol[:,2],np.max(sol[:,2])*0.5))[0][0]])

