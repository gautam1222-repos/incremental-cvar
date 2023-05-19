import numpy as np
import random


def generate_rand_upper_traingular_matrix(w, k):
    # d = max(w,k)
    Cj = np.zeros(shape=(w,k))
    for i in range(w):
        Cj[i][i] = 1
    for i in range(w):
        for j in range(w):
            if i<j:
                Cj[i][j] = random.randrange(2)
    return Cj



def _Cj(w, k):
    return generate_rand_upper_traingular_matrix(w, k)



def get_base_form(idx, k):
    base = np.zeros(k)
    i=0
    while(i<k):
        base[i] = idx%2
        idx = idx//2
        i+=1
    return base


def get_equi_distributed(k, s):
    arr = np.zeros(s)
    for i in range(k):
        arr[random.randint(0,k)%s]+=1
    return arr


def get_random_decimal(k):
    val = 0
    for i in range(int(k)):
        val += random.randrange(2)*(2**(-1*(i+1)))
    return val 


def get_random_shift(s, k):
    arr = np.array([])
    k_arr = get_equi_distributed(k, s)
    for i in range(s):
        arr = np.append(arr, get_random_decimal(k_arr[i]))
    return arr


def float_bin(n, k):
    arr = np.zeros(k)
    for i in range(k):
        n_int = int(n*2)
        n = 2*n - n_int 
        arr[i] = n_int
    return arr


def generate_digital_nets_base2(power=10, axes=2, n_i=0, scramble=True):
    
    k=power
    N = 2**k
    k += int(np.ceil(np.log2(n_i)))
    s = axes
    point_set = []
    random_point_set = []


    C = []
    for j in range(s):
        C.append(_Cj(k,k))


    w = k
    U = np.random.uniform(low=0.0, high=1.0, size=s)

    for i in range(n_i, N+n_i):
        base_form = get_base_form(idx=i, k=k)
        base_form = np.reshape(base_form, (-1,1)) 

        u_i_rqmc = np.array([])
        u_i_qmc = np.array([])
        for j in range(s):
            tmp_qmc = np.matmul(C[j], base_form)
            u_ij_mat = tmp_qmc%2
            Uj = float_bin(U[j], w)
            u_ij_arr = np.reshape(u_ij_mat, (len(u_ij_mat,)))
            
            add_arr = u_ij_arr+Uj
            add_arr = add_arr%2
            u_ij_qmc = 0
            u_ij_rqmc = 0
            for l in range(w):
                u_ij_qmc += (u_ij_arr[l])*(2**(-(l+1)))
                u_ij_rqmc += (add_arr[l])*(2**(-(l+1)))
            
            u_i_qmc = np.append(u_i_qmc, u_ij_qmc)
            u_i_rqmc = np.append(u_i_rqmc, u_ij_rqmc)
            
        u_i_qmc = np.reshape(u_i_qmc, (-1,1))
        u_i_rqmc = np.reshape(u_i_rqmc, (-1,1))

        point_set.append(u_i_qmc)
        random_point_set.append(u_i_rqmc)
    return random_point_set
