import numpy as np
import math
from scipy.stats import truncnorm, qmc


def gen_trunc_gauss_samples(mu, sd, a, b, N):
    a = (a-mu)/sd
    b = (b-mu)/sd
    samples = truncnorm.rvs(a, b, loc=mu, scale=sd, size = N)
    # samples = np.array([])
    # while(len(samples)<N):
    #     val = np.random.normal(loc=mu, scale=sd)
    #     if(val>=a and val <=b):
    #         samples = np.append(samples, val)
    return samples


def log_pdf_wrt_mu(x, mu, sd, a, b):
    
    expr1 = (x-mu)
    expr1/= sd**2

    # expr2_Nr = np.exp(-0.5*( (a-mu)/sd )**2) - np.exp(-0.5*( (b-mu)/sd )**2)
    # expr2_Dr = math.erf( (b-mu)/(np.sqrt(2)*sd) ) - math.erf( (a-mu)/(np.sqrt(2)*sd) )
    # expr2 = expr2_Nr/(2*np.sqrt(2)*expr2_Dr)

    # expr = expr1 + expr2
    
    return expr1

def log_pdf_wrt_sd(x, mu, sd, a, b):
    
    expr1 = (x-mu)**2 - sd**2
    expr1 /= sd**3

    # expr2_Nr = (a-mu)*np.exp(-0.5*(a-mu)**2/sd**2) - (b-mu)*np.exp(-0.5*(b-mu)**2/sd**2)
    # expr2_Dr = math.erf( (b-mu)/(np.sqrt(2)*sd) ) - math.erf( (a-mu)/(np.sqrt(2)*sd) )
    # expr2 = expr2_Nr/( 2*np.sqrt(2)*sd**2 )

    # expr = expr1+expr2

    return expr1

def Reward(samples):
    # sample_matrix = np.array(samples)
    axes = len(samples[0])
    N = len(samples)
    sum = np.zeros(N)
    # print(samples[:,0]+sum)
    for i in range(axes):
        sum = sum + samples[:,i]
    
    expr = 10/( 1+np.exp(-1*np.abs(sum-6)))
    return expr



def CVaRSGD(params, params2, d_params, proj_radius, center, t, eta):
    
    # params2 += eta*d_params
    # params += (1/(t+10))*(params2-params)
    params += eta*d_params
    if(np.linalg.norm(params-center)>proj_radius):
        print('projection')
        norm_val = np.linalg.norm(params-center)
        vec = params-center
        vec /= norm_val
        params = center + proj_radius*vec
    
    return params



'''
        |                          |                       |
        |        SAMPLE MATRIX     |       COORDINATES     |
 ====== | ======================== |=======================|
    i   |    X       Y       Z     |   X   |    Y  |    Z  |
        |    0       1       2     |       |       |       |
 ------ | ------------------------ |-------|-------|-------|
    0   |    x0      y0      z0    |   x0  |       |       |
    1   |    x1      y1      z1    |   x1  |       |       |
    2   |    x2      y2      z2    |   x2  |       |       |
    3   |    x3      y3      z3    |   x3  |       |       |
    4   |    x4      y4      z4    |   x4  |       |       |
    5   |    x5      y5      z5    |   x5  |       |       |
    6   |    x6      y6      z6    |   x6  |       |       |
    7   |    x7      y7      z7    |   x7  |       |       |
    8   |    x8      y8      z8    |   x8  |       |       |
    9   |    x9      y9      z9    |   x9  |       |       |
        |  [9][0]                  |   [9] |       |       |


    Now x6 = X[6], sample_matrix"[row=6][axis=0]"

'''

# grad = GCVAR(alpha, N, log_pdf_wrt_mu, samples, lower_limits, upper_limits, params, variances, rewards, VaR)
def GCVAR(alpha, N, log_pdf_wrt_mu, samples , lower_limits, upper_limits, params, variances, rewards, VaR):
    sample_matrix = np.array(samples)
    axes = len(sample_matrix[0])
    gcvar = np.zeros(axes)
    for i in range(len(rewards)):
        if(rewards[i]<=VaR):
            print('i',end='')
            for j in range(axes): 
                # each row = sample, each column = coordinate
                gcvar[j] += log_pdf_wrt_mu(sample_matrix[i][j], params[j], variances[j], lower_limits[j], upper_limits[j])*(rewards[i]-VaR)
            # # param[0] => mu_x, param[1] => sd_x
            # gcvar[0] += (log_pdf_wrt_mu(x[i], params[0], sd_x, a_x, b_x))*(rewards[i]-VaR)
            # # gcvar[1] += (log_pdf_wrt_sd(x[i], params[0], params[1], a_x, b_x))*(rewards[i]-VaR)
            # # param[2] => mu_y, param[3] =>sd_y
            # gcvar[1] += (log_pdf_wrt_mu(y[i], params[1], sd_y, a_y, b_y))*(rewards[i]-VaR)
            # # gcvar[3] += (log_pdf_wrt_sd(y[i], params[2], params[3], a_y, b_y))*(rewards[i]-VaR)    
    gcvar /= (alpha*N)
    print()
    return gcvar


def fun(alpha, mu_x, sd_x, a_x, b_x, mu_y, sd_y, a_y, b_y):
    N = 4000+np.random.randint(1000)%1000
    
    # generate samples
    x = gen_trunc_gauss_samples(mu_x, sd_x, a_x, b_y, N)
    y = gen_trunc_gauss_samples(mu_y, sd_y, a_y, b_y, N)
    
    # assign rewards
    rewards = Reward(x,y)
    print(rewards)
    # break
    # Calculate Value at Risk
    sort_rew = np.sort(rewards)
    VaR = sort_rew[int(alpha*N)]
    
    print ( np.mean(sort_rew[:int(alpha*N)+1]))

def list_of_pair_to_list_of_vector(pairs):
    vectors = []
    for pair in pairs:
        vectors.append(pair.reshape((-1, 1)))
    return vectors

def inv_var_rqmc_optimization():
    # import digital_nets
    from scipy.stats import truncnorm
    '''
                    INCRMENTAL 
                    CVAR WITH RQMC      
    '''

    inv_var_rqmc_params = []

    axes = 2
    n_samples = 4096

    means = np.array([2.0, 2.0])
    variances = np.array([0.5, 0.5])
    lower_limits = np.array([-4.0, -3.0])
    upper_limits = np.array([4.0, 3.0])


    params = means
    n_params = len(params)
    axes = n_params
    center = np.zeros(axes)
    proj_radius = 1000

    epochs = 500000
    scores = np.array([])
    # from scipy.stats import qmc
    # sampler = qmc.Sobol(d=2, scramble=True)
    # np_uniform_sample = sampler.random_base2(m=14)

    '''doubt: can it give sample everytime based on the same RNG'''            

    alpha = 0.01

    v = 6
    delta = np.zeros(n_params)

    beta = 0.9
    gamma = 0.05
    epsilon = 1e-2

    norm_grad = np.array([])
    norm_params = np.array([])

    for i in range(epochs):
        # sample = sampler.random(1)[0]
        # sample = np.random.uniform(low=0, high=1, size=2)
        # sam_x = gen_trunc_gauss_samples(params[0], variances[0], lower_limits[0], upper_limits[0], 1)[0]
        # sam_y = gen_trunc_gauss_samples(params[1], variances[1], lower_limits[1], upper_limits[1], 1)[0]

        # [X, Y] = digital_nets.generate_digital_nets_base2(0, 2, 11)[0]
        sampler = qmc.Sobol(d=2, scramble=True)
        [x, y] = sampler.random_base2(m=0)[0]
        # print(X, Y)
        # x, y = X.item(), Y.item()
        print('Before inversion = ', x, y)
        sam_x = truncnorm.ppf(x, (lower_limits[0]-params[0])/variances[0], (upper_limits[0]-params[0])/variances[0], loc=params[0], scale=variances[0])
        sam_y = truncnorm.ppf(y, (lower_limits[1]-params[1])/variances[1], (upper_limits[1]-params[1])/variances[1], loc=params[1], scale=variances[1])


        # sample = gen_trunc_gauss_samples
        # print('Before inversion = ', sample[0], sample[1]) 
        # transform_to_trunc_normal(u, a, b, mu, sigma)   
        # sam_x = inverse_transform_sampling.transform_to_trunc_normal(sample[0], lower_limits[0], upper_limits[0], params[0], variances[0])
        # sam_y = inverse_transform_sampling.transform_to_trunc_normal(sample[1], lower_limits[1], upper_limits[1], params[1], variances[1])
        print('After inversion = ', sam_x, sam_y)
        r = Reward(np.array([np.array([sam_x, sam_y])]))
        scores = np.append(scores, r)
        print('iter = ', i, 'reward = ', r, end=' ')
        v_i = v
        v_update = (1-alpha)*int(r<=v)-alpha*int(r>v)
        v = v - beta*v_update
        print('\tv = ', v, end= '\t')
        delta_i = delta
        delta_update = -delta_i
        if(r<=v_i):
            gcvar = np.zeros(2)
            gcvar[0] = log_pdf_wrt_mu(sam_x, params[0], variances[0], lower_limits[0], upper_limits[0])*(r-v_i) 
            gcvar[1] = log_pdf_wrt_mu(sam_y, params[1], variances[1], lower_limits[1], upper_limits[1])*(r-v_i) 
            # print()
            # print('gcvar = ', gcvar)
            delta_update += gcvar
        delta = delta + gamma*delta_update
        print('delta = ', delta, ' r<v = ', r<=v_i, 'params = ', params)
        print('==================================================================================')
        norm_grad = np.append(norm_grad, np.linalg.norm(delta))
        # print('norm_grad = ', norm_grad)
        
        params_i = params
        params = CVaRSGD(params,np.zeros(2), delta, proj_radius, center, i+1, epsilon)
        norm_params= np.append(norm_params, np.linalg.norm(params))
        if(i%300==0):
          inv_var_rqmc_params.append(params.copy())  
    return inv_var_rqmc_params

inc_var_rqmc_params1 = inv_var_rqmc_optimization()

string = ""
for _params in inc_var_rqmc_params1:
    string = string + str(_params[0])+" "+str(_params[1])+"\n"

text_file = open("inc_rqmc_params1.txt", "w")
n = text_file.write(string)
text_file.close()


inc_var_rqmc_params2 = inv_var_rqmc_optimization()

string = ""
for _params in inc_var_rqmc_params2:
    string = string + str(_params[0])+" "+str(_params[1])+"\n"

text_file = open("inc_rqmc_params2.txt", "w")
n = text_file.write(string)
text_file.close()


inc_var_rqmc_params3 = inv_var_rqmc_optimization()

string = ""
for _params in inc_var_rqmc_params3:
    string = string + str(_params[0])+" "+str(_params[1])+"\n"

text_file = open("inc_rqmc_params3.txt", "w")
n = text_file.write(string)
text_file.close()


inc_var_rqmc_params4 = inv_var_rqmc_optimization()

string = ""
for _params in inc_var_rqmc_params4:
    string = string + str(_params[0])+" "+str(_params[1])+"\n"

text_file = open("inc_rqmc_params4.txt", "w")
n = text_file.write(string)
text_file.close()


inc_var_rqmc_params5 = inv_var_rqmc_optimization()

string = ""
for _params in inc_var_rqmc_params5:
    string = string + str(_params[0])+" "+str(_params[1])+"\n"

text_file = open("inc_rqmc_params5.txt", "w")
n = text_file.write(string)
text_file.close()