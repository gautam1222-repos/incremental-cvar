from tetris import Tetris
from policy import Policy
import numpy as np

def run(policy, timesteps):
    
    env = Tetris()
    log_trans_prob = 0

    state = env.get_initial_state()
    
    for _ in range(timesteps):
        softmax_arr = policy.softmax(state)
        action = 31
        max_prob_trans = np.max(softmax_arr)
        
        for idx in range(len(softmax_arr)):
            if softmax_arr[idx] == max_prob_trans:
                action = idx
                break
        
        if action in range(20) and action not in env.valid_actions(env.grid, env.current_piece):
            action = 31
        

        # print(action)
        next_state = env.step(action)
        log_trans_prob += policy.diff_log_policy_wrt_params(action, state)
        
        
        print('-'*10,'>', 'timestep =', _+1, 'env_status = ', env.run)
        if not(env.run):
            break
        
        state = next_state
    
    env.close()
    total_score = env.score
    log_prob_traj = log_trans_prob
    del(env)
    return total_score, log_prob_traj

def CVaRSGD(params, d_params, proj_radius, center, eta):
    
    # params2 += eta*d_params
    # params += (1/(t+10))*(params2-params)
    params += eta*d_params
    if(np.linalg.norm(params-center)>proj_radius):
        # print('projection')
        norm_val = np.linalg.norm(params-center)
        vec = params-center
        vec /= norm_val
        params = center + proj_radius*vec
    
    return params



def inc_var_cvar_optimization(alpha, var_step_length, GCVaR_step_length, params_step_length, epochs):

    env_action_space = range(32)
    policy = Policy(env_action_space)

    var = 10
    delta = np.zeros(21)
    delta = delta.reshape((-1,1))
    proj_radius = 100    
    center = np.zeros(21)
    center = center.reshape((-1,1))

    params_string = ""

    for t in range(epochs):

        print('='*35)
        print('-'*10, 'ITERATION :', t+1, '-'*10)
        print('='*35)

        score, log_total_trans = run(policy, 100)

        # var increment
        v_i = var
        v_update = (1-alpha)*int(score<=var)-alpha*int(score>var)
        var = var - var_step_length*v_update
        
        #gcvar increment
        delta_i = delta
        delta_update = -delta_i
        if(score<=v_i):
            gcvar = np.zeros(21)
            gcvar = log_total_trans
            delta_update += gcvar
        
        policy.params = CVaRSGD(policy.params, gcvar, proj_radius, center, params_step_length)

        print('policy_params = ', policy.params)

        for params_j in policy.params:
            params_string = params_string + str(params_j) + " "
        
        params_string = params_string + "\n"



    # Change this everytime opened    

    text_file = open("params1.txt", "w")
    n = text_file.write(params_string)
    text_file.close()


alpha = 0.01

var_step_length = 0.9
gcvar_step_length = 0.09
params_step_length = 9*1e-3
epochs = 500000



inc_var_cvar_optimization(alpha, var_step_length, gcvar_step_length, params_step_length, epochs)
