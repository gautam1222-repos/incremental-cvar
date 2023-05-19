from tetris import Tetris
from policy import Policy
import numpy as np

def run(policy, timesteps):
    
    env = Tetris()
    log_trans_prob = 0

    state = env.get_initial_state()
    
    for _ in range(timesteps):
        print('-'*20,'>timestep: ',_+1)
        softmax_arr = policy.softmax(state)
        action = 31
        max_prob_trans = np.max(softmax_arr)
        
        for idx in range(len(softmax_arr)):
            if softmax_arr[idx] == max_prob_trans:
                action = idx
                break
        print('piece pos = ', env.current_piece.x, ',', env.current_piece.y)
        print(env.allowed_actions(env.valid_actions(env.grid, env.current_piece), env.current_piece))
        if action in range(20) and action not in env.allowed_actions(env.valid_actions(env.grid, env.current_piece), env.current_piece):
            action = 31
        print('*'*40)
        print('action = ', action)
        print('*'*40)
        next_state = env.step(action)
        log_trans_prob += policy.diff_log_policy_wrt_params(action, state)

        
        if not(env.run):
            break
        
        state = next_state
        
    env.close()
    total_score = env.score
    log_prob_traj = log_trans_prob
    del(env)
    return total_score, log_prob_traj

def CVaRSGD(params, d_params, proj_radius, center, eta):
    
    params += eta*d_params
    if(np.linalg.norm(params-center)>proj_radius):
        print('projection')
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
    proj_radius = 10000    
    center = np.zeros(21)
    center = center.reshape((-1,1))

    delta_string = ""

    file_cnt = 0

    for t in range(epochs):

        
        score, log_total_trans = run(policy, 400)

        # var increment
        v_i = var
        v_update = (alpha-1)*int(score<=var)+alpha*int(score>var)
        var = var + var_step_length*v_update
        print('var = ', var)

        #gcvar increment
        delta_i = delta
        delta_update = -delta_i
        if(score<=v_i):
            gcvar = np.zeros(21)
            gcvar = log_total_trans
            delta_update += gcvar
        delta = delta + GCVaR_step_length*delta_update

        # parameter update
        policy.params = CVaRSGD(policy.params, delta, proj_radius, center, params_step_length)


        for delta_j in delta:
            delta_string = delta_string + str(delta_j) + " "
        
        delta_string = delta_string + "\n"

        if (t+1)%100 == 0:
            file_cnt += 1
            text_file = open("Grad"+str(file_cnt)+".txt", "w")
            text_file.write(delta_string+"\n"+"\n"+str(policy.params))
            text_file.close()


        print('='*35)
        print('-'*10, 'ITERATION :', t+1, '-'*10)
        txt_file = open("grad_iteration_num.txt","w")
        txt_file.write("Iteration: "+str(t+1)+"\n"+str(delta_string))
        txt_file.close()
        print('='*35)

    # Change this everytime opened    

    text_file = open("Grads_Final_1.txt", "w")
    n = text_file.write(delta_string)
    text_file.close()


alpha = 0.01

var_step_length = 0.9
gcvar_step_length = 0.05
params_step_length = 1e-2
epochs = 500000



inc_var_cvar_optimization(alpha, var_step_length, gcvar_step_length, params_step_length, epochs)
