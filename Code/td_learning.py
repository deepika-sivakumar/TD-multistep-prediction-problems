# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:40:22 2019

@author: Deepika
"""
import numpy as np
import math
import matplotlib.pyplot as plt 

def gen_training_data(no_ts,no_seq):
#    no_vec = 5
    # Create a numpy array for training data
    td = np.empty((no_ts,no_seq),dtype=object)
    # Create 100 training sets
    for ts_i in range(no_ts):
        for seq_i in range(no_seq): # Create 10 sequences for each training set
            # For each sequence, we are going to start at state D
            i = 2 # (B,C,D,E,F) = (0,1,2,3,4)
            seq = np.asarray([0,0,1,0,0])
            while 1: # Create vectors until it terminates
                x_vector = np.zeros(5)
                if(i == 0 or i == 4): # If it reaches either state B or F, end the sequence
                    break
                # After making a random choice to move forward/backward, add that state to the sequence
                if(np.random.random() <= 0.5): # 0.5 probability that moves forward
                    i = i + 1
                    x_vector[i] = 1
                else:
                    i = i - 1
                    x_vector[i] = 1
                # Add each state represented as vector to that particular sequence
                seq = np.vstack((seq,x_vector))
            # Add each sequence to that particular training set
            td[ts_i][seq_i] = seq 
    return td

def td_lambda_e1(lam, alpha,ts,weights):
#    print('ts in td_lambda........:',ts)
#    weights = np.array([0.5,0.5,0.5,0.5,0.5])
    seq_delta_wt = np.array([0,0,0,0,0])
    no_seq = len(ts)
    for seq_i in range(no_seq):
        e_t = np.array([0,0,0,0,0])
        for vec_i in range(len(ts[seq_i])):
            
            s_t = np.nonzero(ts[seq_i][vec_i])[0][0] # Current state S_t
            P_t = weights[s_t]
            if(s_t == 0): # If we have reached state B
                P_t_1 = 0 # Since state B leads to terminal state A, for which we know z is 0
            elif(s_t == 4): # If we have reached state F
                P_t_1 = 1 # Since state F leads to terminal state G, for which we know z is 1
            else: # Otherwise, we see what is the next state in the sequence and take its weight
                s_t_1 = np.nonzero(ts[seq_i][vec_i+1])[0][0] # Next transition state S_t+1
                P_t_1 = weights[s_t_1]
            """
            P_t = ts[seq_i][vec_i] * weights.transpose()
            s_t = np.nonzero(ts[seq_i][vec_i])[0][0]
            if(s_t == 0):
                P_t_1 = 0
            elif(s_t == 4):
                P_t_1 = 1
            else:
                P_t_1 = ts[seq_i][vec_i+1] * weights.transpose()
            """
            # From eqn(4), e_t = gradient + lambda * e_t-1 , we need to increment eligibility for the state we are in
            e_t = ts[seq_i][vec_i] + (lam * e_t)
            # From eqn(4), delta_weight = alpha (P_t+1 - P_t) * e_t
            delta_wt = alpha * (P_t_1 - P_t) * e_t
            # We accumulate weights for that sequence
            seq_delta_wt = seq_delta_wt + delta_wt 
    # We update the weights
    weights = weights + seq_delta_wt
#    print('weights....................')
#    print(weights)
    return weights

def td_lambda_e2(lam, alpha,ts):
#    print('ts in td_lambda........:',ts)
    weights = np.array([0.5,0.5,0.5,0.5,0.5])
    seq_delta_wt = np.array([0,0,0,0,0])
    no_seq = len(ts)
    for seq_i in range(no_seq):
        e_t = np.array([0,0,0,0,0])
        for vec_i in range(len(ts[seq_i])):
            s_t = np.nonzero(ts[seq_i][vec_i])[0][0] # Current state S_t
            P_t = weights[s_t]
            if(s_t == 0): # If we have reached state B
                P_t_1 = 0 # Since state B leads to terminal state A, for which we know z is 0
            elif(s_t == 4): # If we have reached state F
                P_t_1 = 1 # Since state F leads to terminal state G, for which we know z is 1
            else: # Otherwise, we see what is the next state in the sequence and take its weight
                s_t_1 = np.nonzero(ts[seq_i][vec_i+1])[0][0] # Next transition state S_t+1
                P_t_1 = weights[s_t_1]
            # From eqn(4), e_t = gradient + lambda * e_t-1 , we need to increment eligibility for the state we are in
            e_t = ts[seq_i][vec_i] + lam * e_t
            # From eqn(4), delta_weight = alpha (P_t+1 - P_t) * e_t
            delta_wt = alpha * (P_t_1 - P_t) * e_t
            # We accumulate weights for that sequence
            seq_delta_wt = seq_delta_wt + delta_wt 
        # We update the weights
        weights = weights + seq_delta_wt
#    print('weights....................')
#    print(weights)
    return weights

def expt():
    np.random.seed(12345)
#    np.random.seed(999999999)
#    np.random.seed(9)
    no_ts = 100 # No. of training sets
    no_seq = 10# No. of sequences
    ideal_wts = np.array([1/6,1/3,1/2,2/3,5/6]) # Ideal predictions for states (B,C,D,E,F)
    # Generate the training data
    td = gen_training_data(no_ts,no_seq)
#    print('****************************Training Data******************************')
#    print(td)
    #print('td.shape:',td.shape)
#    print('*****************************Experiment 1******************************')
    # Lambda
    lam_e1 = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    alpha_e1 = 0.001
    convergence = 0.001
#    alpha_e1 =0.01
    
    # Array to store Average RMSE for different Lambda values
    avg_rmse_e1 = np.empty(11,dtype=float)
    
    for l in range(len(lam_e1)):
        # Array to store the rmse values for each training set
        rmse = np.zeros(no_ts)
#        rmse_new = 0
        # Iterate through the training data and get predictions for each training set
        for ts_i in range(no_ts):
            # Initial weights for each training set
            weights_e1 = np.array([0.5,0.5,0.5,0.5,0.5])
            prev_weights_e1 = np.zeros(5)
            # Train a set until convergence
            while True:
                prev_weights_e1 = weights_e1
                weights_e1 = td_lambda_e1(lam_e1[l],alpha_e1,td[ts_i],weights_e1)  
                if(np.max(np.absolute(prev_weights_e1 - weights_e1)) < convergence):
                    break
            # Calculate the RMSE for each training set
            rmse[ts_i] = np.sqrt(((ideal_wts - weights_e1) ** 2).sum()/5)
#            print('weights after Training set',ts_i,':',weights_e1)
        # Average RMSE over 100 training sets
        avg_rmse_e1[l] = np.average(rmse)
#        avg_rmse_e1[l] = rmse_new/(no_ts)
#        print('Average RMSE for Lambda=',lam_e1[l],':',avg_rmse_e1[l])
#    print('Alpha:',alpha_e1)
    # Plot Figure 3
    plt.title('Experiment 1: Figure 3')
    plt.xlabel('Lambda')
    plt.ylabel('ERROR USING BEST ALPHA')
    plt.plot(lam_e1,avg_rmse_e1,'.-')
    plt.text(lam_e1[7], avg_rmse_e1[10], 'Widrow-Hoff', fontsize=12)
#    plt.show()
    plt.savefig('fig3.png')
    plt.close()
    
#    print('****************************Experiment 2****************************')
    alphas_e2 = np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6])
    lam_e2 = np.array([0,0.3,0.8,1])
    # Array to store Average RMSE for different Lambda values
    rmse_lam_e2 = np.zeros((len(lam_e2),len(alphas_e2)))
    
    for l in range(len(lam_e2)):
#        print('Lambda:',lam_e2[l])
        # Array to store average RMSE over 100 training sets for each alpha of that lambda
        avg_rmse_e2 = np.zeros(len(alphas_e2),dtype=float)
        for a in range(len(alphas_e2)):
#            print('Alpha:',alphas_e2[a])
            # Array to store the rmse values for each training set
            rmse = np.zeros(no_ts)
            # Iterate through the training data and get predictions for each training set
            for ts_i in range(no_ts):
                pred_wts = td_lambda_e2(lam_e2[l],alphas_e2[a],td[ts_i])  
                # Calculate the RMSE for each training set
                rmse[ts_i] = math.sqrt(((ideal_wts - pred_wts) ** 2).sum()/5)
#                rmse[ts_i] = math.sqrt(((ideal_wts - pred_wts) ** 2).sum())
            # Average RMSE over 100 training sets
            avg_rmse_e2[a] = np.average(rmse)
#            print('Average RMSE for Alpha=',alphas_e2[a],':',avg_rmse_e2[a])
        rmse_lam_e2[l] = avg_rmse_e2
#    print('RMSE for various Lambda & alpha:')
#    print(rmse_lam_e2)
#    print('Figure 4 - Lambda:',)
    # Plot Figure 4
    plt.title('Experiment 2: Figure 4')
    plt.xlabel('ALPHA')
    plt.ylabel('ERROR')
#    plt.ylim(top=0.8, bottom=0.0)
    
    plt.plot(alphas_e2,rmse_lam_e2[0],'.-',label='Lambda = 0')
    plt.plot(alphas_e2,rmse_lam_e2[1],'.-',label='Lambda = 0.3')
    plt.plot(alphas_e2,rmse_lam_e2[2],'.-',label='Lambda = 0.8')
    plt.plot(alphas_e2,rmse_lam_e2[3],'.-',label='Lambda = 1')
    plt.legend()
#    plt.show()
    plt.savefig('fig4.png')
    plt.close()
#    print('***************Figure 5**************')
    #alphas_f5 = np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6])
    #lam_f5 = np.array([0,0.3,0.8,1])
    x = []
    alp = 0.0
    while alp < 0.61:
        x.append(alp) # you push your element here, as of now I have push i
        alp = alp + 0.01
    alphas_f5 = np.array(x)
    rmse_lam_f5 = np.zeros((len(lam_e1),len(alphas_f5)))
    # Array to store Average RMSE for different Lambda values
    
    for l in range(len(lam_e1)):
#        print('Lambda:',lam_e1[l])
        # Array to store average RMSE over 100 training sets for each alpha of that lambda
        avg_rmse_f5 = np.zeros(len(alphas_f5),dtype=float)
        for a in range(len(alphas_f5)):
            # Array to store the rmse values for each training set
            rmse = np.zeros(no_ts)
            # Iterate through the training data and get predictions for each training set
            for ts_i in range(no_ts):
                pred_wts = td_lambda_e2(lam_e1[l],alphas_f5[a],td[ts_i])  
                # Calculate the RMSE for each training set
                rmse[ts_i] = math.sqrt(((ideal_wts - pred_wts) ** 2).sum()/5)
#                rmse[ts_i] = math.sqrt(((ideal_wts - pred_wts) ** 2).sum())
            # Average RMSE over 100 training sets
            avg_rmse_f5[a] = np.average(rmse)
#            print('Average RMSE for Alpha=',alphas_e2[a],':',avg_rmse_f5[a])
        rmse_lam_f5[l] = avg_rmse_f5
#    print('RMSE for various Lambda & alpha:')
#    print(rmse_lam_f5)
    best_alpha_error = rmse_lam_f5.min(axis=1)
#    print('best_alpha_error')
#    print(best_alpha_error)
    
    #Plot Figure 5
    best_alpha_error = rmse_lam_f5.min(axis=1)
    plt.title('Experiment 2: Figure 5')
    plt.xlabel('LAMBDA')
    plt.ylabel('ERROR USING BEST ALPHA')
    plt.plot(lam_e1,best_alpha_error,'.-')
    plt.text(lam_e1[7], best_alpha_error[10], 'Widrow-Hoff', fontsize=12)
#    plt.show()
    plt.savefig('fig5.png')
    plt.close()

# Conduct experiments    
expt()