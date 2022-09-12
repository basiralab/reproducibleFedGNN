from tqdm import tqdm
import copy
import numpy as np
from fed_localmodel import average_weights

def federatedLearning(global_model, hospitals, global_args, dataset, model):
    """
    Parameters
    ----------
    global_model : broadcasted deep copy of the global model
    hospitals : hospital-specific local models
    global_args : running arguments
    
    Description
    ----------
    This method applies the federated learning on hospital-specific models
    saves the accuracy and loss results to .npy files
    """  
    hospital_losses = [[], [], []]
    train_losses = []
    test_accs = []
    for i in tqdm(range(global_args.comms_round)):
        print("ROUND:", i)
        local_weights = []
        avg_acc = 0
        avg_loss = 0
        for h, hospital in enumerate(hospitals):
            local_w, test_acc, train_loss = hospital.localUpdate(copy.deepcopy(global_model))
            avg_loss += np.mean(train_loss)
            avg_acc += test_acc
            hospital_losses[h].append(train_loss)
            local_weights.append(local_w)
        
        train_losses.append(avg_loss / global_args.hospital_num)
        test_accs.append(avg_acc / global_args.hospital_num)

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)
    
    for hl in range(global_args.hospital_num):
        hospital_losses[hl] = np.mean(hospital_losses[hl], axis=0)
    hospital_losses = np.array(hospital_losses)
    train_losses = np.array(train_losses)
    test_accs = np.array(test_accs)
    with open("./results/data/"+ dataset + "_" + model + "_fed_loss.npy", 'wb') as f:
        np.save(f, train_losses)
    with open("./results/data/"+ dataset +  "_" + model + "_fed_acc.npy", 'wb') as f:
        np.save(f, test_accs)
    with open("./results/data/" + dataset + "_" + model + "_fed_hospital_loss.npy", 'wb') as f:
        np.save(f, hospital_losses)

def baseLine(global_model, hospitals, global_args, dataset, model):
    """
    Parameters
    ----------
    global_model : broadcasted deep copy of the global model
    hospitals : hospital-specific local models
    global_args : running arguments
    
    Description
    ----------
    This method applies the non-federated learning on hospital-specific models
    saves the accuracy and loss results to .npy files
    """ 
    locals = [copy.deepcopy(global_model) for i in range(global_args.hospital_num)]
    hospital_losses = [[], [], []]
    train_losses = []
    test_accs = []
    for i in tqdm(range(global_args.comms_round)):
        print("ROUND:", i)
        avg_acc = 0
        avg_loss = 0
        for h, hospital in enumerate(hospitals):
            local_w, test_acc, train_loss = hospital.localUpdate(copy.deepcopy(locals[h]))
            avg_loss += np.mean(train_loss)
            avg_acc += test_acc
            hospital_losses[h].append(train_loss)
            locals[h].load_state_dict(local_w)
        
        train_losses.append(avg_loss / global_args.hospital_num)
        test_accs.append(avg_acc / global_args.hospital_num)

    for hl in range(global_args.hospital_num):
        hospital_losses[hl] = np.mean(hospital_losses[hl], axis=0)
    hospital_losses = np.array(hospital_losses)
    train_losses = np.array(train_losses)
    test_accs = np.array(test_accs)
    with open("./results/data/"+ dataset +  "_" + model + "_base_loss.npy", 'wb') as f:
        np.save(f, train_losses)
    with open("./results/data/"+ dataset +  "_" + model + "_base_acc.npy", 'wb') as f:
        np.save(f, test_accs)
    with open("./results/data/"+ dataset + "_" + model + "_base_hospital_loss.npy", 'wb') as f:
        np.save(f,hospital_losses)