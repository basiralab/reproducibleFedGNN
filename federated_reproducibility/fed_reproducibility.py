import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import pandas as pd
import pickle
import os
from Analysis import W_histogram, W_heatmap, get_weights, Top_biomarkers, sim


def get_top_biomarkers(model, dataset, mode, hospital_id, K_i):
    """
    Parameters
    ----------
    model : model name (DiffPool or GCN)
    dataset : dataset
    mode : mode of learning (Federated or Baseline)
    hospital_id : id of a particular hospital
    
    Description
    ----------
    Returns the top K biomarkers of a specific hospital 
    """  
    weights = get_weights(model, dataset, mode, hospital_id)
    top_k_biomarkers = Top_biomarkers(np.abs(weights), K_i)
    return top_k_biomarkers

def save_reproducibility(dataset, mode, hospital_num, image_input_type):
    """
    Parameters
    ----------
    dataset : dataset
    mode : mode of learning (Federated or Baseline)
    hospital_num : number of hospitals
    image_input_type : type of image data input
    
    Description
    ----------
    Plots and saves:
    Hospital-specific reproducibility matrices
    Averaged reproducibility matrices
    Bar plots of weights (learned by the most reproducible model) for connectivity datasets
    Heatmap of weights (learned by the most reproducible model) for image based datasets
    """  
    if not os.path.exists("./results/reproducibility"):
        os.makedirs("./results/reproducibility")
        os.makedirs("./results/reproducibility/average")
        os.makedirs("./results/reproducibility/hospital_specific")
    width = 5
    height = 5
    models = ["GCN", "DiffPool"]
    ks = [20]
    for k in ks:
        avg_data = np.zeros((len(models), len(models)))
        for hi in range(hospital_num):
            data = np.zeros((len(models), len(models)))
            for i in range(len(models)):
                for j in range(len(models)):
                    top_biomarkers_i = get_top_biomarkers(models[i], dataset, mode, hi , k)
                    top_biomarkers_j = get_top_biomarkers(models[j], dataset, mode, hi , k)
                    data[i,j] = sim(top_biomarkers_i, top_biomarkers_j)
                    avg_data[i,j] += data[i,j]
            
            data_frame = pd.DataFrame(data, index = [i for i in models], columns = [i for i in models])
            plt.figure(figsize=(width,height))
            plt.title("Reproducibility Matrix of Hospital-" + str(hi + 1) + " (" + mode + ")")
            sns.heatmap(data_frame, annot=True, vmin=0.5, vmax=1)
            plt.savefig("./results/reproducibility/hospital_specific/hospital-"+ str(hi) +"_" + mode + "_" + dataset + "_k" + str(k) + ".png", dpi=900)
            plt.close()
        avg_data = avg_data / hospital_num
        data_frame = pd.DataFrame(avg_data, index = [i for i in models], columns = [i for i in models])
        sns.set(font_scale=2)
        plt.figure(figsize=(width,height))
        plt.title(mode + " Rep. Matrix")
        sns.heatmap(data_frame, annot=True, vmin=0.5, vmax=1, annot_kws={"fontsize": 24}, cmap='Blues')
        plt.savefig("./results/reproducibility/average/average_" + mode + "_" + dataset + "_k" + str(k) + ".png", dpi=900)
        plt.show()
        plt.close()
        
    if mode == "Federated":
        if "MNIST" in dataset and image_input_type == "adj":
            W_heatmap(dataset, hospital_num)
        elif "ASDNC" in dataset or "Demo" in dataset:
            W_histogram(dataset,0,True, hospital_num)
