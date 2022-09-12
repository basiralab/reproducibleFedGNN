import pickle
from matplotlib.colorbar import ColorbarBase
import numpy as np
import os 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import matplotlib
import torch

## This variable used to check if the models are trained.
is_trained = False

def Models_trained(dataset, view):
    err = False
    models = ["diffpool","gat","gcn","sag","gunet"]
    for model in models:
        if model == "sag":
            path_cv = model+"/weights/W_MainModel_5Fold_"+dataset+"_"+model+"_view_"+str(view)+"_CV_"+str(4)+".pickle"
            path_few = "./"+str(model)+"/weights/W_Few_Shot_"+ dataset +"_"+model+"_view_"+str(view)+"_"+str(99)+".pickle"
        else:
            path_cv = model+"/weights/W_MainModel_5Fold_"+dataset+"_"+model+"_CV_"+str(4)+"_view_"+str(view)+".pickle"
            path_few = "./"+str(model)+"/weights/W_Few_Shot_"+ dataset +"_"+model+str(99)+"_view_"+str(view)+".pickle"
        
        if not os.path.exists(path_few):
            err = True
        if not os.path.exists(path_cv):
            err = True
    return err
            

def new_folder(model):
    """
    Parameters
    ----------
    model : GNN model (diffpool, gat, gcn, gunet or sag)
    
    Description
    ----------
    Creates GNN directories if not exist.
    """
    if not os.path.exists(model):
        os.makedirs(model)
        os.makedirs("./"+model+"/"+"weights")
        os.makedirs("./"+model+"/"+"training_loss")
        os.makedirs("./"+model+"/"+"models")
        os.makedirs("./"+model+"/"+"Labels_and_preds")

def Mean_W_Two_shot(dataset,model, view):
    
    """
    Parameters
    ----------
    dataset : dataset
    model : GNN model (diffpool, gat, gcn, gunet or sag)
    view : index of cortical morphological network
    
    Description
    ----------
    This method returns the average weight of 100 GNN models trained with 100 different sets of 2 samples from each label.
    """
    
    W = []
    for i in range(100):
          if model == "sag":
              file_to_read = open("./"+str(model)+"/weights/W_Few_Shot_"+ dataset +"_"+model+"_view_"+str(view)+"_"+str(i)+".pickle", "rb")
          else:
              file_to_read = open("./"+str(model)+"/weights/W_Few_Shot_"+ dataset +"_"+model+str(i)+"_view_"+str(view)+".pickle", "rb")
          loaded_dict = pickle.load(file_to_read)
          if model == "diffpool" or model == "sag" or model == "gunet":
              nmpy = loaded_dict["w"].detach().numpy()
              W.append(np.mean(nmpy, axis=1))
          else:
              W.append(loaded_dict["w"].detach().numpy()[0])
    W = np.array(W)
    result = np.mean(W, axis=0)
    return result


def Mean_W_Cv(dataset,model, view):
    
    """
    Parameters
    ----------
    dataset : dataset
    model : GNN model (diffpool, gat, gcn, gunet or sag)
    view : index of cortical morphological network 
    
    Description
    ----------
    This method returns the average weight of GNN models trained with 5 different folds.
    
    """
    
    W = []
    for i in range(5):
        if model == "sag":
            file_to_read = open(model+"/weights/W_MainModel_5Fold_"+dataset+"_"+model+"_view_"+str(view)+"_CV_"+str(i)+".pickle", "rb")
        else:
            file_to_read = open(model+"/weights/W_MainModel_5Fold_"+dataset+"_"+model+"_CV_"+str(i)+"_view_"+str(view)+".pickle", "rb")
        
        loaded_dict = pickle.load(file_to_read)
        
        if model == "diffpool" or model == "sag" or model == "gunet":
            nmpy = loaded_dict["w"].detach().numpy()
            W.append(np.mean(nmpy, axis=1))
        else:
            W.append(loaded_dict["w"].detach().numpy()[0])
    W = np.array(W)
    result = np.mean(W, axis=0)
    return result

def get_weights(model, dataset, mode, hospital_id):
     
    """
    Parameters
    ----------
    model : GNN model (diffpool or gcn)
    dataset : dataset
    mode : mode of learning technique (federated or non-federated) 
    hospital_id : id of a hospital for extracting hospital-specific learned weights
    
    Description
    ----------
    This method returns the learned weights of hospital-specific GNN model
    
    """   

    path = "./{}/weights/W_{}_hospital-{}_{}.pickle".format(model.lower(),mode,hospital_id, dataset)
    with open(path,'rb') as f:
        weights = pickle.load(f)
    if model == 'DiffPool':
        weights_vector = torch.mean(weights['w'], 1).detach().numpy()
    if model == 'GCN':
        weights_vector = weights['w'].squeeze().detach().numpy()
    return weights_vector

def mean_W_federated(model, dataset, hospital_num, mode):
    """
    Parameters
    ----------
    model : GNN model (diffpool or gcn)
    dataset : dataset
    hospital_num : number of hospitals
    mode : mode of learning technique (federated or non-federated) 
    
    Description
    ----------
    This method returns the average of learned weights of hospital-specific GNN models
    
    """      
    W = []
    for hospital_id in range(hospital_num):
        w = get_weights(model, dataset, mode, hospital_id)
        W.append(np.abs(w))
    W = np.array(W)
    result = np.mean(W, axis=0)
    return result

def Top_biomarkers(weights,n):
    
    """
    Parameters
    ----------
    weights : Average weight of a GNN architecture.
    n : number of biomarkers
    
    Description
    ----------
    Extracts the top K biomarkers from 35 regions.
    """
    
    result = []
    w_sorted = weights.argsort()
    for i in range(1,n):
      result.append(w_sorted[-1*i])
    return result

def sim(nodes1, nodes2):
    
    """
    Parameters
    ----------
    nodes1: Top K biomarkes of GNN Structure 1.
    nodes2: Top K biomarkes of GNN Structure 2.
    
    Description
    ----------
    Returns the overlap ratio between Top K biomarkes of two GNN architectures.
    """
    
    if len(nodes1)==len(nodes2):
        counter = 0
        for i in nodes1:
            for k in nodes2:
                if i==k:
                    counter+=1
        return counter/len(nodes1)
    else:
        print('nodes vectors are not caompatible')

def Bio_dictionary(n, dataset, view, federated = False, hospital_num = 3):
    
    """
    Parameters
    ----------
    n: number of biomarkers
    dataset: dataset
    view: index of cortical morphological network
    
    Description
    ----------
    Saves the top K biomarkers of all GNN structures to results directory.
    """
    
    if not os.path.exists('results'):
        os.makedirs('results')
    result_d = {}
    if federated:
        models = ["DiffPool", "GCN"]
        for model in models:
            Main_Results = mean_W_federated(model, dataset, hospital_num, "Federated")
            result_d["Federated_"+model+"_"+dataset] = Top_biomarkers(Main_Results,n)
    else:
        models = ["diffpool","gat","gcn","sag","gunet"]
        for model in models:
            Main_Results = Mean_W_Cv(dataset,model, view)
            result_d["Cv_"+model+"_"+dataset] = Top_biomarkers(Main_Results,n)
            Few_Results = Mean_W_Two_shot(dataset, model, view)
            result_d["Few_"+model+"_"+dataset] = Top_biomarkers(Few_Results,n)
    with open("./results/Top_"+str(n)+"_biomarkers_view_"+str(view)+"_dataset_"+dataset+".pickle", 'wb') as f:
      pickle.dump(result_d, f)

def Rep_matrix(dataset, n, view, federated = False, hospital_num = 3):
    
    """
    Parameters
    ----------
    n: number of biomarkers
    dataset: dataset
    view: index of cortical morphological network
    
    Description
    ----------
    Constructs Reproducibility matrix
    """
    
    
    if not os.path.exists("./results/Top_"+str(n)+"_biomarkers_view_"+str(view)+"_dataset_"+dataset+".pickle"):
      Bio_dictionary(n, dataset, view, federated, hospital_num)
     
    file_to_read = open("./results/Top_"+str(n)+"_biomarkers_view_"+str(view)+"_dataset_"+dataset+".pickle", "rb")
    loaded_dict = pickle.load(file_to_read) 
    nodes = []
    if federated:
        models = ["DiffPool", "GCN"]
        for model in models:
            nodes.append(loaded_dict["Federated_"+model+"_"+dataset])
    else:
        models = ["diffpool","gat","gcn","sag","gunet"]
        for model in models:
            nodes.append(loaded_dict["Cv_"+model+"_"+dataset])
            nodes.append(loaded_dict["Few_"+model+"_"+dataset])
    
    
    matrix = []
    for x in nodes:
      row = []
      for y in nodes:
        row.append(sim(x,y))
      matrix.append(row)
    return matrix

def AvgRep_matrix(dataset, K, hospital_num):
    """
    Parameters
    ----------
    dataset : dataset
    K : number of biomarkers
    hospital_num : number of hospitals
    
    Description
    ----------
    Returns the average of all hospital-specific reproducibility matrices
    """ 
    models = ["DiffPool", "GCN"]
    avg_data = np.zeros((len(models), len(models)))
    for hi in range(hospital_num):
        data = np.zeros((len(models), len(models)))
        for i in range(len(models)):
            for j in range(len(models)):
                w_i = get_weights(models[i], dataset, "Federated", hi)
                top_biomarkers_i = Top_biomarkers(np.abs(w_i), K)
                w_j = get_weights(models[j], dataset, "Federated", hi)
                top_biomarkers_j = Top_biomarkers(np.abs(w_j), K)
                data[i,j] = sim(top_biomarkers_i, top_biomarkers_j)
                avg_data[i,j] += data[i,j]
    avg_data /= 3
    return avg_data

def Rep_heatmap(dataset, view):
  
    """
    Parameters
    ----------
    number_of_node: number of biomarkers
    dataset: dataset
    view: index of cortical morphological network
    
    Description
    ----------
    Plots a 2D heatmap of Reproducibility matrix
    """
  
    
    if not os.path.exists('results'):
        os.makedirs('results')
    
    Ks = [5,10,20]
    matrix = 0
    for k in Ks:
        matrix += np.array(Rep_matrix(dataset, k, view))
    
    matrix /=3
        
    x_labels = ["Diffpool", "Diffpool\nfew shot", "GAT", "GAT\nfew shot", "GCN", "GCN\nfew shot", "SAG", "SAG\nfew shot", "GUN", "GUN\nfew shot"]  
    
    df_cm = pd.DataFrame(matrix, index = [i for i in x_labels],
                      columns = [i for i in x_labels])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True ,vmin=0, vmax=1)
    
    plt.title('Dataset:'+  dataset  +' Average Reproducibility Matrix')
    plt.savefig("./results/Weight_matrix_Average_Rep"+"_"+dataset+".png")
    plt.show()
    plt.close()

def strength_centrality(matrix):
    
    """
    Parameters
    ----------
    matrix: Reproducibility matrix of GNN architectures
    
    Description
    ----------
    Returns strengths of GNN architectures.
    """    
    normalized_str = []
    for row in matrix:
        sum_w = np.sum(row)-1
        sum_w /= len(row)-1
        normalized_str.append(sum_w)
    return normalized_str
     
def Rep_scores(dataset, view, federated = False, hospital_num = 3):
    
    """
    Parameters
    ----------
    dataset: dataset
    
    Description
    ----------
    Returns the average strenghs of GNN architectures with Top 5, 10 and 20 biomarkers.
    """    
    if federated:
        Ks = [20]
    else:
        Ks = [5,10,20]
    vectors = []
    for K in Ks:
        if federated:
            matrix = AvgRep_matrix(dataset, K, hospital_num)
        else:
            matrix = Rep_matrix(dataset, K, view, federated, hospital_num)
        vector = strength_centrality(matrix)
        vectors.append(vector)   
    avg = np.mean(vectors, axis = 0)
    return avg
    
def Rep_histograms(dataset, view):
    """
    Parameters
    ----------
    dataset: dataset
    
    Description
    ----------
    Plots histogram of reproducibility scores.
    """    
    
    rep = Rep_scores(dataset, view)   
    x_labels = ["Diffpool", "Diffpool\nfew shot", "GAT", "GAT\nfew shot", "GCN", "GCN\nfew shot", "SAG", "SAG\nfew shot", "GUN", "GUN\nfew shot"] 
    low = min(rep)
    high = max(rep)
    
    fig, axs = plt.subplots(1, 1, figsize=(8, 3), sharey=True)
    if "LH" in dataset:
        axs.bar(x_labels, rep, color = ('indigo', 'tab:purple'))
    else:
        axs.bar(x_labels, rep, color = ('red', 'lightcoral'))
    matplotlib.rc('xtick', labelsize=10) 
    plt.ylabel("reproducibility score", fontsize=12)
    plt.ylim([low-0.2*(high-low),high+0.1*(high-low)])
    plt.savefig("./results/Dataset_"+dataset+".png", bbox_inches='tight')

def Region_W(dataset, view, federated = False, hospital_num = 3):
    
    """
    Parameters
    ----------
    dataset: dataset
    
    Description
    ----------
    Returns the weights of the most reproducible GNN architectures.
    """  
    if federated:
        GNNs = ["DiffPool", "GCN"]
        best = GNNs[np.argmax(Rep_scores(dataset, view, federated, hospital_num))]
        # print(Rep_scores(dataset, view, federated, hospital_num))
    else:
        GNNs = ["Diffpool", "Diffpool\nfew shot", "GAT", "GAT\nfew shot", "GCN", "GCN\nfew shot", "SAG", "SAG\nfew shot", "GUNET", "GUNET\nfew shot"] 
        best = GNNs[np.argmax(Rep_scores(dataset, view, federated, hospital_num))].lower()
    if federated:
        weights = mean_W_federated(best, dataset, hospital_num, "Federated")
    else:
        if "few" in best:
            weights = Mean_W_Two_shot(dataset, best.split("\n")[0], view)
        else:
            weights = Mean_W_Cv(dataset, best, view)
    return weights
    
def  W_histogram(dataset, view, federated = False, hospital_num=3):
    
    """
    Parameters
    ----------
    dataset: dataset
    
    Description
    ----------
    Plots the weights of the most reproducible GNN architecture.s
    """  
    
    Weights = Region_W(dataset, view, federated, hospital_num)
 

    if "ASDNC" in dataset:    
        N = 35
        labels = [
        "Bank of the Superior Temporal Sulcus",#1
        "Caudal Anterior-cingulate Cortex",#2
        "Caudal Middle Frontal Gyrus",#3
        "Unmeasured Corpus Callosum",#4
        "Cunesus Cortex",#5
        "Entorhinal Cortex",#6
        "Fusiform Gyrus",#7
        "Inferior Parietal Cortex",#8
        "Inferior Temporal Gyrus",#9
        "Isthmus-cingulate Cortex",#10
        "Lateral occipital cortex",#11
        "Lateral orbital frontal cortex",#12
        "Lingual gyrus",#13
        "Medial orbital frontal cortex",#14
        "Middle temporal gyrus",#15
        "Parahippocampal gyrus",#16
        "Paracentral lobule",#17
        "Pars opercularis",#18
        "Pars orbitalis",#19
        "Pars triangularis",#20
        "Pericalcarine cortex",#21
        "Postcentral gyrus",#22
        "Posterior-cingulate cortex",#23
        "Precentral gyrus",#24
        "Precuneus cortex",#25
        "Rostral anterior cingulate cortex",#26
        "Rostral middle frontal gyrus",#£7
        "Superior frontal gyrus",#28
        "Superior parietal cortex",#29
        "Superior temporal gyrus",#30
        "Supramarginal gyrus",#31
        "Frontal pole",#32
        "Temporal pole",#33
        "Transverse temporal cortex",#34
        "Insula cortex"#35
        ]
    elif "MNIST" in dataset:
        N = 35
        labels = [i for i in range(1,29)]
    elif "Demo" in dataset:
        N = 35
        labels = [i for i in range(1,36)]

    ind = np.arange(N)
    
    plt.figure(figsize=(25,10))
    
    width = 0.3   
    fig = plt.figure(1)    
    ax = fig.add_subplot(111)
    high = max(Weights)
    low = min(Weights)
    # colors = matplotlib.cm.copper()
    # plot = plt.scatter(Weights, Weights, c = Weights, cmap = 'copper')
    # plt.clf()
    # plt.colorbar(plot)
    cmap = plt.get_cmap("bwr")
    rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
    if "LH" in dataset:
        plt.bar(ind, Weights , width, label= dataset, color = cmap(rescale(Weights)))
    else:
        plt.bar(ind, Weights, width, label= dataset, color = cmap(rescale(Weights)))
    plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(Weights),max(Weights))))
    #maxs = [max(Lh_w), max(Rh_w)]
    #mins = [min(Lh_w), min(Rh_w)]
    
    
    
    matplotlib.rc('xtick', labelsize=18) 
    matplotlib.rc('ytick', labelsize=18) 
    
    ax.set_xticklabels(labels, rotation = 80, ha="right", **{'fontname': 'Arial', 'fontsize': 18})
    
    dx = 35/300.; dy = 0/300. 
    offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform()+ offset)
    
    plt.xticks(ind + width / 2, labels)
    plt.ylim([low-0.01*(high-low),high+0.01*(high-low)])
    # Finding the best position for legends and putting it
    #plt.xlabel("regions", fontsize=25)
    plt.ylabel("average weights", fontsize=25)
    #plt.xticks(rotation=80)
    plt.legend(loc= "upper center")
    plt.savefig("./results/"+dataset+"_regions.png", bbox_inches='tight', dpi = 900)
    plt.savefig("./results/"+dataset+"_regions.svg", bbox_inches='tight', dpi = 900)
    plt.show()
        
    
    
def W_heatmap(dataset, hospital_num = 3):
    """
    Parameters
    ----------
    dataset : dataset
    hospital_num : number of hospitals
    
    Description
    ----------
    Plots the heatmap of weights (specific for image based medical data)
    """  
    Weights = Region_W(dataset, 0, True, hospital_num)
    Weights_mat = np.reshape(Weights, (28,28))
    data_frame = pd.DataFrame(Weights_mat, index = [i for i in range(1,29)], columns = [i for i in range(1,29)])
    plt.figure(figsize=(15,15))
    plt.title("Heatmap of Weights")
    sn.heatmap(data_frame, annot=False, vmin=np.min(Weights), vmax=np.max(Weights))
    plt.savefig("./results/heatmap" + dataset + ".png")
    plt.close()
  

