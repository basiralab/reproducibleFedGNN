from tqdm import tqdm
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO


def get_dl(data_flag='PneumoniaMNIST', download=True,  BATCH_SIZE=1, type="Train"):
    """
    Parameters
    ----------
    data_flag : name of particular dataset of MedMNIST
    download : flag for downloading the dataset if not downloaded yet
    BATCH_SIZE : batch size
    
    Description
    ----------
    This methods returns the training, validation and testing data loaders

    """
    data_flag = data_flag.lower()
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    DataClass = getattr(medmnist, info['python_class'])
    
    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)
    
    pil_dataset = DataClass(split='train', download=download)
    print(train_dataset)
    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
    if type == "Train":
        return train_loader
    elif type == "Val":
        return train_loader_at_eval
    elif type == "Test":
        return test_loader
    return train_loader, train_loader_at_eval, test_loader

def get_adjs(i, j, n):
    """
    Parameters
    ----------
    i : row of pixel
    j : column of pixel
    n : 1-D size of a squared image (nxn)
    
    Description
    ----------
    This method returns the adjacency list of specific pixel

    """
    adj_list = []
    # Upper-Left
    uli, ulj = i - 1, j - 1
    if 0 <= uli < n and 0 <= ulj < n:
        adj_list.append((uli, ulj))
    
    # Up
    ui, uj = i - 1, j
    if 0 <= ui < n and 0 <= uj < n:
        adj_list.append((ui, uj))
    
    # Upper-Right
    uri, urj = i - 1, j + 1
    if 0 <= uri < n and 0 <= urj < n:
        adj_list.append((uri, urj))
    
    # Left
    li, lj = i, j - 1
    if 0 <= li < n and 0 <= lj < n:
        adj_list.append((li, lj))
    
    # Right
    ri, rj = i, j + 1
    if 0 <= ri < n and 0 <= rj < n:
        adj_list.append((ri, rj))

    # Lower-Left
    lli, llj = i + 1, j - 1
    if 0 <= lli < n and 0 <= llj < n:
        adj_list.append((lli, llj))

    # Down
    di, dj = i + 1, j
    if 0 <= di < n and 0 <= dj < n:
        adj_list.append((di, dj))
    
    # Lower-right
    lri, lrj = i + 1, j + 1
    if 0 <= lri < n and 0 <= lrj < n:
        adj_list.append((lri, lrj))
    
    return adj_list
def img_to_adj(image):
    """
    Parameters
    ----------
    image : nxn square image
    
    Description
    ----------
    This method returns the weighted adjacency matrix
    Weigths are determined by the absolute differences of adjacent pixels

    """
    n = image.shape[0]
    adj = np.zeros((n * n,n * n), np.float32)
    for i in range(0,n):
        for j in range(0 ,n):
            adjList = get_adjs(i, j, n)
            for u,v in adjList:
                weight = np.abs(image[i, j] - image[u, v])
                adj[n * i + j, n * u + v] = weight
                adj[n * u + v, n * i + j] = weight
    return adj



def get_G_list(train_loader, args, size=500):
    """
    Parameters
    ----------
    train_loader : training data loader
    args : arguments
    size : maximum size of data samples

    Description
    ----------
    This method returns the adjacency matrices and their labels in a dictionary

    """
    G_list = []
    id = 0
    train_features = []
    lab0, lab1 = 0, 0
    for input, target in tqdm(train_loader):
        if(target[0][0] == 1 and lab1 == size):
            continue
        if(target[0][0] == 0 and lab0 == size):
            continue
        if(target[0][0] == 1):
            lab1 +=1
        if(target[0][0] == 0):
            lab0 += 1
        train_features.extend(input.flatten())
        if args.input_type == 'adj':
            adj = img_to_adj(input[0][0].numpy())
        elif args.input_type == 'image':
            adj = input[0][0]
        G_element = {'adj': adj, 'label': target[0][0], 'id': id}
        id += 1
        G_list.append(G_element)
    print("Sample-0:",lab0, " sample-1:", lab1)
    return G_list