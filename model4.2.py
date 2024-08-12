import torch
import torch.nn as nn
import random
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

import gstlearn as gl
import gstlearn.plot as gp
import gstlearn.document as gdoc
import matplotlib.pyplot as plt

import pandas as pd

import random
import numpy as np
gdoc.setNoScroll()
import torch_geometric
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
#from torch_scatter import scatter_mean

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def MaterPointProcess_0_99(lambdaParent,lambdaDaughter,radiusCluster,plot=False):

  # hpaulkeeler.com/simulating-a-matern-cluster-point-process/ ## adaptated
  # Simulation window parameters
  xMin = 0;
  xMax = 1;
  yMin = 0;
  yMax = 1;

  # Extended simulation windows parameters
  rExt = radiusCluster;  # extension parameter -- use cluster radius
  xMinExt = xMin - rExt;
  xMaxExt = xMax + rExt;
  yMinExt = yMin - rExt;
  yMaxExt = yMax + rExt;

  # rectangle dimensions
  xDeltaExt = xMaxExt - xMinExt;
  yDeltaExt = yMaxExt - yMinExt;
  areaTotalExt = xDeltaExt * yDeltaExt;  # area of extended rectangle

  # Simulate Poisson point process for the parents
  numbPointsParent = np.random.poisson(areaTotalExt * lambdaParent);  # Poisson number of points
  # x and y coordinates of Poisson points for the parent
  xxParent = xMinExt + xDeltaExt * np.random.uniform(0, 1, numbPointsParent);
  yyParent = yMinExt + yDeltaExt * np.random.uniform(0, 1, numbPointsParent);

  # Simulate Poisson point process for the daughters (ie final poiint process)
  numbPointsDaughter = np.random.poisson(lambdaDaughter*np.pi*((radiusCluster)**2), numbPointsParent);
  #l'aire du cercle étant <1 cela nuit gravement au résultat à utiliser en fonction des valeurs des paramètres
  numbPoints = sum(numbPointsDaughter);

  # Generate the (relative) locations in polar coordinates by
  # simulating independent variables.
  theta = 2 * np.pi * np.random.uniform(0, 1, numbPoints);  # angular coordinates
  rho = radiusCluster * np.sqrt(np.random.uniform(0, 1, numbPoints));  # radial coordinates
  # Convert from polar to Cartesian coordinates
  xx0 = rho * np.cos(theta);
  yy0 = rho * np.sin(theta);

  # replicate parent points (ie centres of disks/clusters)
  xx = np.repeat(xxParent, numbPointsDaughter);
  yy = np.repeat(yyParent, numbPointsDaughter);

  # translate points (ie parents points are the centres of cluster disks)
  xx = xx + xx0;
  yy = yy + yy0;

  # thin points if outside the simulation window
  booleInside = ((xx >= xMin) & (xx <= xMax) & (yy >= yMin) & (yy <= yMax));
  # retain points inside simulation window
  xx = xx[booleInside];
  yy = yy[booleInside];

  # keeping the same digits precision as the simtub
  data = {'x1': xx, 'x2': yy}
  echantillon = pd.DataFrame(data)
  #echantillon = echantillon.astype(int)

  if plot:
        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

        # First scatter plot
        axs[0].scatter(xx, yy, edgecolor='b', facecolor='none', alpha=0.5)
        axs[0].set_xlim([0, 1])
        axs[0].set_ylim([0, 1])
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].axis('equal')

        # Heatmap data
        heatmap_data = df.pivot_table(index='x2', columns='x1', values='Simu')

        # Second heatmap plot
        cax = axs[1].imshow(heatmap_data, aspect='auto', origin='lower', cmap='viridis', extent=[0, 1, 0, 1])
        fig.colorbar(cax, ax=axs[1], label='Simu')
        axs[1].scatter(echantillon['x1'], echantillon['x2'], color='red', s=10)
        axs[1].set_xlabel('x1')
        axs[1].set_ylabel('x2')
        axs[1].set_title('Simu Values with Selected Points')

        plt.show()

  return(echantillon)


def créer_edge_graph(echantillon, threshold, max_neighbors, node_features):
    edge_sources = []
    edge_destinations = []
    edge_features = []

    for i in range(len(echantillon)):
        distances = []
        for j in range(len(echantillon)):
            if i != j:
                dist = np.sqrt((echantillon.iloc[i]['x1'] - echantillon.iloc[j]['x1'])**2 +
                               (echantillon.iloc[i]['x2'] - echantillon.iloc[j]['x2'])**2)
                if dist < threshold:
                    distances.append((dist, j))

        if len(distances) > max_neighbors:
            distances = random.sample(distances, max_neighbors)

        for dist, j in distances:
            edge_sources.append(i)
            edge_destinations.append(j)
            edge_features.append(dist)
            edge_sources.append(j)
            edge_destinations.append(i)
            edge_features.append(dist)

    edge_index = torch.tensor([edge_sources, edge_destinations], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float).view(-1, 1)  # Convert to tensor and reshape

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)
    return (data)


def simu_echantillonage_graph(range,param_model,lambdaParent,lambdaDaughter,radiusCluster,threshold,max_neighbors):

    echantillon_a=MaterPointProcess_0_99(lambdaParent,lambdaDaughter,radiusCluster,plot=False)#(100,50,0.10,False)

    db = gl.Db_fromPanda(echantillon_a)
    db.setLocators(["x1","x2"],gl.ELoc.X)

    model = gl.Model.createFromParam(type=gl.ECov.BESSEL_K, range = range, param = param_model)  # comment rajouter la NUGGET si il ne prends pas de typeS ?
    err = gl.simtub(None, db, model, None, nbsimu=1, seed=13126, nbtuba = 1000)
    #err = gl.Model.addCovFromParam(model,type=gl.ECov.NUGGET, sill = 1) ##Ajouter un nugget de variance 1

    df=pd.DataFrame({'Simu': db['z1']})
    echantillon_a = pd.concat([echantillon_a, df],axis=1)
    node_features = torch.tensor(echantillon_a[['Simu']].values, dtype=torch.float)

    data=créer_edge_graph(echantillon_a,threshold,max_neighbors,torch.cat([node_features, torch.zeros((node_features.size(0), 1), dtype=torch.float)], dim=1)) #threshold=0.4 max_neighbors=10
    return(data)


#define the function that made the datatset
#n the number of sets
#constant_product of lambda daughter/parent during Matérn process with radiuscluster fixed at 0.1
def datalistandlabelsmaker(constant_product,n,set_size):
    data_list=[]
    labels=[]
    range_list=np.random.uniform(0.05,0.5,n)
    variance_list=np.random.uniform(0.0,1.0,n)
    lambdaParent = np.random.uniform(50, 150,n)
    lambdaDaughter =[ (constant_product / lambdaParent[i]) for i in range(n)]
    for i in range(n):
        labels.append((range_list[i],variance_list[i]))
        for j in range(set_size):          
            data_list.append(simu_echantillonage_graph(range_list[i],variance_list[i],lambdaParent[i],lambdaDaughter[i],0.10,0.4,10))

    labels=torch.tensor(labels)
    print(f"Generated dataset size: {n*set_size}")
    return data_list,labels

n=3
set_size=3
datalist,labels=datalistandlabelsmaker(7000,n,set_size)

######

class EdgeFeatureMLP2(nn.Module): #(1,128,10)
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EdgeFeatureMLP2, self).__init__()
        self.mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    ).to(device)

    def forward(self, x):
        return self.mlp(x)

def one_hot_encode_distance(threshold, distance, epsilon=1e-5):
    if distance < epsilon or distance > threshold:
        raise ValueError("Distance should be within the range (0, threshold]")
    
    interval_length = threshold / 10
    interval_index = int(distance / interval_length)
    if interval_index == 10:
        interval_index = 9
    
    one_hot = torch.zeros(10, device=device)
    one_hot[interval_index] = 1
    
    return one_hot

class CouchesintermediairesGNN(nn.Module):

    def __init__(self, in_channels, hidden_channels, edge_hidden_dim, edge_output_dim, threshold):
        super(CouchesintermediairesGNN, self).__init__()
        self.a = nn.Parameter(torch.rand(1, device=device)) # a in [0, 1]
        self.b = nn.Parameter(torch.rand(1, device=device) + 1) # b > 0
        self.gamma1 = nn.Parameter(torch.rand(hidden_channels, hidden_channels, device=device))
        self.gamma2 = nn.Parameter(torch.rand(hidden_channels, hidden_channels, device=device))
        self.bias = nn.Parameter(torch.rand(hidden_channels, device=device))
        self.edge_mlp = EdgeFeatureMLP2(1, edge_hidden_dim, edge_output_dim)
        self.threshold = threshold
 
    def rho(self, h_j, h_j_prime):
        return ((torch.abs(self.a * h_j - (1 - self.a) * h_j_prime)) ** self.b).to(device)


    def calculate_w_tilde(self, j, j_prime, edge_attr_combined, neighbors, edge_index):
        key = tuple(sorted((j, j_prime)))
        if key in self.w_tilde_cache:
            return self.w_tilde_cache[key]

        sum_w = torch.zeros(1, 20, device=device)
        for j_prime_prime in neighbors:
            edge_idx = ((edge_index[0] == j) & (edge_index[1] == j_prime_prime))
            sum_w += edge_attr_combined[torch.nonzero(edge_idx, as_tuple=True)].squeeze(0)

        mask = sum_w != 0
        edge_attr_j = edge_attr_combined[torch.nonzero((edge_index[0] == j) & (edge_index[1] == j_prime), as_tuple=True)].view(-1)
        replacement_tensor = torch.full((1, 20), 1e-2, device=device)
        w_tilde = torch.where(mask, edge_attr_j / sum_w, replacement_tensor)

        self.w_tilde_cache[key] = w_tilde

        return w_tilde
 
 # mon forward modifié
    def forward(self, data):
 
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        edge_index = torch_geometric.utils.coalesce(edge_index)
        new_node_features = torch.zeros(x.size(0), 20, device=device)
        
        mlp_output = self.edge_mlp(edge_attr)
        one_hot_features = torch.stack([one_hot_encode_distance(self.threshold, d.item()) for d in edge_attr]).to(device)
        edge_attr_combined = torch.cat((one_hot_features, mlp_output), dim=1)
        self.w_tilde_cache = {}
        data.x = torch.zeros(x.size(0), 2, 20, device=device)

        for j in range(x.size(0)):
            neighbors = torch.unique(edge_index[1][edge_index[0] == j], sorted=False)
            sum_features = torch.zeros([1, 20], device=device)
            for j_prime in neighbors:
                rho_j_j_prime = self.rho(x[j, 0], x[j_prime, 0]).view(1, -1)
                w_tilde_j_j_prime = self.calculate_w_tilde(j, j_prime, edge_attr_combined, neighbors, edge_index)
                sum_features += rho_j_j_prime * w_tilde_j_j_prime
            data.x[j, 1, :] = sum_features

        for a in range(x.size(0)):
            data.x[a, 0, :] = torch.sigmoid(
                self.gamma1 @ x[a, 0].view(-1, 1) + 
                self.gamma2 @ new_node_features[a].view(-1, 1) + 
                self.bias.view(-1, 1)
             ).squeeze()
 
        
        return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr)

class EdgeFeatureMLP1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EdgeFeatureMLP1, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(device) 

    def forward(self, x):
        return self.mlp(x)
 
 
class CoucheinitialeGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_hidden_dim, edge_output_dim, threshold):
        super(CoucheinitialeGNN, self).__init__()
        self.a = nn.Parameter(torch.rand(1, device=device)) # a in [0, 1]
        self.b = nn.Parameter(torch.rand(1, device=device) + 1) # b > 0
        self.gamma1 = nn.Parameter(torch.rand(hidden_channels, 1, device=device))
        self.gamma2 = nn.Parameter(torch.rand(hidden_channels, hidden_channels, device=device))
        self.bias = nn.Parameter(torch.rand(hidden_channels, device=device))
        self.edge_mlp = EdgeFeatureMLP1(1, edge_hidden_dim, edge_output_dim)
        self.threshold = threshold
 

    def rho(self, h_j, h_j_prime):
        return ((torch.abs(self.a * h_j - (1 - self.a) * h_j_prime)) ** self.b)

    def calculate_w_tilde(self, j, j_prime, edge_attr_combined, neighbors, edge_index):
        key = tuple(sorted((j, j_prime)))
        if key in self.w_tilde_cache:
            return self.w_tilde_cache[key]

        sum_w = torch.zeros(1, 20, device=device)
        for j_prime_prime in neighbors:
            edge_idx = ((edge_index[0] == j) & (edge_index[1] == j_prime_prime))
            sum_w += edge_attr_combined[torch.nonzero(edge_idx, as_tuple=True)].squeeze(0)

        mask = sum_w != 0
        edge_attr_j = edge_attr_combined[torch.nonzero((edge_index[0] == j) & (edge_index[1] == j_prime), as_tuple=True)].view(-1)
        replacement_tensor = torch.full((1, 20), 1e-2, device=device)
        w_tilde = torch.where(mask, edge_attr_j / sum_w, replacement_tensor)
        
        self.w_tilde_cache[key] = w_tilde

        return w_tilde

    def forward(self, data):
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        edge_index = torch_geometric.utils.coalesce(edge_index)
        new_node_features = torch.zeros(x.size(0), 20, device=device)
        mlp_output = self.edge_mlp(edge_attr)
        one_hot_features = torch.stack([one_hot_encode_distance(self.threshold, d.item()) for d in edge_attr]).to(device)
        edge_attr_combined = torch.cat((one_hot_features, mlp_output), dim=1)
        self.w_tilde_cache = {} 
        data.x = torch.zeros(x.size(0), 2, 20, device=device)

        for j in range(x.size(0)):
            neighbors = torch.unique(edge_index[1][edge_index[0] == j], sorted=False)
            sum_features = torch.zeros([1, 20], device=device)
            for j_prime in neighbors:
                rho_j_j_prime = self.rho(x[j, 0], x[j_prime, 0]).view(1, -1)
                w_tilde_j_j_prime = self.calculate_w_tilde(j, j_prime, edge_attr_combined, neighbors, edge_index)
                sum_features += rho_j_j_prime * w_tilde_j_j_prime
            data.x[j, 1, :] = sum_features

        for a in range(x.size(0)):
            data.x[a, 0, :] = torch.sigmoid(
                self.gamma1 @ x[a, 0].view(-1, 1) + 
                self.gamma2 @ new_node_features[a].view(-1, 1) + 
                self.bias.view(-1, 1)
            ).squeeze()
        
        return Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr)
    
class ReadoutLayer(nn.Module):
    def __init__(self):
        super(ReadoutLayer, self).__init__()

    def forward(self, data):
        # Perform average pooling on the first node feature for tha entire graph
        device=data.x.device
        return global_mean_pool(data.x[:, 0], torch.zeros(data.x.size(0), dtype=torch.long,device=device)).view(-1)

class expact(nn.Module):
    def forward(self, x):
        return torch.exp(x)

class MappingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MappingLayer, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            expact()
        )

    def forward(self, pooled_feature):
        device=pooled_feature.device
        out = self.mlp(pooled_feature.to(device))
        return out

class DeepSetsModel(nn.Module):
    def __init__(self, node_input_dim, hidden_dim, edge_hidden_dim, edge_output_dim, final_output_dim, mapping_hidden_dim, threshold):
        super(DeepSetsModel, self).__init__()
        self.couche_initiale_gnn = CoucheinitialeGNN(node_input_dim, hidden_dim, edge_hidden_dim, edge_output_dim, threshold)
        self.couche_intermediaire = CouchesintermediairesGNN(hidden_dim, hidden_dim, edge_hidden_dim, edge_output_dim, threshold)
        self.readout = ReadoutLayer()
        self.mapping = MappingLayer(hidden_dim, mapping_hidden_dim, final_output_dim)

    def forward(self, datas):
        out=torch.zeros((len(datas), 20),device=device)
        for i,data in enumerate(datas):
            data=data.to(device)
            data = self.couche_initiale_gnn(data)
            data = self.couche_intermediaire(data)
            out[i] =  global_mean_pool(data.x[:, 0], torch.zeros(data.x.size(0), dtype=torch.long,device=device)).view(-1)
            
        out=global_mean_pool(out,torch.zeros(out.size(0),dtype=torch.long,device=device))
        output = self.mapping(out)
        output=output.view(-1)
        return output
    
class GraphSetDataset(Dataset):
    def __init__(self, data_list, labels, set_size):
        self.data_list = data_list
        self.labels = labels
        self.set_size = set_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        start = idx * self.set_size
        end = start + self.set_size
        return self.data_list[start:end], self.labels[idx]



dataset = GraphSetDataset(datalist, labels, set_size)
dataset.__getitem__(0)

def collate_fn(batch):
    data_list = []
    labels = []
    for sets, label in batch:
        data_list.extend(sets)
        labels.append(label)
    batched_data = Batch.from_data_list(data_list)
    labels = torch.tensor(labels)
    return batched_data, labels


def collate_fn(batch):
    data_list = []
    labels = []

    for sets, label in batch:
        data_list.extend(sets)
        labels.extend([label] * len(sets))

    batched_data = Batch.from_data_list(data_list)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_data, labels

batch_size = 2

loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

model = DeepSetsModel(node_input_dim=1, hidden_dim=20, edge_hidden_dim=128, edge_output_dim=10, final_output_dim=2, mapping_hidden_dim=128, threshold=0.40).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

#total=numbers of sets=n

def train(model, optimizer, loader, n, batch_size, leave=False):

    model.train()
    MAE = torch.nn.L1Loss(reduction="mean")
    sum_loss = 0.0
    total = n // batch_size
    t = tqdm(enumerate(loader), total=total, leave=leave)

    for i, (batched_data, y_batch) in t:
        print('batched_data',batched_data)
        print('y_batch',y_batch)
        optimizer.zero_grad()
        for data in batched_data:
            data.to(device)
        y_batch = y_batch.to(device)
        print('y_batch',y_batch)
        output = model(batched_data)
        print('output',output) 
        batch_loss = MAE(output, y_batch)
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("loss = %.5f" % batch_loss_item)
        t.refresh()  # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()

        leave = bool(i == total - 1)

    return sum_loss / (i + 1)


n_epochs = 1
# n number of sets
for epoch in range(0, n_epochs):
    print('start of epoch :%5f/%6f' %(epoch+1,n_epochs))   
    loss = train(
        model,
        optimizer,
        loader,
        n,
        batch_size,
        leave=bool(epoch == n_epochs - 1),
    )
    print(loss)


