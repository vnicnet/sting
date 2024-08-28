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
import time

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

####  DATA

def MaterPointProcess_0_99(lambdaParent,lambdaDaughter,radiusCluster,plot=False):

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


def créer_edge_idx_atr(echantillon, threshold, max_neighbors):
    edge_sources = []
    edge_destinations = []
    edge_features = []

    for i in range(len(echantillon)):
        distances = []
        for j in range(len(echantillon)):
            if i != j:
                dist = np.sqrt((echantillon.iloc[i]['x1'] - echantillon.iloc[j]['x1'])**2 +(echantillon.iloc[i]['x2'] - echantillon.iloc[j]['x2'])**2)
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

    return (edge_index,edge_attr)


def créer_edge_graph(echantillon, threshold, max_neighbors, node_features):
    edge_sources = []
    edge_destinations = []
    edge_features = []

    for i in range(len(echantillon)):
        distances = []
        for j in range(len(echantillon)):
            if i != j:
                dist = np.sqrt((echantillon.iloc[i]['x1'] - echantillon.iloc[j]['x1'])**2 + (echantillon.iloc[i]['x2'] - echantillon.iloc[j]['x2'])**2)
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


def merge_components(components,set_size):
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    num_nodes_offset = 0
    i=0
    T=torch.empty(set_size)
    for component in components:
        print(num_nodes_offset)
        print('component.x',component.x.size())
        x_list.append(component.x)
        print(component.edge_index + num_nodes_offset)
        print('component.edge_index.size()',component.edge_index.size() )

        edge_index_list.append(component.edge_index + num_nodes_offset)  # Ajuster les indices des arêtes
        
        edge_attr_list.append(component.edge_attr)
        num_nodes_offset += component.num_nodes
        T[i]=component.num_nodes
        i+=1
    x = torch.cat(x_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    graph.T=T
    print('graph fait')
    print(graph.edge_index.size())
    return graph


def simu_graph(set_size, range_val, param_model, lambdaParent, lambdaDaughter, radiusCluster, threshold, max_neighbors):
    data = [simu_echantillonage_graph(range_val, param_model, lambdaParent, lambdaDaughter, radiusCluster, threshold, max_neighbors) for i in range(int(set_size))]
    print('data avant merge componenets ',data)
    data=merge_components(data,set_size)
    data.y=torch.tensor((range_val,param_model))
    return data


def datamaker(constant_product,n,set_size):
    datalist=[]
    range_list=np.random.uniform(0.05,0.5,n)
    variance_list=np.random.uniform(0.0,1.0,n)
    lambdaParent = np.random.uniform(50, 150,n)
    lambdaDaughter =[ (constant_product / lambdaParent[i]) for i in range(n)]
    for i in range(n):          
        datalist.append(simu_graph(set_size,range_list[i],variance_list[i],lambdaParent[i],lambdaDaughter[i],0.10,0.4,10))

    print(f"Generated dataset size: {n*set_size}")
    return datalist

constant_product=5000
n=2
set_size=3
datalist=datamaker(constant_product,n,set_size)

print(datalist)

#problème dans la conception des objets datas ou on a plusieurs fois les mêmes edge index



print(datalist[0].edge_index[0][12])
print(datalist[0].edge_index[1][12])
print(datalist[0].edge_index[0][403])
print(datalist[0].edge_index[1][403])


















#####  MODEL
#torch.autograd.set_detect_anomaly(True)

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
        raise ValueError(distance,threshold,"Distance should be within the range (0, threshold]")
    
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
            print(edge_idx.size())
            print(edge_idx)
            sum_w += edge_attr_combined[torch.nonzero(edge_idx, as_tuple=True)].squeeze(0)

        mask = sum_w != 0
        edge_attr_j = edge_attr_combined[torch.nonzero((edge_index[0] == j) & (edge_index[1] == j_prime), as_tuple=True)].view(-1)
        replacement_tensor = torch.full((1, 20), 1e-2, device=device)
        w_tilde = torch.where(mask, edge_attr_j / sum_w, replacement_tensor)

        self.w_tilde_cache[key] = w_tilde

        return w_tilde

    def calculate_w_tildenew(self, j, j_prime, edge_attr_combined, neighbors, edge_index):
        key = tuple(sorted((j, j_prime)))
        if key in self.w_tilde_cache:
            return self.w_tilde_cache[key]


        neighbors=neighbors.view(-1)
        # Extract relevant edge attributes directly using vectorized operations
        print(edge_index[1].unsqueeze(-1).size())
        print(neighbors.size())
        #mask = (edge_index[0] == j) & (edge_index[1].unsqueeze(-1) == neighbors)
        mask = (edge_index[0] == j) & (edge_index[1].unsqueeze(-1) == neighbors).any(dim=1)
        print(edge_attr_combined.size())
        edge_attr_j = edge_attr_combined[mask].view(-1, edge_attr_combined.size(-1))

        # Calculate sum of weights using vectorized operations
        sum_w = edge_attr_j.sum(dim=0, keepdim=True)

        # Handle division and replacement efficiently
        w_tilde = edge_attr_combined[(edge_index[0] == j) & (edge_index[1] == j_prime)].view(1, -1) / sum_w
        w_tilde = torch.where(sum_w != 0, w_tilde, torch.full_like(w_tilde, 1e-2))

        # Cache the result
        self.w_tilde_cache[key] = w_tilde

        return w_tilde


    # mon forward modifié
    def forward(self, data):

        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        print('inter-edge_index',edge_index.size())
        print('inter-edge_attr',edge_attr.size())
        #print('data.x[3,0,:]',data.x[3,0,:])
        edge_index = torch_geometric.utils.coalesce(edge_index)
        #print('x.size()',x.size()) x.size() torch.Size([862, 2, 20])
        mlp_output = self.edge_mlp(edge_attr)
        print('inter-mlp_output',mlp_output.size())
        #print(mlp_output)
        #print(mlp_output.size())
        #print('fin edge_mlp(edge_attr)',time.time() - time1)
        one_hot_features = torch.stack([one_hot_encode_distance(self.threshold, d.item()) for d in edge_attr]).to(device)
        print('inter-one_hot_features',one_hot_features.size())
        edge_attr_combined = torch.cat((one_hot_features, mlp_output), dim=1)
        print('inter-eac',edge_attr_combined.size())
        #print('fin edge_attr_combined',time.time() - time1)
        self.w_tilde_cache = {}
        dataa = torch.zeros(x.size(0), 2, 20, device=device)
        #print('début boucle sur les noeuds',time.time() - time1)
        for j in range(x.size(0)):
            neighbors = torch.unique(edge_index[1][edge_index[0] == j], sorted=False)
            #sum_features = torch.zeros([1, 20], device=device)
            sum_featuresnew = torch.zeros([1, 20], device=device)

            for j_prime in neighbors:
                rho_j_j_prime = self.rho(x[j, 0], x[j_prime, 0]).view(1, -1)
                #w_tilde_j_j_prime = self.calculate_w_tilde(j, j_prime, edge_attr_combined, neighbors, edge_index)
                w_tilde_j_j_primenew = self.calculate_w_tildenew(j, j_prime, edge_attr_combined, neighbors, edge_index)

                #sum_features += rho_j_j_prime * w_tilde_j_j_prime
                sum_featuresnew += rho_j_j_prime * w_tilde_j_j_prime
            dataa[j, 1, :] = sum_featuresnew

            print(sum_features ==  sum_featuresnew)

            

            dataa[j, 0, :] = torch.sigmoid(
                self.gamma1 @ x[j, 0,:].view(-1,1) + 
                self.gamma2 @ sum_featuresnew.view(-1,1) + 
                self.bias.view(-1,1)
            ).squeeze()
        #print('sum_features',sum_features)
        #print('self.gamma1',self.gamma1)
        #print('self.gamma2',self.gamma2)
        #print('dataa',dataa[3,0,:])
        
        return Data(x=dataa, edge_index=edge_index, edge_attr=edge_attr)

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
            print('edge_idx.size()',edge_idx.size())
            print('edge_attr_combined',edge_attr_combined.size())
            print('RZFZI',torch.nonzero(edge_idx, as_tuple=True).unsqueeze(-1))
            sum_w += edge_attr_combined[torch.nonzero(edge_idx, as_tuple=True)].squeeze(0)

        mask = sum_w != 0
        edge_attr_j = edge_attr_combined[torch.nonzero((edge_index[0] == j) & (edge_index[1] == j_prime), as_tuple=True)].view(-1)
        replacement_tensor = torch.full((1, 20), 1e-2, device=device)
        w_tilde = torch.where(mask, edge_attr_j / sum_w, replacement_tensor)
        
        self.w_tilde_cache[key] = w_tilde

        return w_tilde

    def calculate_w_tildenew(self, j, j_prime, edge_attr_combined, neighbors, edge_index):
        key = tuple(sorted((j, j_prime)))
        if key in self.w_tilde_cache:
            return self.w_tilde_cache[key]


        neighbors=neighbors.view(-1)
        # Extract relevant edge attributes directly using vectorized operations
        print(edge_index[1].unsqueeze(-1).size())
        print(neighbors.size())
        #mask = (edge_index[0] == j) & (edge_index[1].unsqueeze(-1) == neighbors)
        mask = (edge_index[0] == j) & (edge_index[1].unsqueeze(-1) == neighbors).any(dim=1)
        #print(edge_attr_combined.size())  torch.Size([19454, 20])
        edge_attr_j = edge_attr_combined[mask].view(-1, edge_attr_combined.size(-1))
        print('edge_attr_j.size()',edge_attr_j.size())
        # Calculate sum of weights using vectorized operations
        sum_w = edge_attr_j.sum(dim=0, keepdim=True)
        print('sum_w.size()',sum_w.size())
        print(j,j_prime)

        print(edge_index)
        print(edge_index.size())


        # Handle division and replacement efficiently
        print(j,j_prime)
        print(edge_attr_combined[(edge_index[0] == j) & (edge_index[1] == j_prime)].view(1, -1))
        a=torch.nonzero((edge_index[0] == j) & (edge_index[1] == j_prime))
        print(a)
        print(edge_index[0][j])
        print(edge_index[1][j_prime])
        print(edge_index[0][j])
        print(edge_index[1][j_prime])


        w_tilde = edge_attr_combined[(edge_index[0] == j) & (edge_index[1] == j_prime)].view(1, -1) / sum_w
        w_tilde = torch.where(sum_w != 0, w_tilde, torch.full_like(w_tilde, 1e-2))

        # Cache the result
        self.w_tilde_cache[key] = w_tilde

        return w_tilde

    def forward(self, data):
        x, edge_index, edge_attr = data.x.to(device), data.edge_index.to(device), data.edge_attr.to(device)
        
        print('init-edge_index',edge_index.size())
        print('init-edge_attr',edge_attr.size())
        #edge_index,edge_attr = torch_geometric.utils.coalesce(edge_index,edge_attr,reduce='mean', is_sorted=True)
        #print('init-edge_index',edge_index.size())
        #print('init-edge_attr',edge_attr.size())
        
        #print('x.size()',x.size()) x.size() torch.Size([862, 2])

        mlp_output = self.edge_mlp(edge_attr)
        one_hot_features = torch.stack([one_hot_encode_distance(self.threshold, d.item()) for d in edge_attr]).to(device)
        edge_attr_combined = torch.cat((one_hot_features, mlp_output), dim=1)
        self.w_tilde_cache = {} 
        dataa = torch.zeros(x.size(0), 2, 20, device=device)

        for j in range(x.size(0)):
            neighbors = torch.unique(edge_index[1][edge_index[0] == j], sorted=False)
            sum_features = torch.zeros([1, 20], device=device)
            sum_featuresnew = torch.zeros([1, 20], device=device)

            for j_prime in neighbors:
                rho_j_j_prime = self.rho(x[j, 0], x[j_prime, 0]).view(1, -1)
                w_tilde_j_j_primenew = self.calculate_w_tildenew(j, j_prime, edge_attr_combined, neighbors, edge_index)

                #w_tilde_j_j_prime = self.calculate_w_tilde(j, j_prime, edge_attr_combined, neighbors, edge_index)
                sum_features += rho_j_j_prime * w_tilde_j_j_primenew
            dataa[j, 1, :] = sum_features

            dataa[j, 0, :] = torch.sigmoid(
                self.gamma1 @ x[j, 0].view(-1,1) + 
                self.gamma2 @ sum_features.view(-1,1) + 
                self.bias.view(-1,1)
            ).squeeze()

        
        
        return Data(x=dataa, edge_index=edge_index, edge_attr=edge_attr)
    
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
        self.mapping = MappingLayer(hidden_dim, mapping_hidden_dim, final_output_dim)

    def forward(self, data):
        time1 = time.time()
        data = data.to(device)
        data_T=data.T.to(device)
        data_batch=data.T2.to(device)
        data = self.couche_initiale_gnn(data)
        print("first layer done")
        time2 = time.time() - time1
        print(time2)
        data = self.couche_intermediaire(data)
        print("convolutional layer done")
        time3 = time.time() - time1
        print(time3)
        #print('data',data)
        #print('data_T',data_T)
        pooled_graphs = global_mean_pool(data.x[:,0,:], data_T)
        #print('pooled_graphs',pooled_graphs)
        #print('data_batch',data_batch)
        print("pooled layer among graph done")
        time4 = time.time() - time1
        print(time4)
        out=global_mean_pool(pooled_graphs, data_batch)  
        #print('outavantmapping',out)    
        print("pooled layer of DeepSets done")
        time5 = time.time() - time1
        print(time5)
        output = self.mapping(out)
        print("MLP done")
        time6 = time.time() - time1
        print(time6)
        output = output.view(-1)
        return output
    

class GraphSetDataset(Dataset):
    def __init__(self, datalist):
        self.datalist = datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.datalist[idx]









###

dataset = GraphSetDataset(datalist)

batch_size = 2

loader = DataLoader(dataset, batch_size, shuffle=True)

model = DeepSetsModel(node_input_dim=1, hidden_dim=20, edge_hidden_dim=128, edge_output_dim=10, final_output_dim=2, mapping_hidden_dim=128, threshold=0.40).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)


###

























t = tqdm(enumerate(loader))
timeinin = time.time()
timein = timeinin
for i, batched_data in t:

    optimizer.zero_grad()
    T1 = []
    for j in range(len(batched_data.T)):
        T1.extend([j] * int(batched_data.T[j].item()))  
    batched_data.T = torch.tensor(T1, device=device)
    T2=[]
    for w in range(len(batched_data)):
        T2.extend([w] * set_size)
    #print('T2',T2)
    batched_data.T2 = torch.tensor(T2, device=device)
    batched_data.y = batched_data.y.view(batch_size, 2) 
    batched_data = batched_data.to(device)
    print(batched_data)
    
    
    output = model(batched_data)
    output=torch.reshape(output,(batch_size,2))
    print(output)
    timeinter = time.time() - timein
    timein = timeinter
    print(timeinter)

timeout = time.time() - timein
print(timeout)































def train(model, optimizer, loader, n, batch_size, leave=False):
    model.train()
    MAE = torch.nn.L1Loss(reduction="mean")
    sum_loss = 0.0
    total = n // batch_size
    t = tqdm(enumerate(loader), total=total, leave=leave)

    for i, batched_data in t:
        optimizer.zero_grad()

        T1 = []
        for j in range(len(batched_data.T)):
            T1.extend([j] * int(batched_data.T[j].item()))  
        batched_data.T = torch.tensor(T1, device=device)

        T2=[]
        for w in range(len(batched_data)):
            T2.extend([w] * set_size)
        #print('T2',T2)
        batched_data.T2 = torch.tensor(T2, device=device)

        batched_data.y = batched_data.y.view(batch_size, 2) 

        batched_data = batched_data.to(device)

        output = model(batched_data)
        output=torch.reshape(output,(batch_size,2))
        #print('outputdetrain',output)

        if output.size(0) != batch_size or output.size(1) != 2:
            raise ValueError(f"Expected output shape ({batch_size}, 2) but got {output.size()}")

        y_batch = batched_data.y.to(device)
        #print('y_batch de train avant MAE',y_batch)

        batch_loss = MAE(output, y_batch)
        batch_loss.backward()
        optimizer.step()

        sum_loss += batch_loss.item()
        t.set_description(f"loss = {batch_loss.item():.5f}")
        t.refresh()

    return sum_loss / (i + 1)





# Epoch loop
n_epochs = 3
n = len(dataset)
for epoch in range(n_epochs):
    print(f'start of epoch: {epoch+1}/{n_epochs}')
    loss = train(
        model,
        optimizer,
        loader,
        n,
        batch_size,
        leave=(epoch == n_epochs - 1),
    )
    print(f'loss: {loss}')


torch.save(model.state_dict(),'modelparameters_10x5_10ep_batchsize5')

