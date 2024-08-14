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



def créer_edge_idx_atr(echantillon, threshold, max_neighbors):
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

    return (edge_index,edge_attr)



def simu_echantillonage_graph(set_size,range,param_model,lambdaParent,lambdaDaughter,radiusCluster,threshold,max_neighbors):
    
    Node_Features=torch.empty(set_size)
    Edge_Index=torch.empty()
    Edge_Attr=torch.empty()
    N=torch.empty(set_size)
    for i in range(set_size):

        echantillon_a=MaterPointProcess_0_99(lambdaParent,lambdaDaughter,radiusCluster,plot=False)#(100,50,0.10,False)
        N.add_(len(echantillon_a))
        db = gl.Db_fromPanda(echantillon_a)
        db.setLocators(["x1","x2"],gl.ELoc.X)
        model = gl.Model.createFromParam(type=gl.ECov.BESSEL_K, range = range, param = param_model)  # comment rajouter la NUGGET si il ne prends pas de typeS ?
        err = gl.simtub(None, db, model, None, nbsimu=1, seed=13126, nbtuba = 1000)
        #err = gl.Model.addCovFromParam(model,type=gl.ECov.NUGGET, sill = 1) ##Ajouter un nugget de variance 1
        df=pd.DataFrame({'Simu': db['z1']})
        echantillon_a = pd.concat([echantillon_a, df],axis=1)
        #node_features = torch.tensor(echantillon_a[['Simu']].values, dtype=torch.float)
        #node_features=torch.cat([node_features, torch.zeros((node_features.size(0), 1), dtype=torch.float)], dim=1)

        edge_index,edge_attr = créer_edge_idx_atr(echantillon_a, threshold, max_neighbors)
        Node_Features.add_(torch.cat([torch.tensor(echantillon_a[['Simu']].values, dtype=torch.float),torch.zeros((node_features.size(0), 1), dtype=torch.float) ], dim=1))
        Edge_Index.add_(edge_index)
        Edge_Attr.add_(edge_attr)
    
    data = Data(x=Node_Features, edge_index=Edge_Index, edge_attr=Edge_Attr)
    return (data)

data1=simu_echantillonage_graph(5,0.3,2,100,70,0.10,0.4,10)
print(data1)