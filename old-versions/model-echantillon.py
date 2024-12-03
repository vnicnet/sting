


import gstlearn as gl
import gstlearn.plot as gp
import gstlearn.document as gdoc
import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd

gdoc.setNoScroll()

#gl.ECov.printAll()


grid = gl.DbGrid.create(nx=[100,100])
model = gl.Model.createFromParam(type=gl.ECov.BESSEL_K, range = 30, param = 2)  # comment rajouter la NUGGET si il ne prends pas de typeS ?
err = gl.simtub(None, grid, model, None, nbsimu=1, seed=13126, nbtuba = 1000)

ax = grid.plot()
#plt.show()

df = grid.toTL()
#df = df.drop(columns=['rank'])

def MaterPointProcess_0_99(lambdaParent,lambdaDaughter,radiusCluster,df,plot=False):
    # hpaulkeeler.com/simulating-a-matern-cluster-point-process/ ## adaptated
    # Simulation window parameters
    xMin = 0;
    xMax = 99;
    yMin = 0;
    yMax = 99;  

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
    echantillon = echantillon.round(0).astype(int)

    if plot:
        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)

        # First scatter plot
        axs[0].scatter(xx, yy, edgecolor='b', facecolor='none', alpha=0.5)
        axs[0].set_xlim([0, 100])
        axs[0].set_ylim([0, 100])
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        axs[0].axis('equal')

        # Heatmap data
        heatmap_data = df.pivot_table(index='x2', columns='x1', values='Simu')

        # Second heatmap plot
        cax = axs[1].imshow(heatmap_data, aspect='auto', origin='lower', cmap='viridis', extent=[0, 100, 0, 100])
        fig.colorbar(cax, ax=axs[1], label='Simu')
        axs[1].scatter(echantillon['x1'], echantillon['x2'], color='red', s=10)
        axs[1].set_xlabel('x1')
        axs[1].set_ylabel('x2')
        axs[1].set_title('Simu Values with Selected Points')

        plt.show()
    return(echantillon)    
        

#example
echantillon_a=MaterPointProcess_0_99(0.003,0.1,5,df,True) #(lambdaParent,lambdaDaughter,radiusCluster,dataframe_grille de la simu,plot=False)


#print(echantillon['x1'].max()) # verification que on est bien en 0:99
