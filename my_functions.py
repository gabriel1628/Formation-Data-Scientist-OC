import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import dendrogram
from scipy import stats




# PCA

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xticks(np.arange(0, pca.n_components_)+1)
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
    

    
# Clustering

def plot_dendrogram(Z, names):
    plt.figure(figsize=(10,25))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()
    
    


# CORRELATION MATRIX

def print_correlations(df, min_corr, max_corr=1):
    """
    Print the correlations between different variables if the value is between min_corr and max_corr.
    
    
    Parameters
    ----------
    df : class:`pandas.DataFrame`
        The DataFrame from which we compute the correlation matrix.
    
    min_corr : float
        Minimum correlation value to print.
    
    max_corr : float, default: 1
        Maximum correlation value to print.
    """
    
    # We fill the upper right diagonal with NaNs so we don't print the correlation of the variable
    # with itself and we dont print 2 times the same pair of variables
    corr = df.corr()
    for i in range(corr.shape[0]):
        for j in np.arange(corr.shape[1])[i:]:
            corr.iloc[i, j] = np.nan

    mask = (corr > min_corr) & (corr <= max_corr)

    for col in corr.columns:
        values = corr[mask][col].dropna().values.round(3) # we round to the thousandth
        if values.size > 0:
            indexes = corr[mask][col].dropna().index
            for i in range(indexes.size):
                print(f'{col} / {indexes[i]}' + '-'*(60-(len(col)+len(indexes[i]))) + f'({values[i]})')
                
                
                
                
# ANOVA

def anova(X_name, Y_name, data, print_values=False):

    k = len(data[X_name].unique())  # number of groups
    N = len(data.values)  # total number of values
    n = data.groupby(X_name).size() # number of values in each group

    # Degrees of freedom
    databetween = k - 1
    datawithin = N - k
    datatotal = N - 1

    y = data[Y_name]
    y_mean = y.mean()
    yi_means = data.groupby(X_name).mean()[Y_name]
    
    # Sum of Squares Between
    SSbetween = (n*(yi_means - y_mean)**2).sum()

    # Sum of Squares Within
    groups = data[X_name].unique()
    group_var = []
    for group in groups:
        y_group = data.loc[data[X_name] == group, [Y_name]]
        group_var.append(((y_group.values - yi_means[group])**2).sum())
    SSwithin = np.array(group_var).sum()

    # Sum of Squares Total
    SStotal = ((y - y_mean)**2).sum()

    # Mean Square Between
    MSbetween = SSbetween/databetween

    # Mean Square Within
    MSwithin = SSwithin/datawithin

    # Calculating the F-value
    F = MSbetween/MSwithin

    # p-value
    p = stats.f.sf(F, databetween, datawithin)

    # eta squared
    eta_sqrd = SSbetween/SStotal

    # omega squared
    om_sqrd = (SSbetween - (databetween * MSwithin))/(SStotal + MSwithin)
    
    if print_values == True:
        print('eta squared :', eta_sqrd.round(3))
        #print('omega squared :', om_sqrd.round(3))
        #print('F-value :', F)
        print('p-value :', p)
        
    
    return eta_sqrd, p
    

