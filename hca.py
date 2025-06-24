# Copyright (C) 2025 Romulo Pires
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, inconsistent
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class HCA:
    last_instance = None # Rever se é necessário
    """
    Class for multivariate analysis -  Hierarchical Cluster Analysis and PCA

    Parameters:
    -----------
    - y: DataFrame where the first column must have the labels

    Methods:
    --------
    1. summary:
        Generates and displays the Hierarchical Cluster Analysis data
        Usage: `mv.HCA(y).summary()`
    2. plot:
        Generates and displays the Principal Compenent Analysis
        Saves the generated plots as an image file.
        Usage: `mv.HCA(y).plot()`

    
    Notes:
    ------
    - Suitable for multivariate analysis of experimental data.
    - Input `y` should be a DataFrame where:
        - The first column contains sample labels (e.g., experiment codes)
        - Remaining columns are numeric responses.
    - Data is automatically scaled to standardize each variable.
    
    """
    def __init__(self, y, z_score=True, impute='constant', fill_value=0 ):
        self.y = y
        self.labels = self.y.iloc[:, 0].astype(str).tolist()
        self.data = self.y.iloc[:, 1:]

        # Fill data if there are Null values
        if self.data.isnull().values.any():
            if impute not in ['mean', 'median', 'constant']:
                raise ValueError("impute parameter must be 'mean', 'median' or 'constant'")
            self.imputer = SimpleImputer(strategy=impute, fill_value=fill_value)
            self.data = pd.DataFrame(
                self.imputer.fit_transform(self.data),
                columns=self.data.columns,
                index=self.data.index
            )

        # Normalization
        if z_score:
            self.scaled_data = StandardScaler().fit_transform(self.data)
        else:
            self.scaled_data = self.data
            
        self.Z = None 
        self.linkage_matrix = None
        self.clusters = None

    def _run(self, metric = 'euclidean', method='ward', threshold=None, criterion = 'distance',  depth =2, monocrit_vector= None):
        self.metric = metric
        self.method = method
        self.threshold = threshold
        self.criterion = criterion
        self.depth = depth
        self.monocrit_vector = monocrit_vector

        if self.metric not in ['euclidean', 'cityblock', 'chebyshev', 'minkowski', 'canberra', 'braycurtis', 'mahalanobis',
                               'cosine', 'correlation', 'hamming', 'jaccard', 'matching', 'dice', 'kulsinski', 'rogerstanimoto',
                               'russellrao', 'sokalmichener', 'sokalsneath']:
            raise ValueError("Invalid metric")

        if self.method not in ['ward', 'single', 'complete', 'average', 'centroid', 'median','weighted']:
            raise ValueError("Invalid method")

        if self.criterion not in ['maxclust','distance', 'inconsistent', 'monocrit', 'maxclust_monocrit']:
            raise ValueError("Invalid criterion")
            
        self.Z = linkage(self.scaled_data, metric = self.metric, method=self.method)
        
        if self.threshold is None:
            self.clusters = None
        else:
            if self.criterion == 'monocrit':
                self.clusters = fcluster(self.Z, t=self.threshold, criterion=self.criterion, monocrit=self.monocrit_vector)
            elif self.criterion == 'inconsistent':
                self.clusters = fcluster(self.Z, t=self.threshold, criterion=self.criterion, depth=self.depth)
            else:
                self.clusters = fcluster(self.Z, t=self.threshold, criterion=self.criterion)

        # HCA.last_instance = self
        return self
        # return self.Z

    def summary(self, metric='euclidean', method='ward', threshold=None, criterion='distance', depth=2, monocrit_vector=None):
        """
        Performs HCA analysis and returns a summary formatted as a table (DataFrame)
        with the hierarchical clustering steps.
    
        Returns:
        --------
        DataFrame with columns:
        - Cluster ID: index of the newly formed cluster
        - Cluster 1: index of the first merged cluster
        - Cluster 2: index of the second merged cluster
        - Distance: distance between the merged clusters
        - N Elements: number of elements in the new cluster
        """
        self._run(metric=metric, method=method, threshold=threshold,
                  criterion=criterion, depth=depth, monocrit_vector=monocrit_vector)
    
        df = pd.DataFrame(self.Z, columns=["Cluster 1", "Cluster 2", "Distance", "N Elements"])
        df["Cluster 1"] = df["Cluster 1"].astype(int)
        df["Cluster 2"] = df["Cluster 2"].astype(int)
        df["N Elements"] = df["N Elements"].astype(int)
    
        
        df.insert(0, "Cluster ID", range(len(self.scaled_data), len(self.scaled_data) + df.shape[0]))

    
        return df

    
    def get_clusters(self, metric='euclidean', method='ward', threshold=None, criterion='distance', depth=2, monocrit_vector=None):
        
        self._run(metric=metric, method=method, threshold=threshold, criterion=criterion, depth=depth, monocrit_vector=monocrit_vector)
        return self.clusters

        
    def plot(self, metric = 'euclidean', method='ward', threshold=None, criterion = 'distance',  depth =2, monocrit_vector= None, orientation='top'): 
        """
        Plots the dendrogram for Hierarchical Cluster Analysis (HCA).
    
        Parameters
        ----------
        method : str, optional
            Linkage method to use. Possible values:
            
            - 'ward': Minimizes variance within clusters (spherical clusters, low variance).
            - 'single': Minimum linkage, produces elongated clusters (chains).
            - 'complete': Maximum linkage, creates compact, well separated clusters.
            - 'average': Average linkage between clusters, balanced cluster shapes.
            - 'centroid': Distance between centroids, can create varied cluster shapes.
            - 'median': Median linkage, similar to centroid but more sensitive to outliers.
            - 'weighted': Weighted linkage, variation of average linkage.
            
            Default is 'ward'.
    
        metric : str, optional
            Distance metric to use. Possible values include:
            
            - 'euclidean' (default): Euclidean (L2) distance.
            - 'cityblock': Manhattan (L1) distance.
            - 'chebyshev': Chebyshev (L∞) distance.
            - 'minkowski': Minkowski distance (generalized Lp).
            - 'canberra': Canberra distance.
            - 'braycurtis': Bray-Curtis dissimilarity.
            - 'mahalanobis': Mahalanobis distance (requires VI).
            - 'cosine': Cosine distance.
            - 'correlation': Correlation distance.
            - 'hamming': Hamming distance.
            - 'jaccard': Jaccard distance.
            - 'matching': Matching distance.
            - 'dice': Dice distance.
            - 'kulsinski': Kulsinski distance.
            - 'rogerstanimoto': Rogers-Tanimoto distance.
            - 'russellrao': Russell-Rao distance.
            - 'sokalmichener': Sokal-Michener distance.
            - 'sokalsneath': Sokal-Sneath distance.

            Defaul is 'euclidean'
    
        threshold : float or int, optional
            Threshold value used to cut the dendrogram.
    
        criterion : str, optional
            Criterion to use in forming flat clusters. Possible values:
            
            - 'maxclust': Maximum number of clusters.
            - 'distance': Cut height in the dendrogram.
            - 'inconsistent': Statistical inconsistency threshold.
            - 'monocrit': Custom monotonic criterion vector.
            - 'maxclust_monocrit': Max clusters using monocrit vector.
            
            Default is 'distance'.
    
        depth : int, optional, default=2
            Depth parameter used when criterion is 'inconsistent'.
    
        monocrit_vector : array-like, optional
            Custom monotonic criterion vector required for 'monocrit' and 'maxclust_monocrit' criteria.

        orientation : str, {'top', 'bottom', 'left', 'right'}, optional
            Orientation of the dendrogram. Default is 'top'.
        
        Returns
        -------
        None
            Displays and saves the dendrogram plot.
            
        """
        self.orientation = orientation
        self._run(metric = metric, method=method, threshold=threshold, criterion=criterion, depth=depth, monocrit_vector=monocrit_vector)

        # Dendogram
        plt.figure(figsize=(10, 6))
        
        color_threshold = None
        cut_height = None
        
        n_samples = self.Z.shape[0] + 1  
        
        if self.threshold is not None:
            if criterion == 'distance':
                color_threshold = self.threshold
                cut_height = self.threshold
            elif criterion == 'maxclust':
                fusion_index = n_samples - int(self.threshold) - 1  
                cut_height = self.Z[fusion_index, 2] + 1e-5  
                color_threshold = cut_height
        
        dendrogram(self.Z, color_threshold=color_threshold, labels=self.labels, orientation=self.orientation)
        
        if cut_height is not None:
            if self.orientation in ['top', 'bottom']:
                plt.axhline(y=cut_height, color='r', linestyle='--')
            else:
                plt.axvline(x=cut_height, color='r', linestyle='--')

        plt.title('Dendrogram')
        if self.orientation in ['top', 'bottom']:
            plt.xlabel('Samples')
            plt.ylabel('Distance')
        else:
            plt.xlabel('Distance')
            plt.ylabel('Samples')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('Dendrogram - Hierarchical Cluster Analysis.png', transparent=True)
        plt.show()

        print('\n')

        # Linkage Distances
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, self.Z.shape[0] + 1), self.Z[:, 2], linestyle='-', color='blue')
        plt.title('Plot of Linkage Distances across Steps')
        plt.xlabel('Step')
        plt.ylabel('Linkage Distance')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('Linkage Distances - Hierarchical Cluster Analysis.png', transparent=True)
        plt.show()
                
