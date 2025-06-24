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

    """
    Hierarchical Cluster Analysis (HCA) for multivariate experimental data.

    This class performs Hierarchical Cluster Analysis on multivariate datasets, where rows
    represent observations (e.g., experimental runs) and columns represent numeric variables.

    Parameters
    ----------
    y : pandas.DataFrame
        Input data to be analyzed. The first column must contain sample labels 
        (e.g., experiment or treatment identifiers). The remaining columns must be numeric.

    z_score : bool, default=True
        If True, standardizes the data using z-score normalization before clustering.

    impute : str or None, default=None
        Strategy to handle missing values. Options are:
        - 'mean'    : replace missing values with the mean of each column
        - 'median'  : replace missing values with the median of each column
        - 'constant': replace missing values with a constant value defined by `fill_value`
        - None      : no imputation is performed; missing values remain as-is

    fill_value : int or float, default=0
        Constant value used to fill missing data when `impute='constant'`.

    Attributes
    ----------
    labels : list of str
        Sample labels extracted from the first column of `y`.
    data : pandas.DataFrame
        Numeric portion of the input DataFrame, excluding the label column.
    scaled_data : numpy.ndarray
        Standardized (or raw, if `z_score=False`) data used for clustering.
    Z : Any
        Internal linkage structure (reserved for future use or extension).
    linkage_matrix : numpy.ndarray or None
        Linkage matrix produced during hierarchical clustering. Used to generate dendrograms.
    clusters : numpy.ndarray or None
        Cluster labels assigned to each sample after HCA is performed.

    Methods
    -------
    1. summary:
        Performs hierarchical clustering and displays a summary of the resulting groups.
        Typically includes a dendrogram and cluster assignments.
        Usage: mv.HCA(y).summary()
    2. plot:
        Generates and displays the Principal Compenent Analysis
        Saves the generated plots as an image file.
        Usage: mv.HCA(y).plot()

    Notes
    -----
    - This class is designed for exploratory analysis of multivariate experimental data.
    - It is assumed that all columns except the first are numeric.
    - Standardizing data with z-score normalization is recommended when variables are on different scales.

    """
    def __init__(self, y, z_score=True, impute=None, fill_value=0 ):
        self.y = y
        self.labels = self.y.iloc[:, 0].astype(str).tolist()
        self.data = self.y.iloc[:, 1:]

        # Fill values if there are Null data
        if impute is not None:
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
    
        return self

    def summary(self, metric='euclidean', method='ward', threshold=None, criterion='distance', depth=2, monocrit_vector=None):
        """
        Performs Hierarchical Cluster Analysis (HCA) and returns a summary table describing each step of the clustering process.
    
        This method computes the linkage matrix and presents a structured summary of how individual samples and clusters were
        merged, based on the specified distance metric and linkage method.
    
        Parameters
        ----------
        metric : str, default='euclidean'
            The distance metric used to compute pairwise distances between observations.
            Common options include:
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
    
        method : str, default='ward'
            The linkage algorithm to use. Supported methods include:
            - 'ward': minimizes the variance of clusters being merged
            - 'single': nearest point algorithm
            - 'complete': farthest point algorithm
            - 'average': average distance between all observations of the two clusters
    
        threshold : float or None, default=None
            Threshold to apply when forming flat clusters. If None, all steps of the hierarchy are computed.
    
        criterion : str, default='distance'
            Criterion to use in forming flat clusters when threshold is set.
            Options include 'distance', 'maxclust', etc.
    
        depth : int, default=2
            Used for inconsistency method (not applicable with 'distance' criterion). Reserved for future use.
    
        monocrit_vector : array-like or None, default=None
            Optional custom distance or scoring vector used for monocriterion cutting. Reserved for advanced use.
    
        Returns
        -------
        pandas.DataFrame
            A DataFrame summarizing the clustering steps. Columns include:
            - Cluster ID   : ID of the newly formed cluster
            - Cluster 1    : ID of the first cluster merged
            - Cluster 2    : ID of the second cluster merged
            - Distance     : Distance between the two merged clusters
            - N Elements   : Total number of elements in the new cluster
    
        Notes
        -----
        - The cluster IDs in "Cluster 1" and "Cluster 2" refer to either individual sample indices or previously formed cluster
        IDs.
        - The resulting DataFrame is useful for understanding the merge sequence and distances used in building the hierarchical
        tree (dendrogram).
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
        """
        Returns cluster labels for each observation based on Hierarchical Cluster Analysis (HCA).

        This method performs HCA using the same parameters as `summary()` and returns the flat cluster assignments as a 1D array.
        It is useful for downstream tasks,  such as visualizing clusters using PCA or analyzing group-specific patterns.
    
        Parameters
        ----------
        metric, method, threshold, criterion, depth, monocrit_vector :
            Same as in `summary()`. See `summary()` docstring for details.
    
        Returns
        -------
        numpy.ndarray
            A 1D array of integers representing the cluster label assigned to each observation.
    
        Notes
        -----
        - To control the number of clusters, set `criterion='maxclust'` and specify a `threshold`
          equal to the desired number of clusters.
        - If `threshold` is None, no flat clusters will be returned.
        - This method is especially useful for combining HCA results with PCA visualizations or other analyses.
        """
       
        self._run(metric=metric, method=method, threshold=threshold, criterion=criterion, depth=depth,
                  monocrit_vector=monocrit_vector)
        return self.clusters

        
    def plot(self, metric='euclidean', method='ward', threshold=None, criterion='distance',  depth=2, monocrit_vector=None, orientation='top'): 
        """
        Plots the dendrogram and linkage distances for the Hierarchical Cluster Analysis (HCA).
    
        This function computes the hierarchical clustering using the given parameters and visualizes the results as:
        
        1. A dendrogram showing the merge structure and optional cluster cut.
        2. A line plot of linkage distances over successive clustering steps.
    
        Parameters
        ----------
        metric, method, threshold, criterion, depth, monocrit_vector :
            Same as in `summary()`. See that method for details.
    
        orientation : str, default='top'
            Orientation of the dendrogram. Options:
            - 'top', 'bottom' (horizontal layout)
            - 'left', 'right' (vertical layout)
    
        Returns
        -------
        None
            Displays and saves the dendrogram and linkage distance plots.
    
        Notes
        -----
        - If `threshold` is provided, the corresponding cluster cut is indicated
          with a dashed red line.
        - Dendrogram is saved as: 'Dendrogram - Hierarchical Cluster Analysis.png'
        - Linkage distance plot is saved as: 'Linkage Distances - Hierarchical Cluster Analysis.png'
        - Useful for assessing the optimal number of clusters visually (e.g., by identifying "elbows").
            
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
                
