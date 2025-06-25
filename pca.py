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
import seaborn as sns
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
from IPython.display import display, HTML

class PCA:
    """
    Principal Component Analysis (PCA) for Multivariate Data Exploration.

    This class implements a complete PCA workflow, including preprocessing,
    dimensionality reduction, and visualization (scree plot, scatter plot, biplot),
    with optional integration of clustering results.

    Parameters
    ----------
    y : pd.DataFrame
        Input DataFrame where the first column contains sample labels and
        the remaining columns contain numeric variables.

    z_score : bool, default=True
        If True, applies z-score standardization to the numeric data.

    impute : str or None, default=None
        Strategy for imputing missing values:
        - 'mean': replaces missing values with column mean.
        - 'median': replaces with column median.
        - 'constant': replaces with `fill_value`.
        - None: no imputation.

    fill_value : scalar, default=0
        Value used to fill missing data when `impute='constant'`.

    Attributes
    ----------
    scaled_data : np.ndarray
        Preprocessed (and optionally standardized) numeric data.

    pca : sklearn.decomposition.PCA
        Fitted PCA object.

    pca_df : pd.DataFrame
        DataFrame with principal component scores and sample labels.

    Methods
    -------
    scree()
        Displays the scree plot and prints explained variance summary and loadings.

    plot(pcs=None, clusters=None)
        Plots the PCA score scatter plot using selected principal components.
        Optionally colors points by cluster assignments.

    biplot(scale_vectors=1, pcs=None, clusters=None)
        Draws a PCA biplot including samples and loading vectors.
        Optionally colors samples by cluster.

    Notes
    -----
    - This class is suited for exploratory analysis of multivariate experimental data.
    - Loadings are available in the scree and biplot visualizations.
    - Clustering results (e.g., from HCA) can be visually integrated via `clusters`.
    - Input `y` should be a DataFrame where:
        - The first column contains sample labels (e.g., experiment codes)
        - Remaining columns are numeric responses.
    """

    def __init__(self, y, z_score=True,impute=None, fill_value=0):
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
        
    def _run(self):
    
        self.pca = SklearnPCA()
        self.pca_result = self.pca.fit_transform(self.scaled_data)
        
        self.pca_df = pd.DataFrame(
            self.pca_result,
            columns=[f'PC{i+1}' for i in range(self.pca_result.shape[1])]
        )
        self.pca_df['Label'] = self.labels

            
    def scree(self):
        """    
        Generate and display a Scree Plot with explained variance summary and PCA loadings.
    
        This method performs the following steps:
        - Executes PCA 
        - Plots the eigenvalues of each principal component (Scree Plot)
        - Displays a table with:
            - Eigenvalues
            - Explained variance ratio
            - Cumulative explained variance
        - Displays a table with the component loadings (weights of original variables)
        
        The scree plot is saved as a PNG file named "Scree Plot.png".
    
        Notes
        -----
        - The eigenvalues represent the variance explained by each principal component.
        - Loadings indicate how much each original variable contributes to the components.
        - Data is automatically standardized if `z_score=True` was set during initialization.
        - Missing values are imputed according to the `impute` strategy (if provided).
        """
        
        self._run()
        
        explained_variance = self.pca.explained_variance_
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_explained_variance = explained_variance_ratio.cumsum()
        
        # Scree plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o', linestyle='--', color='blue')
        for i, val in enumerate(explained_variance):
            plt.text(i + 1, val, f'PC{i+1}', ha='center', va='bottom')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalues')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('Scree Plot.png', transparent=True)
        plt.show()

        summary = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
            'Eigenvalues': explained_variance,
            'Explained Variance Ratio': explained_variance_ratio,
            'Cumulative Explained Variance': cumulative_explained_variance
        })
        print('\n')
        display(HTML("<div style='text-align: center; font-weight: bold;'>Explained Variance Summary</div>"))
        display(HTML("<div style='display: flex; justify-content: center;'>" + summary.to_html() + "</div>"))
        print('\n')
        
        loadings = pd.DataFrame(
            self.pca.components_.T, 
            columns=[f'PC{i+1}' for i in range(len(self.pca.components_))],
            index=self.data.columns
        )
        display(HTML("<div style='text-align: center; font-weight: bold;'>PCA Loadings</div>"))
        display(HTML("<div style='display: flex; justify-content: center;'>" + loadings.to_html() + "</div>"))
    

    def plot(self, pcs=None, clusters=None):
        """
        Plot a PCA score scatter plot with optional cluster coloring.
    
        This method displays a 2D scatter plot of the selected principal components.
        Optionally, it can color points by cluster labels (e.g., from HCA).
    
        Parameters
        ----------
        pcs : list of int, optional
            List containing two integers indicating which principal components to plot.
            Example: [1, 2] will plot PC1 vs PC2. Defaults to [1, 2].
        
        clusters : array-like, optional
            Cluster labels to color the points. Should match the number of samples.
            Typically provided by an external clustering method (e.g., HCA).
    
        Raises
        ------
        ValueError
            If `pcs` is not a list of exactly two integers within valid component range.
    
        Notes
        -----
        - Points are annotated with their corresponding labels from the input data.
        - Dashed lines are drawn at zero for both axes to highlight origin.
        - If clusters are provided, points are colored using a distinct palette.
        - This method calls `.show()` to immediately display the plot.

        The PCA plot is saved as a PNG file named "PCA Analysis.png".
        """
        self._run()
    
        df_pca = self.pca_df.copy()
        n_components = self.pca_result.shape[1]
    
        # Define the components
        if pcs is None:
            pc_x, pc_y = 1, 2
        else:
            if (not isinstance(pcs, list)) or (len(pcs) != 2):
                raise ValueError("`pcs` must to be a list with two components.")
            if any((not isinstance(pc, int) or pc < 1 or pc > n_components) for pc in pcs):
                raise ValueError(f"Each index in `pcs` must to be a intenger between 1 and {n_components}.")
            pc_x, pc_y = pcs
    
        x_comp = f'PC{pc_x}'
        y_comp = f'PC{pc_y}'
    
        # Add cluster
        if clusters is not None:
            df_pca['Cluster'] = clusters
            palette = sns.color_palette("tab10", len(set(clusters)))
            hue = 'Cluster'
        else:
            hue = None
            palette = None
    
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=df_pca, x=x_comp, y=y_comp, hue=hue, palette=palette, s=100)
    
    
        for i, label in enumerate(df_pca['Label']):
            plt.text(df_pca.loc[i, x_comp] + 0.1, df_pca.loc[i, y_comp], label, fontsize=9)
    
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.title(f'PCA Analysis - {x_comp} vs {y_comp}')
        plt.xlabel(x_comp)
        plt.ylabel(y_comp)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('PCA Analysis.png', transparent=True)
        plt.show()

    def biplot(self, scale_vectors=1, pcs=None, clusters=None):
        """
        Plot a PCA biplot with principal component scores and loading vectors.
    
        Displays a 2D biplot of the selected principal components, showing both the
        projected samples (scores) and the feature contributions (loadings) as arrows.
        Optionally, points can be colored according to provided cluster labels.
    
        Parameters
        ----------
        scale_vectors : float, default=1
            Scaling factor for the length of the loading vectors (arrows). 
            Use to adjust visual clarity depending on data spread.
    
        pcs : list of int, optional
            List of two integers indicating the principal components to plot.
            Example: [1, 2] for PC1 vs PC2. Defaults to [1, 2].
    
        clusters : array-like, optional
            Cluster labels used to color the sample points. Must match the number of samples.
            Typically obtained from external clustering algorithms such as HCA.
    
        Notes
        -----
        - Points are annotated with their labels.
        - Red arrows represent the loading vectors (i.e., contribution of each original variable).
        - Dashed lines at zero help identify the axes.
        - The figure is saved as 'PCA Biplot.png' with transparent background.
        - The plot is displayed immediately with `plt.show()`.
    
        Raises
        ------
        ValueError
            If `pcs` is not a list of two valid component indices.
    
        Example
        -------
        >>> pca = PCA(df)
        >>> pca.biplot(scale_vectors=1.5, pcs=[1, 3], clusters=labels)
        """
        self._run()
    
        n_components = self.pca_result.shape[1]
    
        if pcs is None:
            pc_x, pc_y = 1, 2
        else:
            pc_x, pc_y = pcs
    
        x_comp = f'PC{pc_x}'
        y_comp = f'PC{pc_y}'
    
        df_pca = self.pca_df.copy()


        # Add clusters
        if clusters is not None:
            df_pca['Cluster'] = clusters
            palette = sns.color_palette("tab10", len(set(clusters)))
            hue = 'Cluster'
        else:
            hue = None
            palette = None
 
        
        plt.figure(figsize=(10, 6))   
        # Samples
        sns.scatterplot(data=df_pca, x=x_comp, y=y_comp, hue=hue, palette=palette, s=100)
 
    
        for i, label in enumerate(df_pca['Label']):
            plt.text(df_pca.loc[i, x_comp] + 0.1, df_pca.loc[i, y_comp], label, fontsize=9)
    
        # Vectors (loadings)
        loadings = self.pca.components_.T
        features = self.data.columns
    
        for i, feature in enumerate(features):
            plt.arrow(0, 0,
                      loadings[i, pc_x - 1] * scale_vectors,
                      loadings[i, pc_y - 1] * scale_vectors,
                      color='red', alpha=0.5, head_width=0.05)
            plt.text(loadings[i, pc_x - 1] * scale_vectors * 1.1,
                     loadings[i, pc_y - 1] * scale_vectors * 1.1,
                     feature, color='red', ha='center', va='center', fontsize=9)
    
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.title(f'PCA Biplot - {x_comp} vs {y_comp}')
        plt.xlabel(x_comp)
        plt.ylabel(y_comp)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('PCA Biplot.png', transparent=True)
        plt.show()
