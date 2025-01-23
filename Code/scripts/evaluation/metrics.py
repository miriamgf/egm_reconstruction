from scipy.stats import pearsonr, spearmanr
import numpy as np

def correlation_by_node(array1, array2):
        """
        Calcula la correlación de Spearman entre las columnas de dos arrays.

        Args:
            array1: un array de numpy de dimensión (n,m)
            array2: otro array de numpy de dimensión (n,m)

        Returns:
            Un array de numpy de dimensión (m,) que contiene la correlación de Spearman
            de las columnas de array1 y array2.
        """

        # Verificar si ambos arrays tienen las mismas dimensiones
        assert (
            array1.shape == array2.shape
        ), "Los arrays deben tener las mismas dimensiones."

        # Calcular la correlación de Spearman de las columnas de ambos arrays
        n_cols = array1.shape[1]
        print('Computing correlation in :', n_cols, 'nodes')
        corr = np.zeros(n_cols)
        for i in range(n_cols):
            corr[i], _ = spearmanr(array1[:, i], array2[:, i]) # or pearsonr

        return corr

from scipy.stats import pearsonr, spearmanr

def rmse_by_node(array1, array2):
        """
        Calcula el RMSE entre las columnas de dos arrays.

        Args:
            array1: un array de numpy de dimensión (n,m)
            array2: otro array de numpy de dimensión (n,m)

        Returns:
            Un array de numpy de dimensión (m,) que contiene la RMSE 
            de las columnas de array1 y array2.
        """

        # Verificar si ambos arrays tienen las mismas dimensiones
        assert (
            array1.shape == array2.shape
        ), "Los arrays deben tener las mismas dimensiones."

        n_cols = array1.shape[1]
        print('Computing rmse in', n_cols, 'columns')
        rmse = np.zeros(n_cols)
        for i in range(n_cols):
            rmse[i]=np.sqrt(np.mean((array1[:, i] - array2[:, i]) ** 2))

        return rmse