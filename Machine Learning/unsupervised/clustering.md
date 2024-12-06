<h1 align='center'> Machine learning - Clustering 👥</h1>


Dans ce document sera présenté quelques bases de code, notamment avec la libraire `scikit-learn`, pour faire du clustering sur python.


## K-Means
K-means est un algorithme de clustering non supervisé utilisé pour partitionner un ensemble de données en **K** groupes (ou clusters) basés sur des caractéristiques similaires. 

```python
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# Séparation 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choix du nombre de classes pour séparer les données
nb_clusters = ...

# Création d'un objet KMeans avec nb_clusters clusters
kmeans = KMeans(n_clusters=nb_clusters)  

# Ajustement du modèle sur les données d'entraînement
kmeans.fit(X_train)  

# Prédiction des clusters pour chaque point de X_train
y_pred = kmeans.predict(X_train)

```

*<u>Remarque:</u> Ici $\mathcal{X}$ peut être une matrice ou un vecteur*.



<br>
<br>

## PCA
La PCA, ou **Analyse en Composantes Principales** (Principal Component Analysis en anglais), est une technique de réduction de dimensionnalité en identifiant les directions dans lesquelles les données varient le plus (appelées composantes principales). 


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Choix de la taille de la réduction 
nb_dim = ...

# Création de l'objet PCA et réduction à nb_dim 
pca = PCA(n_components=nb_dim)

# Ajustement et transformation des données
X_reduced = pca.fit_transform(X)  

# Affichage des résultats
plt.figure(figsize=(22, 8))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='k')
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Classe')
plt.grid()
plt.show()
```
<br>

Dans le cas où il n'y a pas d'objectifs de dimension:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=['X1', 'X2'])

print(f"Variance expliquée par la première composante : {pca.explained_variance_ratio_[0]:.2f}")
print(f"Variance expliquée par la deuxième composante : {pca.explained_variance_ratio_[1]:.2f}")

plt.figure(figsize=(22, 8))
plt.scatter(df_pca['X1'], df_pca['X2'])
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.title('Projection des paramètres sur les deux premières composantes principales')
plt.show()
```