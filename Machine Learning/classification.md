<h1 align='center'> Machine learning - Classification </h1>

Dans ce document sera présenté quelques bases de code, notamment avec la libraire `scikit-learn`, pour faire de la classification sur python.


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



## SVM
SVM est un algorithme de classification supervisée pour séparer deux classes en maximisant la marge entre les deux points les plus proches de l'hyperplan séparateur.


```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création du modèle SVM avec un noyau linéaire.
svm_model = SVC(kernel='linear')  # noyaux: linéaire, polynomial, gaussien (RBF), sigmoïd...

# Entraînement du modèle
svm_model.fit(X_train, y_train)

# Prédiction
y_pred = svm_model.predict(X_test)

# Affichage des résultats
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='o', edgecolor='k')
plt.title("SVM Classification")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
```

<br>
<br>
<br>

## Estimation de densité
L'estimation de densité permet de quantifier et de modéliser comment les données sont réparties dans un espace donné, facilitant ainsi la détection des anomalies et la classification.


### OneClassSVM
Le One-Class SVM est conçu pour apprendre à partir d'un seul type de données (généralement les données "normales") et à identifier les points qui s'écartent de ce modèle comme des anomalies ou des points d'exception.

**Estimation de la Densité** :
   - L'algorithme crée un hyperplan qui délimite la zone densément peuplée des points de données normaux. En d'autres termes, il établit une frontière dans l'espace des caractéristiques qui capture la région où se trouvent la majorité des points normaux.
   - Les points qui se trouvent en dehors de cette frontière sont considérés comme ayant une faible probabilité d'appartenir à la distribution des données normales, ce qui les qualifie d'anomalies.


```python
import numpy as np
from sklearn.svm import OneClassSVM

# Création de données pour la détection d'anomalies
X_train = np.random.normal(loc=0, scale=1, size=(100, 2))  # Données normales
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))  # Anomalies
X_combined = np.vstack((X_train, X_outliers))  # Combinaison des données

# Création du modèle One-Class SVM avec un noyau RBF.
oc_svm = OneClassSVM(kernel='rbf', gamma='auto')  # noyaux: linéaire, polynomial, gaussien (RBF), sigmoïd...

# Entraînement du modèle
oc_svm.fit(X_train)

# Prédiction
y_pred = oc_svm.predict(X_combined)

# Affichage des résultats
plt.scatter(X_combined[:, 0], X_combined[:, 1], c=y_pred, cmap='coolwarm', marker='o', edgecolor='k')
plt.title("One-Class SVM for Anomaly Detection")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
plt.show()
```