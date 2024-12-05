<h1 align='center'> Machine learning - Détection d'anomalies 🔎</h1>

### Qu'est-ce qu'une anomalie?
* Généralement: un individu (ligne) rare dans un ensemble de données qui diffère significativement de la majorité des données.
* Parfois: les anomalies ne sont pas si rares, et peuvent ne pas être si différentes de la majorité des données...


Pour faire de la détection d'anomalies, l'apprentissage supervisé avec un ensemble de données étiquetées ne peut pas être utilisé car:
- L'ensemble de données serait très déséquilibré: Par définition les anomalies sont **rares**, si alors il y a 5 anomalies pour 100 000 points normaux, il est difficile de faire un bon modèle.
- Il est impossible de couvrir tous les types d'anomalies étant donné qu'une anomalie est quelque chose d'inattendu. Alors, si un nouveau type d'anomalie apparait, il serait classé comme normal, alors que non!

<br>

Il faut alors utiliser d'autres approches:
- Détection des valeurs aberrantes: l'ensemble de données contient des anomalies au sens de rareté et statistiquement différent. Détecter les éléments de ce même ensemble de données qui diffèrent de la majorité des données
- Détection de nouveauté: Soit un ensemble de données propre sans anomalies, il faut apprendre le comportement normal, afin de pouvoir vérifier si un nouvel élément est normal ou une anomalie





## Détection des valeurs aberrantes
Détecter dans un ensemble de données les éléments qui diffèrent de la majorité des données.

En 1D:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotAnomalies1D(s, anomalies, threshold1, threshold2):
   """
      s: Pandas Series containing all the points to plot
      anomalies: Pandas Series containing all the points which are anomalies
      threshold1: Float value - minimum threshold to be normal
      threshold2: Float value - maximum threshold to be normal
   """
   fig = plt.figure(figsize=(22,8))
   plt.plot(s, [0]*len(s), 'bo')
   plt.plot(anomalies, [0]*len(anomalies), 'ro')
   plt.plot([threshold1]*2, [-1,1], 'g--')
   plt.plot([threshold2]*2, [-1,1], 'g--')
   plt.show()

   return 0



med = df["col_name"].mean()
mad = (df["col_name"] - med).abs().mean()

threshold1 = med - 3*mad
threshold2 = med + 3*mad
anomalies = df["col_name"][(df["col_name"] - med).abs() > 3*mad].copy()

plotAnomalies1D(df["col_name"], anomalies, threshold1, threshold2)
```
La moyenne et l'écart-type ne sont pas toujours fiables. En effet, elles quantifient la densité des données dans le cas d'une distribution normale, mais ce n'est pas toujours le cas. Ces valeurs sont donc très sensibles aux valeurs aberrantes: S'il y a trop ou beaucoup de valeurs aberrantes, alors l'estimation sera faussée.   
Comme alternative la **MAD** ($\text{Median Absolute Deviation} = \text{median}( | x - \text{median}(x) | )$) est utilisée.


<br>

En nD:
Il est possible de **réduire la dimension** du problème en un problème 2D, avec une ACP par exemple (cf [[pre-processing](../pre-processing.md)]). Ensuite, une fois que le problème est en 2D, il sera possible de visualiser la séparation via ces fonctions:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotAnomalies2D(df, clf_name, clf):
   """
      df: Pandas DataFrame containing all the points to plot (for features X1 and X2)
      clf_name: String value - name of the outlier detection model
      clf: Scikit Learn model instance - the trained outlier detection model
   """
   fig = plt.figure(figsize=(22,8))
   plt.plot(df['X1'],df['X2'], 'o')
   plt.xlabel('X1')
   plt.ylabel('X2')
   plt.xlim([df['X1'].min()-3,df['X1'].max()+3])
   plt.ylim([df['X2'].min()-3,df['X2'].max()+3])
   plt.title(clf_name)
   
   if clf_name == 'LOF':
      ypred = clf.fit_predict(df[['X1','X2']])
      plt.plot(df['X1'][ypred==-1],df['X2'][ypred==-1],'ro')
   else:
      xx = np.meshgrid(np.linspace(df['X1'].min()-3,df['X1'].max()+3, 500)
      yy = np.linspace(df['X2'].min()-3,df['X2'].max()+3, 500))
      Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
      Z = Z.reshape(xx.shape)
      plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='r')

   plt.show()

   return 0



def plotAnomalyScore2D(df, clf_name, clf):
   """
      df: Pandas DataFrame containing all the points to plot (for features X1 and X2)
      clf_name: String value - name of the outlier detection model
      clf: Scikit Learn model instance - the trained outlier detection model
   """
   if clf_name == 'LOF':
      score = clf.negative_outlier_factor_
   else:
      score = clf.decision_function(df[['X1','X2']])
   
   fig = plt.figure(figsize=(22,8))
   sc = plt.scatter(x=df['X1'],y=df['X2'], c=-score, cmap='Reds')
   plt.colorbar(sc, label='anomaly score')
   plt.xlabel('X1')
   plt.ylabel('X2')
   plt.title(clf_name)
   plt.show()

   return 0
```

<br>

Il existe plusieurs méthodes pour le traiter:

### 1. Elliptic envelope 
L’enveloppe elliptique est une méthode statistique classique pour la détection des anomalies dans des ensembles de données multivariés. Elle repose sur l’hypothèse que les données suivent une distribution normale multivariée. Les points situés loin de l’enveloppe définie par cette distribution sont considérés comme des anomalies:

L'enveloppe elliptique correspond à une région définie par l'équation de la distance de Mahalanobis:
$D^2(x) = (x - \mu)^T \Sigma^{-1} (x - \mu)$ où $D^2(x)$ est la distance de Mahalanobis d’un point $x$ par rapport à la distribution, $\mu$ la moyenne (qui est le centre de l’enveloppe) et $\Sigma$ la matrice de covariance (qui définit la forme et l’étendue de l’enveloppe).   
Les points dont $D^2(x)$ dépasse un certain seuil sont considérés comme des anomalies:

#### a. **Estimation de la moyenne et de la covariance:**
   - Les paramètres $\mu$ et $\Sigma$ sont estimés à partir des données en supposant une distribution normale multivariée.
   - Pour rendre la méthode plus robuste aux anomalies, des techniques comme le **Minimum Covariance Determinant (MCD)** peuvent être utilisées pour initialiser ces paramètres.

#### b. **Calcul des distances de Mahalanobis:**
   - Une fois $\mu$ et $\Sigma$ estimés, la distance de Mahalanobis est calculée pour chaque point.

#### c. **Fixation d’un seuil:**
   - Le seuil est souvent choisi en fonction du degré de confiance $95\%$, $99\%$ dans une distribution $\chi^2$ à $d$ degrés de liberté $d$ est le nombre de dimensions des données.

#### d. **Identification des anomalies:**
   - Les points situés en dehors de l’enveloppe sont marqués comme anomalies.

```python
from sklearn.covariance import EllipticEnvelope
from sklearn.model_selection import GridSearchCV

clf_name = 'Elliptic Envelope'
clf = EllipticEnvelope()

param_grid = {
    'support_fraction': [0.8, 0.9, 1.0],
    'contamination': [0.05, 0.1, 0.2],   
    'assume_centered': [True, False]  
}

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring=custom_scorer,  
    cv=3,
    verbose=True,
    n_jobs=-1
)

grid_search.fit(df_2d[['X1', 'X2']])

best_params = grid_search.best_params_
print(f"Meilleurs hyperparamètres: {best_params}")

best_clf = EllipticEnvelope(**best_params)

y_pred_elliptic_envelope = best_clf.fit_predict(df_2d[['X1', 'X2']])

plotAnomalies2D(df_2d, clf_name, best_clf)
```
Il est à noter qu'une recherche des meilleurs hyper-paramètres est effectué via une *grid search*, ce qui évite de tester les paramètres "à la main".

#### **Paramètres importants:**

1. **`support_fraction`:**
   - Fraction minimale des points considérés comme non-anormaux lors de l’estimation de la covariance.
   - Plus faible, elle améliore la robustesse mais peut ignorer plus de points normaux.

2. **`contamination`:**
   - Fraction estimée d’anomalies dans les données.

3. **`random_state`:**
   - Gère la reproductibilité des résultats.


#### **Avantages:**

1. **Simplicité mathématique:**
   - Basée sur des concepts statistiques solides et largement utilisés (distribution normale multivariée, distance de Mahalanobis).

2. **Efficace pour des données gaussiennes:**
   - Si les données suivent effectivement une distribution gaussienne multivariée, cette méthode est optimale.

3. **Robustesse avec le MCD:**
   - L’utilisation du MCD pour initialiser $\mu$ et $\Sigma$ rend la méthode robuste face à un petit nombre d’anomalies.


#### **Limitations:**

1. **Hypothèse de normalité:**
   - L’efficacité de l’enveloppe elliptique repose sur l’hypothèse que les données normales suivent une distribution gaussienne multivariée.
   - Si les données ne suivent pas cette hypothèse, la méthode peut être moins performante.

2. **Moins efficace dans les hautes dimensions:**
   - Les estimations de la covariance deviennent moins fiables avec des dimensions élevées (problème de la malédiction de la dimensionnalité).

3. **Risque de sur-apprentissage:**
   - Si les données contiennent trop d’anomalies ou d’outliers, elles peuvent influencer les estimations de $\mu$ et $\Sigma$.



<br>
<br>

---

### 2. Isolation Forest
L'*isolation forest* est une méthode efficace pour la détection d'anomalies qui repose sur l'idée que les anomalies ou points atypiques sont plus faciles à isoler que les points normaux dans un ensemble de données car **les anomalies sont rares et différentes**.   
L'isolation forest exploite ces deux caractéristiques en construisant plusieurs arbres de séparation (arbres d'isolement) et en mesurant combien de coupures sont nécessaires pour isoler chaque point:

#### a. **Construction des arbres d'isolement:**
   - Chaque arbre est construit en divisant récursivement l'ensemble de données le long de dimensions choisies aléatoirement.
   - À chaque division, un seuil est choisi aléatoirement dans la plage de valeurs de la dimension sélectionnée.
   - Un arbre est construit jusqu'à ce que:
     - Chaque point soit isolé dans une feuille, **ou**
     - Une profondeur maximale soit atteinte.

#### b. **Répétition avec plusieurs arbres:**
   - Pour obtenir des résultats robustes, on construit plusieurs arbres (généralement entre 100 et 1000).

#### c. **Calcul de la profondeur moyenne:**
   - Pour chaque point, on mesure la profondeur moyenne à laquelle il est isolé dans tous les arbres.
   - Un point facilement isolé (faible profondeur) est considéré comme une anomalie.
   - Les points qui nécessitent beaucoup de divisions (profondeur élevée) sont considérés comme normaux.

#### d. **Score d'anomalie:**
   - Le score est une fonction inverse de la profondeur moyenne.
   - Un score élevé indique un point probablement anormal.

#### **Avantages de l'isolation forest:**
1. **Efficacité:**
   - Complexité linéaire (O(n)) en termes de taille des données, ce qui en fait une méthode adaptée aux grands ensembles de données.
2. **Peu d'hyperparamètres:**
   - Nombre d’arbres ($n_{\text{trees}}$) et sous-échantillon (subsample) sont les principaux paramètres.
   - Pas besoin de normaliser ou standardiser les données.
3. **Flexibilité:**
   - Fonctionne bien pour les ensembles de données de grande dimension et hétérogènes.
4. **Détection automatique:**
   - L’algorithme ne nécessite pas de modélisation explicite des données normales ou anormales.


<br>



```python
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV

clf_name='Isolation Forest'
clf = IsolationForest()

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_samples': [0.1, 0.5, 1.0],
    'contamination': [0.05, 0.1, 0.2, 'auto'],
    'max_features': [1, 2]
}

grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=3,
    verbose=True,
    n_jobs=-1
)

grid_search.fit(df_2d[['X1', 'X2']])

best_params = grid_search.best_params_
print(f"Meilleurs hyperparamètres: {best_params}")

best_clf = IsolationForest(**best_params)

y_pred_isolation_forest = best_clf.fit_predict(df_2d[['X1', 'X2']])

plotAnomalies2D(df_2d, clf_name, best_clf)
```

#### **Paramètres clés dans `IsolationForest`:**
1. **`n_estimators`**: Nombre d’arbres dans la forêt. Un plus grand nombre améliore la robustesse, mais augmente le temps de calcul.
2. **`max_samples`**: Taille du sous-échantillon utilisé pour construire chaque arbre. Par défaut, utilise \( \min(256, \text{taille totale des données}) .
3. **`contamination`**: Proportion attendue d’anomalies dans les données. Si inconnue, choisissez une valeur comme 0.05.
4. **`random_state`**: Pour la reproductibilité des résultats.


#### **Limitations:**
1. **Sensibilité au paramètre de contamination:**
   - Si la proportion d'anomalies est mal estimée, cela peut affecter les résultats.
2. **Risque d'erreurs pour des données très complexes:**
   - Les anomalies proches des clusters normaux peuvent passer inaperçues.
3. **Moins efficace pour des distributions non tabulaires:**
   - Peut avoir du mal à traiter des relations très non linéaires entre les variables.


<br>
<br>

---

### 3. Local Outlier Factor 
Le *Local Outlier Factor* (LOF) est une méthode de détection d'anomalies basée sur une analyse locale de la densité. Contrairement aux méthodes globales, qui comparent chaque point à l'ensemble des données, le LOF compare la densité locale d'un point à celle de ses voisins proches. Un point est considéré comme une anomalie si sa densité locale est significativement plus faible que celle de ses voisins:

#### 1. **Distance $k$-plus-proche-voisin ($k$-NN):**
   - La distance $k$-NN d'un point est la distance au $k$-ième plus proche voisin. Cette distance est utilisée pour définir la "localité".

#### 2. **Distance atteignable:**
   - La distance atteignable entre deux points $A$ et $B$ est définie comme:
    $ \text{distance atteignable}(A, B) = \max(\text{distance k-NN de B}, \text{distance euclidienne}(A, B))$ 
   - Cela permet d'atténuer l'effet des points très proches dans des zones de forte densité.

#### 3. **Densité locale atteignable:**
   - La densité locale atteignable d’un point $A$ est inversement proportionnelle à la distance moyenne atteignable entre $A$ et ses $k$-plus-proches-voisins:
    $\text{densité locale atteignable}(A) = \frac{k}{\sum_{B \in \text{voisins de } A} \text{distance atteignable}(A, B)}$

#### 4. **Facteur de détection locale (LOF):**
   - Le LOF d’un point $A$ est le rapport entre la densité locale moyenne de ses $k$-plus-proches-voisins et sa propre densité locale atteignable:
    $\text{LOF}(A) = \frac{\text{moyenne des densités locales des voisins de } A}{\text{densité locale atteignable}(A)}$
   - Si $ \text{LOF}(A) \approx 1 $, le point est normal.
   - Si $ \text{LOF}(A) > 1 $, le point est un outlier. Plus le LOF est grand, plus le point est isolé.

```python
from sklearn.neighbors import LocalOutlierFactor

clf_name = 'Local Outlier Factor'
clf = LocalOutlierFactor(n_neighbors=5, contamination=0.5, novelty=True)
clf.fit(df[['X1','X2']])

plotAnomalies2D(df_2d, clf_name, clf)
```

#### **Paramètres importants:**

1. **`n_neighbors`:**
   - Nombre de voisins à considérer. Influence directement la "localité". Un $k$ plus grand favorise une évaluation plus globale, un $k$ plus petit favorise une détection plus locale.

2. **`contamination`:**
   - Fraction estimée des anomalies dans les données. Permet de calibrer le seuil de détection.

3. **`metric`:**
   - Méthode utilisée pour calculer les distances (euclidienne par défaut, mais d’autres métriques comme Minkowski ou Mahalanobis sont possibles).

#### **Avantages du LOF**

1. **Analyse locale:**
   - Contrairement aux méthodes globales (comme l’Isolation Forest), le LOF détecte des anomalies qui peuvent être "normales" à l'échelle globale, mais inhabituelles dans leur voisinage local.

2. **Non-paramétrique:**
   - Pas besoin de supposer une distribution particulière des données.

3. **Adaptabilité:**
   - Fonctionne bien avec des données hétérogènes ou ayant des densités variables.


#### **Inconvénients du LOF**

1. **Sensibilité au choix de $k$:**
   - Le nombre de voisins $k$ influence directement les résultats. Une mauvaise sélection peut entraîner des erreurs.

2. **Complexité:**
   - La complexité temporelle est quadratique ($O(n^2)$) pour le calcul des distances, bien que des optimisations ($k$-d arbres, approximations) réduisent ce coût.

3. **Difficulté en haute dimension:**
   - Comme beaucoup d'algorithmes basés sur les distances, le LOF souffre de la "malédiction de la dimensionnalité".



<br>
<br>

---

### 4. One Class SVM
Le *One-Class SVM* est l'extension des machines à vecteurs de support (SVM) adaptée aux cas où l'on dispose uniquement de données normales pour l'entraînement, sans exemples d'anomalies:

#### 1. **Projection dans un espace à haute dimension:**
   - Les données sont transformées via une fonction noyau pour mieux capturer des structures complexes.

#### 2. **Optimisation:**
   - Le modèle optimise un hyperplan ou une hypersphère qui sépare les données normales du reste, en minimisant une fonction objectif qui équilibre la marge et le nombre de points rejetés.

#### 3. **Prédiction:**
   - Après l'entraînement, le modèle attribue une étiquette à chaque point:
    - $+1$: Données normales (inliers).
    - $-1$: Anomalies (outliers).
    
```python
from sklearn.svm import OneClassSVM

clf_name = 'One Class SVM'
clf = OneClassSVM(nu=0.15, kernel="rbf", gamma=0.33)
clf.fit(df_2d[['X1','X2']])

y_pred_one_class_svm = clf.fit_predict(df_2d[['X1','X2']])

plotAnomalies2D(df_2d, clf_name, clf)
```

#### **Paramètres importants:**

1. **`kernel`:**
   - Définit le type de noyau utilisé pour capturer la frontière:
     - `"linear"`: Hyperplan linéaire.
     - `"rbf"` (par défaut): Non-linéaire, adapté aux distributions complexes.

2. **`nu`:**
   - Fraction des points rejetés comme anomalies. Correspond à la contamination attendue dans les données.

3. **`gamma`:**
   - Paramètre de forme du noyau radial. Contrôle la courbure de la frontière.

4. **`contamination` (post-analyse):**
   - Si besoin, ajustez la proportion des anomalies après entraînement pour optimiser les prédictions.

#### **Avantages du One-Class SVM**

1. **Capacité à modéliser des distributions complexes:**
   - Grâce à l’utilisation de noyaux, il peut capturer des frontières non linéaires.

2. **Apprentissage non supervisé:**
   - Ne nécessite que des données normales pour s’entraîner.

3. **Robustesse:**
   - Fonctionne bien avec des distributions complexes et des dimensions élevées.


#### **Limitations du One-Class SVM**

1. **Sensibilité au choix des hyperparamètres:**
   - Les performances dépendent fortement de la sélection du noyau, du paramètre $ \nu $ (contamination) et de $ \gamma $ (influence du noyau).

2. **Complexité:**
   - Sa complexité temporelle est quadratique à cubique $ O(n^2) $ à $ O(n^3) $, ce qui le rend coûteux pour de grands ensembles de données.

3. **Difficulté avec des données déséquilibrées:**
   - Si une grande proportion d’anomalies est présente dans les données d’entraînement, cela peut nuire aux performances.

   
<br>
<br>

---

### 5. Minimum covariance determinent
La méthode du **Déterminant de Covariance Minimale (MCD)** est une technique robuste utilisée en statistique, notamment pour l'estimation de paramètres dans des distributions multivariées et pour la détection de valeurs aberrantes:

#### **a. Sélection aléatoire d'un sous-ensemble de points de données**
- On commence par choisir aléatoirement un sous-ensemble $\mathcal{h}$ de points de données parmi l'ensemble total $\mathcal{n}$.
- Typiquement, $\mathcal{h}$ est choisi pour être suffisamment grand pour permettre une estimation robuste, mais plus petit que $\mathcal{n}$, afin de limiter l'influence des valeurs aberrantes.


#### **b. Calculer la matrice de covariance et son déterminant**
- Pour chaque sous-ensemble sélectionné, on calcule:
  - La **moyenne** vectorielle $\mu_h$ des points du sous-ensemble.
  - La **matrice de covariance**  $S_h$  du sous-ensemble.
  - Le **déterminant de la matrice de covariance**  $\det(S_h)$ , qui reflète le volume de la distribution estimée à partir du sous-ensemble.

> **Pourquoi le déterminant?**  
Le déterminant de la matrice de covariance mesure l'étendue ou la "dispersion" des points dans l'espace. **Plus ce déterminant est petit, plus la distribution estimée est compacte.**


#### **c. Répéter plusieurs fois et conserver le déterminant le plus petit**
- On répète les étapes 1 et 2 un grand nombre de fois avec différents sous-ensembles aléatoires.
- Finalement, on conserve le sous-ensemble  $\mathcal{h}$  qui donne le **plus petit déterminant**  $\det(S_h)$.     
Cela permet de sélectionner un sous-ensemble qui capture le cœur des données, en minimisant l'effet des valeurs aberrantes.


#### **d. Calculer la distance de Mahalanobis**
- À partir de la moyenne $\mu_h$ et de la matrice de covariance $S_h$ obtenues à l'étape précédente, on peut calculer la **distance de Mahalanobis**  $d_M$  pour chaque observation $x$ de l'ensemble initial.

La distance de Mahalanobis est définie comme: $d_M(x) = \sqrt{(x - \mu_h)^T S_h^{-1} (x - \mu_h)}$ 


> **Interprétation**: Cette distance mesure la "proximité" d'une observation par rapport à la distribution estimée. Elle tient compte de la corrélation entre les variables, contrairement à une simple distance euclidienne.


#### **e. Définir un seuil pour identifier les valeurs aberrantes**
- Une fois les distances de Mahalanobis calculées pour toutes les observations, il faut définir un seuil pour distinguer les observations normales des valeurs aberrantes.
- Les observations dont la distance dépasse ce seuil (souvent basé sur une distribution  $\chi^2$  avec  $p$  degrés de liberté, où $p$ est le nombre de variables) sont considérées comme des **outliers**.


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def covariance_matrix(X):
   return np.cov(X.T)  

def mahalanobis_distance(X, mean, cov_matrix):
   inv_cov = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6)
   diffs = X - mean
   # Calling diagonal to take only relative measure of distance with regards to the mean of distribution, not comparing to others
   return np.sqrt(np.diag(np.dot(np.dot(diffs, inv_cov), diffs.T)))


# MCD implementation
def mcd(X, h, n_iterations=100):
   np.random.seed(42)
   best_det = float('inf')
   best_subset = None
   
   # Random selection of subsets and optimisation
   for _ in range(n_iterations):
      subset = X[np.random.choice(X.shape[0], h, replace=False)]
      cov_matrix = covariance_matrix(subset)
      det = np.linalg.det(cov_matrix)
      if det < best_det:
         best_det = det
         best_subset = subset
   
   # Calculation of the mean and covariance matrix for the best subset
   best_mean = np.mean(best_subset, axis=0)
   best_cov = covariance_matrix(best_subset)
   
   # Calculation of Mahalanobis distances for all observations
   distances = mahalanobis_distance(X, best_mean, best_cov)
   return distances


data = np.array([X1, X2]).T

# Standardisation of data to avoid scale effects
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Threshold based on the χ² law:
threshold = chi2.ppf(0.775, df=data.shape[1])

# Threshold 85ᵉ centile:
threshold = np.percentile(distances, 85)


# Test with different values of h
h_values = [int(data.shape[0] * frac) for frac in [0.5, 0.6, 0.7]]

fig, axs = plt.subplots(1, len(h_values), figsize=(15, 5), sharex=True, sharey=True)

for i, h in enumerate(h_values):
   distances = mcd(data, h)
   outliers_pred = distances > threshold
   print(f"h={h}, nombre d'outliers: {np.sum(outliers_pred)}")
   
   # Display results
   axs[i].scatter(data[:, 0], data[:, 1], c=outliers_pred, cmap='coolwarm')
   axs[i].set_title(f"h={h}, Outliers={np.sum(outliers_pred)}")
   axs[i].set_xlabel('X1')
   axs[i].set_ylabel('X2')

plt.tight_layout()
plt.show()

# Histogram of Mahalanobis distances
for h in h_values:
   distances = mcd(data, h)
   plt.figure(figsize=(22, 8))
   plt.hist(distances, bins=30, color='skyblue', alpha=0.7, label=f"h={h}")
   plt.axvline(threshold, color='red', linestyle='--', label=f'Seuil={threshold:.2f}')
   plt.title(f"Histogramme des distances Mahalanobis (h={h})")
   plt.xlabel("Distance")
   plt.ylabel("Fréquence")
   plt.legend()
   plt.show()
```

Le MCD est particulièrement adapté dans les contextes suivants:
1. **Robustesse**: Il réduit l'impact des valeurs aberrantes sur les estimations.
2. **Fiabilité**: L'utilisation du déterminant garantit que l'estimation est fondée sur le cœur dense des données.
3. **Applications pratiques**: Il est souvent utilisé dans l'identification d'anomalies, la classification robuste, et la détection d'observations atypiques dans les ensembles de données multivariées.


### **Comparaison des méthodes:**

| Méthode               | Nature               | Avantages                              | Limites                              |
|-----------------------|----------------------|---------------------------------------|--------------------------------------|
| **One-Class SVM**     | Globale, Kernel      | Modélise des frontières complexes     | Sensible aux hyperparamètres         |
| **Isolation Forest**  | Globale             | Rapide, robuste, non paramétrique     | Moins efficace sur de petites données|
| **Local Outlier Factor** | Locale              | Détecte des anomalies locales         | Sensible à $k$, coûteux en calcul  |
| **Elliptic Envelope** | Globale             | Simple, efficace si données gaussiennes | Inefficace pour des distributions non-gaussiennes |


<br>
<br>
<br>

## Détection de nouveauté
