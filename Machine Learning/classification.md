<h1 align='center'> Machine learning - Classification 🗂️</h1>

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
SVM est un algorithme de classification supervisée pour séparer deux classes en maximisant la **marge** entre les deux points les plus proches de l'hyperplan séparateur.


```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création du modèle SVM avec un noyau linéaire
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


## Classificateur Bayesien naïf 
La classification naïve bayésienne est un type de classification bayésienne probabiliste simple basée sur le théorème de Bayes avec une forte **indépendance** des hypothèses. Elle met en œuvre un classifieur bayésien naïf, ou classifieur naïf de Bayes, appartenant à la famille des classifieurs linéaires.


```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création du modèle Bayes naïf
gnb = GaussianNB()

# Entraînement du modèle
gnb.fit(X_train, y_train)

# Prédiction
y_pred = gnb.predict(X_test)

# Affichage des résultats
print(f"Prediction: {y_pred})")

# Affiche les probabilités prédites par le modèle pour chaque classe pour les données de test
print(f"Probas: {gnb.predict_proba(X_test)}")

# Affiche les log-probabilités (log vraisemblance) pour chaque classe.
print(f"Log probas: {gnb.predict_log_proba(X_test)}")


# Affichage des performances
print(f"Generalization error: {np.sum(np.not_equal(y_pred, y_test))/len(y_test)}")
print(f"Generalization score: {digits_nbc.score(X_test, y_test)}")
print(f"Confusion matrix: {confusion_matrix(y_test, y_pred)}")
```

<br>
<br>

## Arbres de décision

Un arbre de décisions est un algorithme d'apprentissage supervisé non paramétrique, utilisé à la fois pour les tâches de classification et de régression. Il possède une structure hiérarchique et arborescente, qui se compose d'un nœud racine, de branches, de nœuds internes et de nœuds feuille.


```python
from sklearn import tree
from sklearn.model_selection import train_test_split

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création de l'arbre
dt = tree.DecisionTreeClassifier(criterion='entropy',max_depth=10, min_samples_leaf=10)

# Entraînement du modèle
dt.fit(X_train, y_train)

# Affichage de l'arbre
tree.plot_tree(dt)
```

Il est à noter que dans l'instanciation de l'arbre plusieurs paramètres sont donnés:
1. **`criterion`**: C'est le **critère de division** qui spécifie la fonction de mesure de la qualité d'un découpage dans l'arbre de décision.   
   Lorsque `criterion='entropy'`, cela signifie que l'arbre utilisera l'**entropie** comme critère pour décider comment diviser les données à chaque nœud: L'entropie est une mesure de l'incertitude ou du désordre dans les données. Plus précisément, elle est utilisée pour calculer la *gain d'information* à chaque division, et l'arbre choisit la division qui maximise ce gain d'information.   
   L'autre option courante pour `criterion` est `gini`, qui utilise l'**indice de Gini** comme critère de décision. 

2. **`max_depth`**: C'est la **Profondeur maximale de l'arbre** qui définit le nombre maximal de niveaux de nœuds dans l'arbre, à partir de la racine jusqu'aux feuilles (les nœuds finaux où des prédictions sont faites).   
   Limiter la profondeur d'un arbre est une manière courante de prévenir le **sur-apprentissage** (overfitting) en réduisant la complexité du modèle. Un arbre trop profond pourrait mémoriser trop les données d'entraînement, ce qui peut nuire à sa capacité à généraliser à de nouvelles données.

3. **`min_samples_leaf`**: C'est le **nombre minimum d'échantillons par feuille**. Ce paramètre permet de contrôler la taille des feuilles de l'arbre et peut aider à éviter des feuilles trop petites, ce qui peut réduire le sur-apprentissage.   
   Ce paramètre a une influence sur la structure de l'arbre. Si cette valeur est augmentée, un arbre plus "large" et moins "profond" sera obtenu. Cela peut également rendre le modèle plus robuste et moins susceptible de s'adapter aux bruits ou aux variations spécifiques dans les données d'entraînement.




Pour afficher les frontières de décision:
```python
def plot_decision_boundary_tree(t, X, y, fig_size=(22, 8)):
    plot_step = 0.02
    x0_min, x0_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, plot_step), np.arange(x1_min, x1_max, plot_step))
    yypred = t.predict(np.c_[xx0.ravel(),xx1.ravel()])
    yypred = yypred.reshape(xx0.shape)
    plt.figure(figsize=fig_size)
    plt.contourf(xx0, xx1, yypred, cmap=plt.cm.Paired)
    y_pred = t.predict(X)
    Xblue_good = X[np.equal(y,-1)*np.equal(y,y_pred)]
    Xblue_bad  = X[np.equal(y,-1)*np.not_equal(y,y_pred)]
    Xred_good  = X[np.equal(y,1)*np.equal(y,y_pred)]
    Xred_bad   = X[np.equal(y,1)*np.not_equal(y,y_pred)]
    plt.scatter(Xblue_good[:,0],Xblue_good[:,1],c='b')
    plt.scatter(Xblue_bad[:,0],Xblue_bad[:,1],c='c',marker='v')
    plt.scatter(Xred_good[:,0],Xred_good[:,1],c='r')
    plt.scatter(Xred_bad[:,0],Xred_bad[:,1],c='m',marker='v')
    plt.show()

plot_decision_boundary_tree(dt, X, y)
```

## Forêts aléatoires 

Les **forêts aléatoires** (*Random Forest*) sont un algorithme d'**ensemble learning** dont l'idée est de combiner plusieurs arbres de décision pour obtenir une prédiction plus robuste et plus précise. En particulier, l'algorithme construit une **forêt** d'arbres de décision en introduisant de l'**aléatoire** à deux niveaux:
1. **Échantillonnage bootstrap** des données (sub-sampling des données d'entraînement).
2. **Sélection aléatoire des caractéristiques** à chaque division (split) dans chaque arbre.

L'algorithme des forêts aléatoires fonctionne en plusieurs étapes:

#### 1. **Création d'arbres de décision aléatoires**
   - L'idée de base est de créer un ensemble (ou une "forêt") de plusieurs **arbres de décision**, chacun étant construit à partir d'un sous-ensemble aléatoire des données d'entraînement.
   - L'**échantillonnage bootstrap** est utilisé pour créer des sous-ensembles de données. Cela signifie que pour chaque arbre, on tire aléatoirement des échantillons de l'ensemble d'entraînement, avec remplacement (certains exemples peuvent être répétés, d'autres non).
   
#### 2. **Sélection aléatoire des caractéristiques**
   - À chaque nœud d'un arbre de décision, au lieu d'examiner toutes les caractéristiques pour décider de la meilleure division, un sous-ensemble aléatoire des caractéristiques est choisi et utilisé pour effectuer la division à ce nœud.
   - Cela permet d'introduire encore plus de diversité entre les arbres et de réduire la corrélation entre eux, ce qui améliore la performance du modèle global.

#### 3. **Entraînement des arbres**  
   - Chaque arbre est entraîné sur un sous-ensemble différent de données (grâce au bootstrap) et en utilisant un sous-ensemble aléatoire de caractéristiques à chaque division. 
   - Cela signifie que chaque arbre peut avoir des structures légèrement différentes, ce qui aide à réduire le sur-apprentissage (overfitting) lorsque ces arbres sont combinés.

#### 4. **Prédiction finale**
   - Une fois que tous les arbres ont été entraînés, la prédiction de la forêt est effectuée par **vote majoritaire** (pour la classification) ou **moyenne** (pour la régression).   
  Chaque arbre "vote" pour une classe, et la classe ayant le plus grand nombre de votes devient la prédiction finale.

### Avantages des Forêts Aléatoires:
1. **Robustesse**
2. **Réduction du sur-apprentissage**
3. **Capacité à gérer des données complexes**

### Inconvénients des Forêts Aléatoires:
1. **Complexité computationnelle**
2. **Moins interprétable qu'un arbre de décision simple**


```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer une forêt aléatoire
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

# Entraîner la forêt aléatoire
rf.fit(X_train, y_train)

# Prédire les labels pour les données de test
y_pred = rf.predict(X_test)

# Calculer l'exactitude
accuracy = (y_pred == y_test).mean()
print(f"Accuracy: {accuracy}")

# Importance des features
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(22, 8))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
```

Voici quelques hyperparamètres importants que à ajuster pour améliorer la performance du modèle de forêt aléatoire:
- **`n_estimators`**: Le nombre d'arbres de décision dans la forêt. Plus il y a d'arbres, plus la prédiction sera robuste, mais cela augmente également le coût computationnel.
- **`max_features`**: Le nombre maximal de caractéristiques à considérer pour chaque division. Cela contrôle le degré de diversité entre les arbres de la forêt.
- **`bootstrap`**: Si l'échantillonnage bootstrap est utilisé pour la création des sous-ensembles de données (par défaut, c'est vrai).
- **`oob_score`**: Le score Out-Of-Bag (OOB), qui permet d'estimer la performance du modèle sans utiliser un ensemble de validation séparé.


<br>
<br>


## Boosting
### AdaBoost 
AdaBoost (pour Adaptive Boosting) est un algorithme d'ensemble learning qui combine plusieurs classificateurs faibles (c'est-à-dire des modèles qui, seuls, n'ont pas une grande performance) pour créer un classificateur plus puissant. Il fonctionne en "boostant" progressivement la performance des modèles en les combinant de manière adaptative.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création du modèle
boosted_forest = AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='entropy',max_depth=3), n_estimators=100)

# Entraînement du modèle
boosted_forest.fit(X_train, y_train)

# Affichage des frontières de décision
plot_decision_boundary(boosted_forest, X_train,  y_train)

# Affichage des métriques
print(f"Training score: {boosted_forest.score(X_train, y_train)}")
print(f"Testing score: {boosted_forest.score(X_test, y_test)}")
```

*<u>Remarque</u>: Le paramètre **`n_estimators`** fait référence au **nombre d'estimateurs faibles** (classificateurs) que l'algorithme va entraîner. Il spécifie combien de classificateurs faibles l'algorithme AdaBoost va entraîner dans le cadre de la procédure de boosting.*


### Gradient Boost

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création du modèle
gtb = GradientBoostingClassifier(n_estimators=100)

# Entraînement du modèle
gtb.fit(X_train, y_train)

# Affichage des frontières de décision
plot_decision_boundary(gtb, X_train,  y_train)

# Affichage des métriques
print(f"Training score: {gtb.score(X_train, y_train)}")
print(f"Testing score: {gtb.score(X_test, y_test)}")
```