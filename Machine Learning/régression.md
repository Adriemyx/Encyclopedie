<h1 align='center'> Machine learning - Régression </h1>

Dans ce document sera présenté quelques bases de code, notamment avec la libraire `scikit-learn`, pour faire de la régression sur python.

## Régression linéaire 
A utiliser dans le cas d'un problème **supervisé** avec un label **quantitatif** et dont la relation entre la cible et l'entrée (**une variable**) semble être linéaire $y = \alpha \times x + \beta$.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciation de sklearn.linear_model.LinearRegression
lr = LinearRegression()

# Entraînement du modèle de régression linéaire sur les données d'entraînement
lr.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred = lr.predict(X_test)
```

### Métriques de test
Pour tester de la pertinence de la régression, il faut pouvoir l'éavluer. Pour évaluer les performances d'un modèle de régression linéaire avec `scikit-learn`, il existe plusieurs métriques de test prédéfinies:

1. **MSE (Mean Squared Error)**: C'est la moyenne des carrés des erreurs. Il mesure la dispersion des erreurs.

   ```python
   from sklearn.metrics import mean_squared_error
   mse = mean_squared_error(y_true, y_pred)
   ```

2. **RMSE (Root Mean Squared Error)**: C'est la racine carrée du MSE. Il est dans la même unité que la variable cible, ce qui le rend plus interprétable.

   ```python
   rmse = mean_squared_error(y_true, y_pred, squared=False)
   ```

3. **MAE (Mean Absolute Error)**: C'est la moyenne des erreurs absolues. Cela donne une idée de la taille des erreurs en unités de la variable cible.

   ```python
   from sklearn.metrics import mean_absolute_error
   mae = mean_absolute_error(y_true, y_pred)
   ```

4. **R² (Coefficient de détermination)**: Cette métrique indique la proportion de la variance dans la variable cible qui est prédit par le modèle. Elle varie entre 0 et 1, avec des valeurs plus élevées indiquant un meilleur ajustement.

   ```python
   from sklearn.metrics import r2_score
   r2 = r2_score(y_true, y_pred)
   ```



### Methodes de détection d'*outliers*
Certains modèles peuvent présenter des métriques assez mauvaises à cause de certaines valeurs abérantes qui induisent le modèle en erreur. Pour détecter ces *"outliers"*, il existe plusieurs méthodes:


1. **Boîte à Moustaches (*Boxplot*)**: Un boxplot visualise la distribution des données et identifie les outliers comme des points situés en dehors des moustaches ($1,5$ fois l'intervalle interquartile). Cela permet d'identifier **visuellement** les outliers.

```python
import matplotlib.pyplot as plt

plt.boxplot(data)
plt.title('Boxplot des données')
plt.show()
```


2. **Distance de Cook**: La distance de Cook mesure l'influence d'un point de données sur les coefficients du modèle de régression. Elle évalue l'impact d'une observation sur les valeurs ajustées: Un point avec une distance de Cook supérieure à $1$ ou à $\frac{4}{n}$, où $n$ est le nombre d'observations, peut être considéré comme influent.   

La distance de cook peut se calculer manuellement ou via statsmodels:
  
```python
import statsmodels.api as sm

model = sm.OLS(y, X).fit()
influence = model.get_influence()
cooks_d = influence.cooks_distance
```


3. **Résidus Studentisés**: Les résidus studentisés sont des résidus standardisés qui tiennent compte de la variance des résidus, permettant d'identifier les observations atypiques: Des résidus studentisés supérieurs à $3$ ou inférieurs à $-3$ indiquent des points qui s'écartent de la tendance générale.

```python
studentized_residuals = influence.resid_studentized
```


Il est possible également de faire des tests statistiques.


<br>
<br>
<br>

## Régression linéaire multiple
A utiliser dans le cas d'un problème **supervisé** avec un label **quantitatif** et dont la cible semblerait être proche d'une **combinaision linéaire des entrées** (**plusieurs variables**) semble être linéaire $y = \sum_{i} \alpha_{i} \times x + \beta_{0}$.

Le code pour effectuer une régression linéaire multiple est le même que celui pour créer une régression linéaire sauf que maintenant, $\mathcal{X}$ est une matrice où chaque colonne représente une variable indépendante (ou prédicteur), et chaque ligne représente une observation. 

### Sélection des variables
Toutes les variables à disposition dans $\mathcal{X}$ ne sont pas forcément nécessaire pour la prédiction de la sortie. Il faut alors procéder à la sélection des variables. Il existe plusieurs méthodes (cf. [Pre-processing doc.](pre-processing.md#sélection-des-variables)
).    
Une fois les variables pertienentes sélectionnées, une régression linéaire multiple peut être réalisée. 

<br>
<br>

### Régression avec régularisation
#### Régression linéaire avec régularisation Lasso
La régression lasso présente plusieurs intérêts, notamment:

1. **Sélection de variables**: Lasso pénalise les coefficients des variables, ce qui peut conduire à mettre certains d'entre eux à zéro. Cela permet d'identifier et de conserver uniquement les variables les plus pertinentes, simplifiant ainsi le modèle.

2. **Réduction du surapprentissage**: En ajoutant une pénalité sur la complexité du modèle, la régression lasso aide à réduire le risque de surapprentissage, ce qui peut améliorer la capacité de généralisation du modèle sur des données non vues.

3. **Robustesse face à la multicolinéarité**: Lorsque des variables sont corrélées, lasso peut choisir l'une d'elles et ignorer les autres, offrant ainsi une solution stable même en présence de multicolinéarité.

4. **Interprétabilité**: En réduisant le nombre de variables, le modèle devient plus interprétable. Cela facilite l'analyse et la compréhension des relations entre les variables et la variable cible.

5. **Flexibilité**: Lasso peut être utilisé dans de nombreux contextes, qu'il s'agisse de régression linéaire, de classification, ou d'autres types de problèmes de modélisation.

En somme, la régression lasso est un outil puissant pour gérer la complexité des modèles, améliorer leur performance et faciliter l'interprétation des résultats.

Voici le code d'implémentation d'une régression Lasso simple:
```python
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Coefficient de contrôle de l'ampleur de la pénalisation (L1) appliquée aux coefficients de la régression
# Plus il est élevé, plus la pénalisation est grande
alpha = 0.5

# Instanciation de sklearn.linear_model.Lasso
lasso_regressor = Lasso(alpha=alpha)

# Entraînement du modèle de régression Lasso sur les données d'entraînement
lasso_regressor.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred_lasso = lasso_regressor.predict(X_test)

# Affichage des coefficients du modèle
coefficients = pd.Series(lasso_regressor.coef_.flatten(), index=X.columns)
print("\nCoefficients du modèle Lasso:")
print(coefficients)
```


#### Régression linéaire avec régularisation Ridge
La régression **Ridge** est idéale pour les situations avec multicolinéarité et quand on veut des coefficients plus stables sans nécessairement réduire le nombre de variables.

1. **Gestion de la multicolinéarité**: Ridge est particulièrement efficace lorsque les variables d'entrée sont fortement corrélées. En ajoutant une pénalité sur la somme des carrés des coefficients (pénalité l2), elle stabilise les estimations et réduit la variance.

2. **Amélioration de la prédiction**: En contraignant les coefficients, Ridge aide à éviter le surapprentissage, ce qui peut améliorer les performances prédictives sur des données non vues, surtout lorsque le modèle est complexe.

3. **Aucune sélection de variables**: Bien que Ridge ne réalise pas de sélection de variables (tous les coefficients restent non nuls), il fournit une estimation plus stable des coefficients, ce qui peut être souhaitable dans certains cas.


Voici le code d'implémentation d'une régression Ridge simple:
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Coefficient de contrôle de l'ampleur de la pénalisation (L2) appliquée aux coefficients de la régression
# Plus il est élevé, plus la pénalisation est grande
alpha = 0.5

# Instanciation de sklearn.linear_model.Ridge
ridge_regressor = Ridge(alpha=alpha)

# Entraînement du modèle de régression Ridge sur les données d'entraînement
ridge_regressor.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred_ridge = ridge_regressor.predict(X_test)

# Affichage des coefficients du modèle
coefficients = pd.Series(ridge_regressor.coef_.flatten(), index=X.columns)
print("\nCoefficients du modèle Ridge:")
print(coefficients)
```



#### Régression Elastic Net

La régression **Elastic Net** est préférable quand on a de nombreuses variables corrélées et qu'on souhaite réaliser une sélection de variables tout en gardant une certaine robustesse.

1. **Combinaison des avantages de Lasso et Ridge**: Elastic Net combine les pénalités l1 et l2, ce qui permet à la fois la sélection de variables (comme avec Lasso) et la gestion de la multicolinéarité (comme avec Ridge).

2. **Robustesse en cas de nombreuses variables**: Elastic Net est particulièrement utile lorsque le nombre de variables prédictives est supérieur au nombre d'observations ou en présence de variables corrélées. Il peut sélectionner plusieurs variables corrélées tout en maintenant la stabilité des estimations.

3. **Flexibilité**: En ajustant le paramètre \( l1\_ratio \), tu peux contrôler la balance entre la sélection de variables et la régularisation, permettant une personnalisation selon le problème.


Voici le code d'implémentation d'une régression Elastic Net simple:
```python
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Coefficient de contrôle de l'ampleur de la pénalisation appliquée aux coefficients de la régression
# alpha contrôle la force de la pénalisation, l1_ratio contrôle la combinaison entre l1 et l2
alpha = 0.5
l1_ratio = 0.5  # 0.5 pour un mélange égal de Lasso et Ridge

# Instanciation de sklearn.linear_model.ElasticNet
elastic_net_regressor = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)

# Entraînement du modèle de régression Elastic Net sur les données d'entraînement
elastic_net_regressor.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred_elastic_net = elastic_net_regressor.predict(X_test)

# Affichage des coefficients du modèle
coefficients = pd.Series(elastic_net_regressor.coef_.flatten(), index=X.columns)
print("\nCoefficients du modèle ElasticNet:")
print(coefficients)
```




### Régression PLS (Partial Least Squares)

La régression PLS est une méthode statistique utilisée principalement pour des situations où **le nombre de variables prédictives est élevé par rapport au nombre d'observations, ou lorsque les variables prédictives sont corrélées**. Voici les principaux intérêts et applications de la régression PLS:

1. **Réduction de la dimensionnalité**: PLS combine la réduction de dimensionnalité et la modélisation prédictive. Elle projette les données dans un espace de dimensions inférieures, facilitant ainsi l'analyse sans perdre trop d'information.

2. **Gestion de la multicolinéarité**: Lorsque les variables prédictives sont corrélées, PLS peut être plus efficace que d'autres méthodes, comme la régression linéaire ordinaire, qui peuvent donner des estimations instables en raison de la multicolinéarité.

3. **Optimisation de la prédiction**: PLS maximise la covariance entre les variables prédictives et la variable cible. Cela permet de construire des modèles prédictifs qui capturent mieux les relations entre les variables.

4. **Adaptabilité**: PLS est flexible et peut être utilisé dans divers contextes, que ce soit pour des données expérimentales, des données spectrales, ou d'autres types de données complexes.


Voici le code d'implémentation d'une régression PLS simple:
```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciation du modèle PLS avec un nombre de composantes latentes
n_components = 5  # Choisir le nombre de composantes latentes
pls = PLSRegression(n_components=n_components)

# Entraînement du modèle
pls.fit(X_train, y_train)

# Prédiction sur les données de test
y_pred_pls = pls.predict(X_test)

# Affichage des coefficients du modèle
coefficients = pd.Series(pls.coef_.flatten(), index=X.columns)
print("\nCoefficients du modèle PLS:")
print(coefficients)
```




### Régression Logistique

La régression logistique est une méthode statistique utilisée pour modéliser la relation entre une variable dépendante binaire (ou catégorique) et une ou plusieurs variables indépendantes. Elle est couramment utilisée pour des problèmes de classification où l'objectif est de prédire l'appartenance à l'une des deux catégories.    
Voici quelques caractéristiques clés de la régression logistique:

1. **Variable dépendante binaire**: La régression logistique est principalement utilisée lorsque la variable cible est binaire (par exemple, succès/échec, oui/non, 0/1).

2. **Fonction logistique**: La régression logistique utilise la fonction logistique (ou sigmoïde) pour modéliser la probabilité que la variable dépendante prenne la valeur 1. La fonction logistique est définie comme: $P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_k X_k)}}$ où \( P(Y=1|X) \) est la probabilité que \( Y \) soit égal à 1, \( \beta_0 \) est l'ordonnée à l'origine, et \( \beta_1, \beta_2, \ldots, \beta_k \) sont les coefficients des variables indépendantes.

3. **Estimation des paramètres**: Les coefficients du modèle sont généralement estimés par la méthode de maximum de vraisemblance, qui cherche à maximiser la probabilité d'observer les données données les paramètres du modèle.

4. **Interprétation des coefficients**: Les coefficients dans une régression logistique peuvent être interprétés en termes d'odds (cotes). Par exemple, un coefficient positif indique qu'une augmentation de la variable indépendante augmente les chances que la variable dépendante soit égale à 1.

5. **Extensions**: Bien qu'elle soit principalement utilisée pour des problèmes de classification binaire, la régression logistique peut également être étendue à des cas multiclasse à l'aide de techniques telles que la régression logistique multinomiale.


Voici le code d'implémentation d'une régression logistique simple:
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instanciation du modèle régression logistique
clf = LogisticRegression(random_state=0)

# Entraînement du modèle
clf.fit(X_train, y_train)

# Prédiction sur les données de test
clf.predict(X_test)
clf.predict_proba(X_test)
clf_score =  clf.score(X_test, y_test)

# Affichage du score du modèle
print(f"Score clf: {clf_score}")
print(f"Pourcentage d'erreur sur les tests: {round((1-clf_score)*100, 2)}%")

# Affichage des coefficients du modèle
coefficients = pd.Series(clf.coef_.flatten(), index=X.columns)
print("\nCoefficients du modèle logistique:")
print(coefficients)
```

<br>
<br>

## Régression Gaussienne

```python
from smt.surrogate_models import KRG

# Instanciation du modèle
gpr = KRG()

# Entraînement du modèle
gpr.set_training_values(X_data, y_data)
gpr.train()

print('Theta optimal', gpr.optimal_theta)
```


## Forets aléatoires

### Forêts aléatoires pour la régression

Les **forêts aléatoires** (*Random Forest*) sont un algorithme d'**ensemble learning** dont l'idée est de combiner plusieurs arbres de décision pour obtenir une prédiction plus robuste et plus précise. En particulier, l'algorithme construit une **forêt** d'arbres de décision en introduisant de l'**aléatoire** à deux niveaux:
1. **Échantillonnage bootstrap** des données (sub-sampling des données d'entraînement).
2. **Sélection aléatoire des caractéristiques** à chaque division (split) dans chaque arbre.

L'algorithme des forêts aléatoires fonctionne en plusieurs étapes:

#### 1. **Création d'arbres de décision aléatoires**
   - Comme dans la version pour la classification, on crée plusieurs **arbres de décision** en échantillonnant aléatoirement les données d'entraînement à chaque arbre, à l'aide de l'**échantillonnage bootstrap**.
   - Chaque arbre est donc construit sur un sous-ensemble différent des données, ce qui introduit de la diversité et améliore la généralisation du modèle.

#### 2. **Sélection aléatoire des caractéristiques**
   - À chaque nœud de l'arbre, au lieu d'examiner toutes les caractéristiques pour choisir la meilleure coupure, un sous-ensemble aléatoire des caractéristiques est choisi. Cela contribue à la diversité des arbres et à la réduction de la corrélation entre eux.

#### 3. **Entraînement des arbres**
   - Chaque arbre est entraîné indépendamment sur un sous-ensemble des données, avec une sélection aléatoire des caractéristiques à chaque nœud, ce qui permet de créer des arbres avec des structures différentes.

#### 4. **Prédiction finale**
   - La prédiction finale de la forêt aléatoire est la **moyenne** des prédictions de tous les arbres. Cela permet d'obtenir une estimation plus précise que celle d'un seul arbre de décision.

---

### Avantages des Forêts Aléatoires:
1. **Robustesse**
2. **Précision**
3. **Adaptabilité**

### Inconvénients des Forêts Aléatoires pour la régression:
1. **Complexité computationnelle**
2. **Moins interprétable**
3. **Risque d'overfitting avec des arbres trop profonds**


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Créer une forêt aléatoire pour la régression
rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

# Entraîner la forêt aléatoire
rf_regressor.fit(X_train, y_train)

# Prédire les valeurs pour les données de test
y_pred = rf_regressor.predict(X_test)

# Calculer l'erreur absolue moyenne (MAE) ou l'erreur quadratique moyenne (RMSE)
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Visualiser les résultats
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Vrai y')
plt.ylabel('Prédiction y')
plt.title('Prédictions vs Réel')
plt.show()

# Importance des features
importances = rf_regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_regressor.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Importance des caractéristiques")
plt.bar(range(X.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
```

### Hyperparamètres importants à ajuster pour la régression:
1. **`n_estimators`**: Le nombre d'arbres dans la forêt. Un nombre plus élevé peut améliorer la précision, mais augmente aussi le temps de calcul.
2. **`max_features`**: Le nombre maximal de caractéristiques à considérer pour chaque division d'arbre. Cela augmente la diversité des arbres et réduit la corrélation entre eux.
3. **`bootstrap`**: Si l'échantillonnage bootstrap est utilisé pour l'échantillon des données (par défaut, c'est `True`).
4. **`oob_score`**: Le score Out-Of-Bag, qui permet d'estimer la performance du modèle sans un ensemble de validation séparé.


<br>
<br>
<br>
<br>

## Affichage 

Pour tracer une régression, il est possible d'utiliser `plotly.express` ou `plotly.graph_objects` pour créer des visualisations interactives:
```python
import numpy as np
import pandas as pd
import plotly.express as px


# Si X est une matrice (n_samples, n_features): Utiliser une seule colonne pour le tracé
data = pd.DataFrame({'Feature': X[:, 0], 'Target': y})  # Prendre la première caractéristique

# Si X est un vecteur (n_samples):
data = pd.DataFrame({'Feature': X.flatten(), 'Target': y})

# Instanciation et entraînement du modèle de régression choisi
model = #RegressionModelFunction()
model.fit(X, y)

# Créer un espace de valeurs pour la prédiction
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_pred = model.predict_proba(X_range)[:, 1]  # Probabilités de la classe positive

# Tracer les données et la courbe de régression logistique avec Plotly
fig = px.scatter(data, x='Feature', y='Target', title='Régression Logistique', labels={'Target': 'Classe', 'Feature': 'Feature'})
fig.add_scatter(x=X_range.flatten(), y=y_pred, mode='lines', name='Courbe de régression', line=dict(color='red'))

# Afficher la figure
fig.show()
```