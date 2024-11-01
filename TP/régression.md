<h1 align='center'> Machine learning - Régression </h1>

Dans ce document sera présenté quelques bases de code, notamment avec la libraire `scikit-learn`, pour faire de la régression sur python.

## Régression linéaire 
A utiliser dans le cas d'un problème **supervisé** avec un label **quantitatif** et dont la relation entre la cible et l'entrée (**une variable**) semble être linéaire $y = \alpha \times x + \beta$.

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Séparation 
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

1. **MSE (Mean Squared Error)** : C'est la moyenne des carrés des erreurs. Il mesure la dispersion des erreurs.

   ```python
   from sklearn.metrics import mean_squared_error
   mse = mean_squared_error(y_true, y_pred)
   ```

2. **RMSE (Root Mean Squared Error)** : C'est la racine carrée du MSE. Il est dans la même unité que la variable cible, ce qui le rend plus interprétable.

   ```python
   rmse = mean_squared_error(y_true, y_pred, squared=False)
   ```

3. **MAE (Mean Absolute Error)** : C'est la moyenne des erreurs absolues. Cela donne une idée de la taille des erreurs en unités de la variable cible.

   ```python
   from sklearn.metrics import mean_absolute_error
   mae = mean_absolute_error(y_true, y_pred)
   ```

4. **R² (Coefficient de détermination)** : Cette métrique indique la proportion de la variance dans la variable cible qui est prédit par le modèle. Elle varie entre 0 et 1, avec des valeurs plus élevées indiquant un meilleur ajustement.

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
Toutes les variables à disposition dans $\mathcal{X}$ ne sont pas forcément nécessaire pour la prédiction de la sortie. Il faut alors procéder à la sélection des variables. Il existe plusieurs méthodes:

#### Régression linéaire avec régularisation Lasso
#### Forward BIC