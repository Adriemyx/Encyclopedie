<h1 align='center'> Machine Learning - Fine tuning 🚀</h1>

## **I. Différence entre paramètres et hyperparamètres**

- **Paramètres**: Ce sont les valeurs apprises par le modèle pendant l'entraînement, comme les poids d'un réseau neuronal ou les coefficients d'une régression linéaire.  
  Exemple: $\mathbf{w}$ et $b$ dans $y = \mathbf{w}^T\mathbf{x} + b$.

- **Hyperparamètres**: Ce sont les paramètres définis avant l'entraînement et non appris directement à partir des données. Ils influencent la structure ou le comportement de l'apprentissage.  
  Exemple: le taux d'apprentissage, la profondeur des arbres, le nombre de neurones dans une couche cachée.

L'optimisation des hyperparamètres permet de:
- Réduire le **sous-ajustement** (modèle trop simple, faible performance).
- Réduire le **surajustement** (modèle trop complexe, mauvais sur les nouvelles données).
- Maximiser les performances sur un ensemble de validation ou en cross-validation.

<br>

## **II. Méthodes d’optimisation des hyperparamètres**

### **3.1. Recherche en grille (Grid Search)**

La recherche en grille explore toutes les combinaisons possibles d’hyperparamètres dans un espace défini.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Chargement des données
data = load_iris()
X, y = data.data, data.target

# Définir les hyperparamètres à explorer
param_grid = {
    'n_estimators': [10, 50, 100],  # Nombre d'arbres
    'max_depth': [3, 5, 10],        # Profondeur maximale
    'min_samples_split': [2, 5],    # Minimum d'échantillons pour diviser un nœud
}

# Création du modèle et de la recherche en grille
model = RandomForestClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

# Entraînement
grid_search.fit(X, y)

# Résultats
print("Meilleurs hyperparamètres:", grid_search.best_params_)
print("Meilleure précision:", grid_search.best_score_)
```

#### **Avantages**:
- Facile à comprendre et à implémenter.

#### **Inconvénients**:
- Coût computationnel élevé, surtout si l’espace d’hyperparamètres est grand.

---

### **3.2. Recherche aléatoire (Random Search)**

La recherche aléatoire explore des combinaisons aléatoires d’hyperparamètres dans l’espace défini.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(10, 200),  # Distribution uniforme pour le nombre d'arbres
    'max_depth': randint(3, 15),       # Distribution uniforme pour la profondeur
}

# Création du modèle et de la recherche aléatoire
random_search = RandomizedSearchCV(model, param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search.fit(X, y)

print("Meilleurs hyperparamètres:", random_search.best_params_)
print("Meilleure précision:", random_search.best_score_)
```

#### **Avantages**:
- Plus rapide que Grid Search.
- Utile si certains hyperparamètres sont moins importants.

#### **Inconvénients**:
- Ne garantit pas d’explorer toutes les bonnes combinaisons.

---

### **3.3. Optimisation bayésienne**

L’optimisation bayésienne construit un modèle probabiliste de la fonction d’objectif et choisit les hyperparamètres en fonction des résultats précédents.

**Exemple avec Optuna:**
```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

# Fonction d'optimisation
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    model = GradientBoostingClassifier(**params)
    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return score

# Exécution de l'optimisation
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Meilleurs hyperparamètres:", study.best_params)
print("Meilleure précision:", study.best_value)
```

#### **Avantages**:
- Plus efficace pour les espaces complexes.
- Explore les hyperparamètres en fonction des performances passées.

#### **Inconvénients**:
- Plus complexe à mettre en œuvre.

(cf. [processus_gaussien](supervised/processus_gaussien.md))

---

### **3.4. Algorithmes évolutionnaires**

Ces techniques, comme l’algorithme génétique, utilisent des principes d’évolution pour explorer l’espace des hyperparamètres.

Exemple: **TPOT** (Tool for Optimized Pipeline).
TPOT (**Tree-based Pipeline Optimization Tool**) est une bibliothèque Python qui utilise des algorithmes génétiques pour automatiser la sélection des modèles et l'optimisation des hyperparamètres. Voici comment utiliser TPOT pour un problème de classification:

```python
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Chargement des données
data = load_iris()
X, y = data.data, data.target

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création de l'instance TPOT
tpot = TPOTClassifier(
    generations=5,            # Nombre de générations pour l'évolution
    population_size=20,       # Taille de la population à chaque génération
    verbosity=2,              # Niveau de détails des logs
    random_state=42,          # Reproductibilité
    scoring='accuracy',       # Métrique d'évaluation
    cv=5                      # Validation croisée
)

# Entraînement
tpot.fit(X_train, y_train)

# Prédictions et évaluation
print(f"Score sur les données de test: {tpot.score(X_test, y_test):.4f}")

# Exporter le meilleur pipeline
tpot.export('best_pipeline.py')  # Sauvegarde le pipeline en Python
```

<br>

Voici comment utiliser TPOT pour un problème de régression:

```python
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Chargement des données
data = fetch_california_housing()
X, y = data.data, data.target

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création de l'instance TPOT
tpot = TPOTRegressor(
    generations=5,             # Nombre de générations pour l'évolution
    population_size=20,        # Taille de la population à chaque génération
    verbosity=2,               # Niveau de détails des logs
    random_state=42,           # Reproductibilité
    scoring='r2',              # Métrique d'évaluation
    cv=5                       # Validation croisée
)

# Entraînement
tpot.fit(X_train, y_train)

# Prédictions et évaluation
print(f"Score R² sur les données de test: {tpot.score(X_test, y_test):.4f}")

# Exporter le meilleur pipeline
tpot.export('best_pipeline_regression.py')
```



### **Interprétation des résultats**
1. **Generations et population_size** :
   - `generations` : Le nombre d'itérations pour l'optimisation.
   - `population_size` : Le nombre de pipelines différents testés par génération.
   
2. **Exportation du pipeline** :
   - Le fichier exporté (`best_pipeline.py`) contient le meilleur pipeline trouvé par TPOT. Vous pouvez l'utiliser sans réexécuter TPOT.

---

*<u>Remarques:</u>*
- **TPOT est gourmand en ressources** : Pour les grands ensembles de données ou de nombreuses générations, utilisez des machines puissantes ou réduisez les tailles de population/générations.
- Si votre temps est limité, vous pouvez :
  - Réduire `population_size` et `generations`.
  - Ajouter une contrainte de temps avec `max_time_mins` (temps maximal pour l'exécution totale).



<br>
<br>

### **III. Bonnes pratiques pour le fine-tuning**

#### **1. Définir un espace d’hyperparamètres pertinent**
- En se basant sur les **caractéristiques des données** et les **limites du modèle**.
- Exemple: Pour un arbre de décision, une très grande profondeur peut entraîner du surajustement.

#### **4.2. Utiliser la validation croisée**
- Toujours évaluer les performances sur un ensemble de validation pour éviter le surajustement.

#### **4.3. Surveiller le temps de calcul**
- Équilibrer la précision souhaitée avec le coût computationnel.

#### **4.4. Éviter les pièges courants**
- Ne pas tester trop de combinaisons sans justification.
- Ne pas optimiser sur l’ensemble de test (réserver un ensemble pour l’évaluation finale).

<br>
<br>

### **IV. Importance des hyperparamètres courants**

| Modèle                    | Hyperparamètres clés                               |
|---------------------------|---------------------------------------------------|
| Arbres de décision         | `max_depth`, `min_samples_split`, `min_samples_leaf` |
| Forêt aléatoire            | `n_estimators`, `max_depth`, `max_features`       |
| Gradient Boosting          | `n_estimators`, `learning_rate`, `subsample`      |
| Réseaux neuronaux          | `learning_rate`, `batch_size`, `epochs`, `layers` |
| KNN                        | `n_neighbors`, `metric`                           |


<br>
<br>


### **V. Structure, rétrécissement et randomisation**
Pour contrôler le **surajustement** dans les arbres à gradient boosté, trois principaux leviers peuvent être utilisés: la **structure de l'arbre**, le **rétrécissement** (learning rate), et la **randomisation** (sous-échantillonnage des données ou des features).


#### **1. Contrôler la structure de l’arbre**
La structure de l’arbre détermine la complexité des modèles individuels (arbres). Une complexité excessive peut entraîner un surajustement.

- **`max_depth`**: Limite la profondeur des arbres pour réduire leur complexité.
- **`min_child_weight`**: Requiert un nombre minimum d’échantillons dans une feuille pour qu’elle soit créée.
- **`gamma`**: Représente le gain minimum requis pour diviser un noeud. Plus il est grand, plus le modèle sera régulier.

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Chargement des données California Housing
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Contrôle de la structure de l'arbre
params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,                # Limite la profondeur des arbres
    'min_child_weight': 10,        # Minimum d'échantillons par feuille
    'gamma': 0.2,                  # Gain minimal pour diviser un nœud
    'eta': 0.1,                    # Taux d'apprentissage (rétrécissement)
    'subsample': 0.8,              # Randomisation (voir plus bas)
    'colsample_bytree': 0.8        # Randomisation (voir plus bas)
}

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], early_stopping_rounds=10)

# Prédictions
y_pred = model.predict(dtest)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.2f}")
```

<br>

#### **2. Rétrécissement (Learning Rate ou `eta`)**
Le **learning rate** réduit l’impact de chaque arbre individuel sur le modèle final. Cela empêche les ajustements excessifs à une itération donnée. Cependant, cela nécessite généralement un plus grand nombre d’arbres pour converger.

- **`eta`**: Contrôle la contribution de chaque arbre (valeurs typiques: 0.01 à 0.3).

```python
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.05,                  # Learning rate faible pour plus de régularisation
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Entraîner avec un taux d'apprentissage faible
model = xgb.train(params, dtrain, num_boost_round=300, evals=[(dtest, 'test')], early_stopping_rounds=10)

# Prédictions
y_pred = model.predict(dtest)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE avec learning rate faible: {rmse:.2f}")
```

<br>

#### **3. Randomisation (Sous-échantillonnage)**
La randomisation ajoute de la diversité entre les arbres, réduisant le risque de surajustement tout en maintenant une bonne performance globale.

- **`subsample`**: Fraction des données utilisées pour chaque arbre.
- **`colsample_bytree`**: Fraction des features utilisées pour construire chaque arbre.
- **`colsample_bylevel`**: Fraction des features à chaque niveau de l’arbre.
- **`colsample_bynode`**: Fraction des features à chaque noeud.

```python
params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'eta': 0.1,
    'subsample': 0.7,             # Utiliser 70% des échantillons par arbre
    'colsample_bytree': 0.6,      # Utiliser 60% des features par arbre
}

# Entraîner avec randomisation
model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], early_stopping_rounds=10)

# Prédictions
y_pred = model.predict(dtest)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE avec randomisation: {rmse:.2f}")
```

---

#### **Résumé: Combiner les 3 leviers**
Pour un contrôle optimal du surajustement, combiner ces trois techniques:
- Ajuster la **structure de l’arbre** pour éviter des modèles trop complexes.
- Réduiser l’impact de chaque arbre avec un **learning rate faible**.
- Utiliser la **randomisation** pour introduire de la diversité entre les arbres.

#### **Exemple combiné:**
```python
params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'min_child_weight': 5,
    'gamma': 0.3,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,                # Régularisation L2 pour davantage de contrôle
    'alpha': 0.5                  # Régularisation L1
}

model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, 'test')], early_stopping_rounds=15)

# Évaluation
y_pred = model.predict(dtest)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE final: {rmse:.2f}")
```