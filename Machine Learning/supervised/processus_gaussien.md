<h1 align='center'> Machine learning - Processus Gaussiens 🧮 </h1>

Les **processus gaussiens (GP)** sont une généralisation infinie des distributions gaussiennes pour des ensembles de données continus.
Ils sont principalement utilisés en régression et en optimisation de fonctions. 
Un GP définit une distribution sur des fonctions, permettant de faire des prédictions probabilistes dans des problèmes non paramétriques.

Un processus gaussien est spécifié par:
- Une fonction moyenne: $\mu(x)$, qui donne l'espérance des sorties en un point.
- Une fonction de covariance (ou noyau): $k(x, x')$, qui encode la dépendance entre les points $x$ et $x'$.

Formellement: $f(x) \sim \mathcal{GP}(\mu(x), k(x, x'))$
<br>

## **I. Principes Fondamentaux**

### 1. **Définition:** Tout ensemble de points $\{x_1, ..., x_n\}$ tiré d'un processus gaussien suit une distribution multivariée gaussienne: $f(x) = \mathcal{N}(\mu, K)$
où:
   - $\mu$ est un vecteur de moyennes,
   - $K$ est la matrice de covariance définie par le noyau.

### 2. **Noyaux Courants:**
   - **RBF (Radial Basis Function)** ou noyau gaussien:
   $k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$
   - **Linéaire:**
   $k(x, x') = x^\top x'$
   - **Périodique:**
   $k(x, x') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi |x - x'|/p)}{\ell^2}\right)$

### 3. **Régression Gaussienne:**
   Étant donné des observations $X = \{x_1, ..., x_n\}$ et leurs sorties $y = \{y_1, ..., y_n\}$, 
   le GP utilise les relations probabilistes pour prédire les valeurs en des points non observés $X^*$.

<br>

## **II. Régression Gaussienne**

Soit:
- Les observations: $\mathbf{y} \sim \mathcal{N}(\mathbf{\mu}, \mathbf{K} + \sigma^2 \mathbf{I})$,
- Les points de prédiction: $\mathbf{f}^{*} \sim \mathcal{N}(\mu, K_{*})$.

Les prédictions sont données par: $\mathbf{f}^{*} \mid \mathbf{X}, \mathbf{y}, \mathbf{X}^{*} \sim \mathcal{N}(\mathbf{\mu}^{*}, \mathbf{\Sigma}^{*})$
où:
- Moyenne prédictive: $\mu_* = K(X_*, X) [K(X, X) + \sigma^2 I]^{-1} y$
- Covariance prédictive: $\mathbf{\Sigma}^{*} = K(X_{*}, X_{*}) - K(X_{*}, X) \big[K(X, X) + \sigma^2 I\big]^{-1} K(X, X_{*})$

---

Voici une implémentation simple d'une régression gaussienne avec un noyau RBF.

```python
# Importation des Librairies
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky, solve


# Définition du Noyau RBF
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    """
    Noyau RBF (Radial Basis Function).
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# Régression Gaussienne
class GaussianProcessRegressor:
    def __init__(self, kernel, noise=1e-5):
        self.kernel = kernel
        self.noise = noise

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        K = self.kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        self.L = cholesky(K, lower=True)
        self.alpha = solve(self.L.T, solve(self.L, y_train))

    def predict(self, X_test):
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test) + 1e-8 * np.eye(len(X_test))
        mean = K_s.T @ self.alpha
        v = solve(self.L, K_s)
        covariance = K_ss - v.T @ v
        return mean, covariance


# Exemple d'Utilisation:

# Données simulées
X_train = np.array([[-4.0], [-3.0], [-2.0], [0.0], [2.0]])
y_train = np.sin(X_train).ravel()

X_test = np.linspace(-5, 5, 100).reshape(-1, 1)

# Modèle GP
gp = GaussianProcessRegressor(kernel=lambda X1, X2: rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0))
gp.fit(X_train, y_train)

# Prédiction
mean, cov = gp.predict(X_test)

# Intervalle de confiance
std_dev = np.sqrt(np.diag(cov))
lower_bound = mean - 1.96 * std_dev
upper_bound = mean + 1.96 * std_dev

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'ro', label="Données d'entraînement")
plt.plot(X_test, mean, 'b-', label="Prédiction moyenne")
plt.fill_between(X_test.ravel(), lower_bound, upper_bound, color='blue', alpha=0.2, label="95% IC")
plt.legend()
plt.show()
```

<br>

## **III. Applications des Processus Gaussiens**

1. **Régression Non-Paramétrique:** Les GP offrent une méthode flexible pour modéliser des fonctions complexes.
2. **Optimisation Bayésienne:** Utilisés pour optimiser des fonctions coûteuses (e.g., ajustement d'hyperparamètres en ML).
3. **Modèles Spatiaux:** Appropriés pour les problèmes où les données dépendent fortement de leur position géographique.

<br>
<br>

## **IV. Optimisation Bayésienne**
L’**optimisation bayésienne** est une méthode puissante pour optimiser des fonctions coûteuses à évaluer. 
Elle est couramment utilisée en machine learning, notamment pour l’ajustement des hyperparamètres. 
Contrairement aux méthodes classiques comme la descente de gradient, elle convient pour des fonctions non convexes, non dérivables et bruitées.

Le principe central repose sur deux composants principaux:
1. **Modèle probabiliste (ex.: Processus Gaussien)**: Il modélise la fonction cible.
2. **Fonction d’acquisition**: Elle guide le choix des points à évaluer.

<br>

### **1. Processus de l’Optimisation Bayésienne**

1. **Définir la fonction cible** $f(x)$, souvent coûteuse à évaluer.
2. **Construire un modèle probabiliste** (souvent un GP) pour approximer $f(x)$.
3. **Optimiser la fonction d’acquisition** (comme l’*Expected Improvement* ou l’*UCB*) pour sélectionner le prochain point à évaluer.
4. **Évaluer $f(x)$ au nouveau point** et mettre à jour le modèle.
5. **Répéter** jusqu’à convergence ou jusqu’à atteindre un budget donné (nombre maximal d’évaluations).

<br>

### **2. Fonction d’Acquisition**

Les fonctions d’acquisition équilibrent l’**exploitation** (explorer les points où le modèle prédit de bonnes performances) 
et l’**exploration** (explorer des zones incertaines).

- **Expected Improvement (EI)**:
$EI(x) = \mathbb{E}[\max(0, f(x) - f_{\text{best}})]$
  Où $f_{\text{best}}$ est la meilleure valeur observée jusqu’à présent.

- **Upper Confidence Bound (UCB)**:
$UCB(x) = \mu(x) + \kappa \sigma(x)$
  Où $\mu(x)$ est la moyenne prédite, $\sigma(x)$ est l’écart-type prédite, et $\kappa$ contrôle l’exploration.

<br>

### **3. Implémentation**
#### **a. Minimisation d'une fonction mathématique**
Voici une implémentation simple de l’optimisation bayésienne pour minimiser une fonction $f(x)$:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Définir la Fonction Cible (Coûteuse à Évaluer):

# Fonction cible à minimiser (exemple: sin(x) avec un minimum global)
def objective_function(x):
    return np.sin(3 * x) + 0.5 * x**2 + 0.2 * x


#Noyau RBF
def rbf_kernel(X1, X2, length_scale=1.0, sigma_f=1.0):
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# Régression Gaussienne
class GaussianProcess:
    def __init__(self, kernel, noise=1e-6):
        self.kernel = kernel
        self.noise = noise

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        K = self.kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, y_train))

    def predict(self, X_test):
        K_s = self.kernel(self.X_train, X_test)
        K_ss = self.kernel(X_test, X_test) + 1e-8 * np.eye(len(X_test))
        mean = K_s.T @ self.alpha
        v = np.linalg.solve(self.L, K_s)
        covariance = K_ss - v.T @ v
        return mean.ravel(), np.sqrt(np.diag(covariance))
        
# Fonction d’Acquisition (Expected Improvement)
def expected_improvement(X, gp, y_max, xi=0.01):
    mu, sigma = gp.predict(X)
    sigma = sigma.reshape(-1, 1)
    with np.errstate(divide='ignore'):
        Z = (mu - y_max - xi) / sigma
        ei = (mu - y_max - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei.ravel()
    
# Optimisation Bayésienne
def bayesian_optimization(objective, bounds, n_iters=10, gp_params={}):
    # Initialisation
    X_train = np.random.uniform(bounds[0], bounds[1], size=(5, 1))
    y_train = np.array([objective(x) for x in X_train]).ravel()
    
    # Modèle GP
    gp = GaussianProcess(kernel=lambda X1, X2: rbf_kernel(X1, X2, **gp_params))
    gp.fit(X_train, y_train)
    
    for _ in range(n_iters):
        # Maximiser la fonction d'acquisition
        def acquisition(X):
            return -expected_improvement(X.reshape(-1, 1), gp, y_train.max())
        
        res = minimize(acquisition, x0=np.random.uniform(bounds[0], bounds[1]), bounds=[bounds])
        x_next = res.x
        
        # Évaluer la fonction cible
        y_next = objective(x_next)
        
        # Ajouter la nouvelle observation
        X_train = np.vstack((X_train, [[x_next]]))
        y_train = np.append(y_train, y_next)
        
        # Mettre à jour le modèle GP
        gp.fit(X_train, y_train)
        
    return X_train, y_train


# Exemple d’Application
from scipy.stats import norm

# Optimiser la fonction cible
bounds = (-2, 2)
X_train, y_train = bayesian_optimization(objective_function, bounds, n_iters=15, gp_params={"length_scale": 0.5, "sigma_f": 1.0})

# Visualiser les résultats
X_test = np.linspace(bounds[0], bounds[1], 100).reshape(-1, 1)
y_test = objective_function(X_test)

plt.figure(figsize=(12, 6))
plt.plot(X_test, y_test, 'r--', label="Fonction cible")
plt.plot(X_train, y_train, 'bo', label="Échantillons")
plt.legend()
plt.title("Optimisation Bayésienne")
plt.show()
```

<br>

#### **b. Optimisation des hyperparamètres**
Voici un exemple où l’optimisation bayésienne est utilisée pour ajuster les hyperparamètres d’un modèle machine learning, comme une **régression Ridge**.   
Exemple: Optimisation de l'Hyperparamètre $\alpha$ d'une Régression Ridge

```python
# Importation des Librairies
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
import numpy as np

# Fonction Cible: Validation Croisée

# Données simulées
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Fonction cible: erreur quadratique moyenne en validation croisée
def objective_function(alpha):
    model = Ridge(alpha=alpha)
    # Validation croisée (Moyenne des erreurs négatives pour avoir une "perte positive")
    score = -np.mean(cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error'))
    return score


# Application de l’Optimisation Bayésienne
bounds = (1e-3, 10)  # Recherche d'alpha entre 0.001 et 10
X_train, y_train = bayesian_optimization(objective_function, bounds, n_iters=20, gp_params={"length_scale": 0.5, "sigma_f": 1.0})

# Meilleur hyperparamètre trouvé
best_alpha = X_train[np.argmin(y_train)]
print(f"Meilleur alpha trouvé: {best_alpha}")

# Visualisation des Résultats
plt.figure(figsize=(10, 6))
plt.plot(X_train, y_train, 'bo-', label="Erreurs observées")
plt.xlabel("Valeurs d'alpha")
plt.ylabel("Erreur quadratique moyenne (MSE)")
plt.title("Optimisation Bayésienne des Hyperparamètres")
plt.legend()
plt.show()
```

En utilisant l’optimisation bayésienne, la méthode explore intelligemment l’espace des hyperparamètres pour trouver la meilleure valeur de $\alpha$ 
pour la régression Ridge, tout en minimisant le nombre d’évaluations coûteuses via la validation croisée.


<br>
<br>

## **V. Applications de l’Optimisation Bayésienne**

1. **Ajustement d’Hyperparamètres:**
   - Recherche d’hyperparamètres dans les modèles de machine learning (e.g., SVM, réseaux de neurones).
2. **Conception d’Expériences:**
   - Optimiser les paramètres dans des expériences physiques ou biologiques.
3. **Optimisation de Fonctions Coûteuses:**
   - Exemples: simulations, algorithmes d’IA, tests logiciels.
