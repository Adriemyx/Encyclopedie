## Réduction de dimension

### PCA
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
<br>

## Traitement des données textuelles
Le traitement des données textuelles est une étape cruciale dans l'analyse de texte, souvent utilisée dans des applications telles que l'analyse de sentiments, la classification de documents, et l'extraction d'informations. Voici quelques techniques couramment utilisées.


### CountVectorizer
`CountVectorizer` est une classe de la bibliothèque `scikit-learn` qui convertit une collection de documents en une matrice de compte de mots. Cela permet de représenter le texte sous forme numérique, facilitant ainsi l'analyse.

#### Fonctionnement:

- **Tokenisation** : Divise le texte en mots (ou tokens).
- **Fréquence** : Compte le nombre d'occurrences de chaque mot dans chaque document.

```python
from sklearn.feature_extraction.text import CountVectorizer

# Exemple de document
documents = [
    "Le chat est sur le tapis.",
    "Le chien est dans le jardin.",
    "Le chat et le chien jouent ensemble."
]

# Initialisation de CountVectorizer
vectorizer = CountVectorizer()

# Transformation des documents en matrice de comptes
X = vectorizer.fit_transform(documents)

# Affichage de la matrice de comptes
print("Matrice de comptes :")
print(X.toarray())
print("Mots : ", vectorizer.get_feature_names_out())
```


### TfidfTransformer
`TfidfTransformer` est une classe de `scikit-learn` qui transforme la matrice de comptes en une matrice de poids TF-IDF (Term Frequency-Inverse Document Frequency). Cette méthode permet de réduire l'importance des mots fréquents dans tous les documents et de mettre en valeur les mots spécifiques à chaque document.

#### Fonctionnement:

- **TF (Term Frequency)** : Fréquence d'un mot dans un document.
- **IDF (Inverse Document Frequency)** : Mesure de l'importance d'un mot dans le corpus.


```python
from sklearn.feature_extraction.text import TfidfTransformer

# Création de la matrice de comptes
count_vectorizer = CountVectorizer()
X_counts = count_vectorizer.fit_transform(documents)

# Initialisation de TfidfTransformer
tfidf_transformer = TfidfTransformer()

# Transformation en matrice TF-IDF
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Affichage de la matrice TF-IDF
print("Matrice TF-IDF :")
print(X_tfidf.toarray())
```




## NLTK

`nltk` (Natural Language Toolkit) est une bibliothèque Python très utilisée pour le traitement du langage naturel (NLP). Elle offre des outils pour la tokenisation, la lemmatisation (réduction des mots à leur forme de base (par exemple, "manges" à "manger")), l'analyse de sentiments (évaluation des émotions exprimées dans le texte), la classification, l'analyse syntaxique, et bien plus.

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Téléchargement des ressources nécessaires
nltk.download('punkt')
nltk.download('wordnet')

# Exemple de texte
text = "Les chats mangent des poissons et les chiens jouent dans le jardin."

# Tokenisation
tokens = word_tokenize(text)
print("Tokens :", tokens)

# Lemmatisation
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print("Lemmatisés :", lemmatized_tokens)
```


#### Étapes de Nettoyage des Données
Avant d'appliquer des techniques de traitement, il est souvent nécessaire de nettoyer les données textuelles (suppression des stop words, ponctuation, etc.).

1. **Suppression de la Ponctuation**
2. **Conversion en Minuscules**
3. **Suppression des Stop Words**
4. **Lemmatisation ou Stemming**
5. **Suppression des Caractères Non Alphanumériques**

Voici un exemple complet qui intègre ces étapes en utilisant `nltk` et `re` (module pour les expressions régulières) en Python.


```python
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Télécharger les ressources nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Exemple de texte
text = "Les chats, mangent des poissons ! Les chiens jouent dans le jardin, et ils sont très heureux."

# Étape 1 : Suppression de la Ponctuation
text = re.sub(r'[^\w\s]', '', text)
print("Après suppression de la ponctuation :", text)

# Étape 2 : Conversion en Minuscules
text = text.lower()
print("Après conversion en minuscules :", text)

# Étape 3 : Suppression des Stop Words
stop_words = set(stopwords.words('french'))
tokens = word_tokenize(text)
filtered_tokens = [word for word in tokens if word not in stop_words]
print("Après suppression des stop words :", filtered_tokens)

# Étape 4 : Lemmatisation
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print("Après lemmatisation :", lemmatized_tokens)

# Étape 5 : Suppression des Caractères Non Alphanumériques (déjà fait dans l'étape 1)

# Affichage final
print("Texte nettoyé final :", lemmatized_tokens)
```



<br>
<br>

## Sélection des variables
Toutes les variables à disposition dans $\mathcal{X}$ ne sont pas forcément nécessaire pour la prédiction de la sortie. Il faut alors procéder à la sélection des variables. Il existe plusieurs méthodes:

### Forward BIC
La méthode **Forward BIC** (Bayesian Information Criterion) est une technique qui vise à sélectionner un sous-ensemble de variables qui contribue le mieux à la prédiction de la variable cible, tout en minimisant la complexité du modèle. Le BIC est un critère qui pénalise la complexité du modèle (nombre de paramètres) tout en tenant compte de la qualité de l'ajustement. Il aide à éviter le surapprentissage en ajoutant une pénalité plus forte pour les modèles complexes par rapport à d'autres critères comme l'AIC (Akaike Information Criterion).

Voici le code d'implémentation de la méthode forward BIC:
```python
import numpy as np
import statsmodels.api as sm

# Fonction pour calculer le BIC
def calculate_bic(y_true, y_pred, num_params):
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    n = len(y_true)
    return n * np.log(residual_sum_of_squares / n) + num_params * np.log(n)

# Fonction de Forward BIC
def forward_bic(X, y):
    remaining_features = list(X.columns)
    selected_features = []
    best_bic = float('inf')

    while remaining_features:
        bic_values = []
        
        # Essayer d'ajouter chaque variable restante
        for feature in remaining_features:
            current_features = selected_features + [feature]
            X_current = sm.add_constant(X[current_features])
            model = sm.OLS(y, X_current).fit()
            bic = calculate_bic(y, model.predict(), len(current_features))
            bic_values.append(bic)
        
        # Trouver la variable qui minimise le BIC
        best_bic_index = np.argmin(bic_values)
        best_bic_value = bic_values[best_bic_index]
        
        if best_bic_value < best_bic:
            best_bic = best_bic_value
            selected_features.append(remaining_features[best_bic_index])
            remaining_features.remove(remaining_features[best_bic_index])
        else:
            break  # Arrêter si le BIC ne s'améliore pas

    return selected_features

# Exécution de la sélection Forward BIC
selected_features = forward_bic(X, y)
print(f"Variables sélectionnées: {selected_features}")
```




### Backward Elimination
Cette méthode commence avec un modèle contenant toutes les variables et procède en supprimant progressivement les variables les moins significatives, basées sur des critères comme le p-value ou le BIC. Elle continue jusqu'à ce qu'aucune variable ne puisse être éliminée sans dégrader la performance du modèle.

Voici le code d'implémentation de la méthode backward elimination:
```python
import numpy as np
import statsmodels.api as sm

# Fonction pour calculer le BIC
def calculate_bic(y_true, y_pred, num_params):
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    n = len(y_true)
    return n * np.log(residual_sum_of_squares / n) + num_params * np.log(n)

# Fonction de Backward Elimination
def backward_elimination(X, y):
    features = list(X.columns)
    best_bic = float('inf')

    while features:
        X_current = sm.add_constant(X[features])
        model = sm.OLS(y, X_current).fit()
        
        # Calculer le BIC pour le modèle actuel
        bic = calculate_bic(y, model.predict(), len(features))
        
        # Trouver la variable avec la plus haute p-value
        p_values = model.pvalues.iloc[1:]  # Ignorer l'interception
        max_p_value = p_values.max()
        
        if max_p_value > 0.05:  # Seuil de signification
            feature_to_remove = p_values.idxmax()
            features.remove(feature_to_remove)
        else:
            break  # Arrêter si toutes les p-values sont en dessous du seuil

    return features

# Exécution de la sélection Backward Elimination
selected_features = backward_elimination(X, y)
print(f"Variables sélectionnées: {selected_features}")
```
