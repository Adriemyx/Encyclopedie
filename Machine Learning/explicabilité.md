<h1 align='center'> Machine learning - Explicabilité 👨‍🏫</h1>

L’explicabilité ou l’interprétabilité de l’IA vise à rendre le fonctionnement et les résultats des modèles davantage intelligibles et transparents pour les humains, sans pour autant faire de compromis sur les performances. 



## LIME
La méthode *LIME* consiste à approcher localement un classifieur "boite noire" par un classifieur plus simple et surtout interprétable (dont les prises de décisions s’appuient sur des éléments compréhensible pour un humain).   

LIME consiste donc à trouver un classifieur intérptétable $g \in \mathcal{G}$ donnant une approximation locale d’un classifieur $f$ non interprétable. Ici $G$ désigne l’ensemble des modèles interpétables (ex: modèles linéaires ou arbres de décision). On suppose que $f$ s’applique à des tenseurs $x \in \mathcal{R}^d$ (texte, image, …) de dimension $d$. $g$ prend en entrée un vecteur binaire $x' = \{0,1\}^{d'}$, de dimension $d^′$, indicant la présence ou l’absence d’un élément constitutif et interprétable d’une entrée de $f$. Dans le cas où $x$ est une image, de dimension $d=height \times width \times channels$ découpée en $d^′$ **super-pixels**, $x^′$ indique quels **super-pixels** sont fournis en entrée de $g$. Trouver $g$ passe alors par la résolution d’un problème minimilisation sous contrainte:
$g= argmin_{g \in \mathcal{G}}L(f, g, \pi x)+\Omega(g)$, avec $\Omega$
une fonction croissante de la complexité de $g$, $L$ une *loss* pénalisant les écarts de prédiction entre $f$ et $g$, et $\pi x$ un voisinage autour de $x$ réglant la pertinence du modèle interprétable par rapport à $f$.

### Pour du texte

```python
import pandas as pd
import re
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from lime.lime_text import LimeTextExplainer

# Data loading
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Data cleaning
df_train.columns = map(str.lower, df_train.columns)
df_test.columns = map(str.lower, df_test.columns)

df_train = df_train.rename(columns={'class index': 'target'})
df_test = df_test.rename(columns={'class index': 'target'})


#TF-IDF
tfidf_vc = TfidfVectorizer(
    min_df = 10,
    max_features = 100000,
    analyzer = "word",
    ngram_range = (1, 2),
    stop_words = 'english',
    lowercase = True
)

# Logistic Regression
model = LogisticRegression(C = 0.5, solver = "sag")

# Pipeline definition
pipe = make_pipeline(tfidf_vc, model)

# Pipeline training
pipe.fit(df_train["description"], df_train.target)

# Predictions on test_set
test_pred = pipe.predict(df_test["description"])

# Evaluation
print(classification_report(df_test.target, test_pred))
print(confusion_matrix(df_test.target, test_pred))


# Explicability
idx = df_test.index[0]

class_names = [] # ex: ["World", "Sports", "Business", "Sci/Tech"]
explainer = LimeTextExplainer(class_names = class_names)
exp = explainer.explain_instance(
    df_test["description"][idx],
    pipe.predict_proba,
    num_features = 10,
    top_labels=3
)

exp.show_in_notebook(text=df_test["description"][idx])
```

<br>

### Pour une image
<p align="center">
<img src=https://www.oreilly.com/content/wp-content/uploads/sites/2/2019/06/figure3-2cea505fe733a4713eeff3b90f696507.jpg width=500>
</p>
<p align="center">
<img src=https://www.oreilly.com/content/wp-content/uploads/sites/2/2019/06/figure4-99d9ea184dd35876e0dbae81f6fce038.jpg width=500>
</p>

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from keras.applications import inception_v3 as inc_net

# Load model
inception_model = InceptionV3(weights='imagenet')

# Image processing
def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

path_to_img = "..."
images = transform_img_fn([path_to_img])

# display the image
plt.imshow(images[0] / 2 + 0.5)
plt.axis('off')
plt.show()

#
#
# Make some predictions
# decode the results into a list of tuples (class, description, probability)
preds = inception_model.predict(images)
for x in decode_predictions(preds)[0]:
    print(x)


#
#
# Explicability
#  Train lime image explainer
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(images[0].astype('double'), inception_model.predict, top_labels=5, hide_color=0, num_samples=200)

# Plot boundaries
selected_label = 2 # From the predictions
temp, mask = explanation.get_image_and_mask(explanation.top_labels[selected_label], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

# Plot boundaries on the full image
temp, mask = explanation.get_image_and_mask(explanation.top_labels[selected_label], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

# Select the same class explained on the figures above.
ind =  explanation.top_labels[selected_label]

# Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

#Plot the heatmap
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
plt.show()

```


<br>
<br>

## SHAP
La méthode *SHAP* (**SH**apley **A**dditive ex**P**lanations) est inspirée de la **théorie des jeux**. En effet, dans la théorie des jeux, la [valeur de Shapley](https://en.wikipedia.org/wiki/Shapley_value) (1953) est un concept de solution qui consiste à répartir équitablement les gains et les coûts entre plusieurs acteurs travaillant en coalition.   
La valeur de Shapley s'applique principalement aux situations où les contributions de chaque acteur sont inégales, mais où ils travaillent en coopération les uns avec les autres pour obtenir un gain.


Il faut commencer par identifier la contribution de chaque joueur lorsqu'il joue individuellement, lorsque 2 joueurs jouent ensemble, et lorsque les 3 joueurs jouent ensemble:
<p align="center">
<img src=https://clearcode.cc/app/uploads/2016/11/ABC-wide.png width=500>
</p>

Ensuite, il faut considérer tous les ordres possibles et calculer leur valeur marginale - par exemple, quelle est la valeur ajoutée par chaque joueur lorsque le joueur A entre dans le jeu en premier, suivi par le joueur B, puis par le joueur C:
<p align="center">
<img src=https://clearcode.cc/app/uploads/2016/11/ABC-updated.png width=500>
</p>

Maintenant que  la valeur marginale de chaque joueur pour les 6 combinaisons d'ordre possibles a été calculé, il faut les additionner et calculer la valeur de Shapley (c'est-à-dire la moyenne) pour chaque joueur. Exemple pour le joueur A:   
$\text{Shapley}_{A} = \frac{7+7+10+3+9+10}{6} \approx 7,7$

Le calcul de la valeur de Shapley pour chaque joueur permet de **connaître la contribution réelle de chaque joueur au jeu** et d'attribuer les crédits de manière équitable.

<br>

Pour appliquer la méthode SHAP à une méthode d'explicabilité, il faut alors considérer que chaque valeur d'une variable indépendante ou d'une caractéristique pour un échantillon donné fait partie d'un jeu coopératif dans lequel nous supposons que la prédiction est en fait le gain.   
Les valeurs de Shapley correspondent alors à la contribution de chaque caractéristique à l'éloignement de la prédiction de la valeur attendue.


### Avantages
* SHAP repose sur une base théorique solide en matière de théorie des jeux. La prédiction est équitablement répartie entre les valeurs des caractéristiques. Nous obtenons des explications contrastives qui comparent la prédiction à la prédiction moyenne.
* SHAP dispose d'une implémentation rapide pour les modèles basés sur les arbres.
* Lorsque le calcul des nombreuses valeurs de Shapley est possible, des interprétations globales du modèle peuvent être élaborées. Les méthodes d'interprétation globale comprennent l'importance des caractéristiques, la dépendance des caractéristiques, les interactions, le regroupement et les diagrammes de synthèse.
* La librairie SHAP de Python offre de très bonnes visualisations.

### Inconvénients
* Calcul lent pour de nombreuses instances (sauf pour les modèles basés sur les arbres).
* Les inconvénients des valeurs de Shapley s'appliquent également à SHAP: Les valeurs de Shapley peuvent être mal interprétées.
* Les inconvénients des valeurs de Shapley s'appliquent également à SHAP: les valeurs de Shapley peuvent être mal interprétées.

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

# create train/validation split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=10)
dt = xgb.DMatrix(X_train, label=y_train.values)
dv = xgb.DMatrix(X_test, label=y_test.values)

# Solve a logistic regression with a logloss evaluation
params = {
    "eta": 0.5,
    "max_depth": 4,
    "objective": 'binary:logistic',
    "silent": 1,
    "base_score": np.mean(y_train),
    "eval_metric": 'logloss'
}
# Code the training part for 300 iterations with early stopping rounds at 5 and a verbose eval at 25
model = xgb.train(params, dt, 200, [(dt, "train"),(dv, "valid")], early_stopping_rounds=5, verbose_eval=25)

# compute the SHAP values for every prediction in the validation dataset
explainer = shap.TreeExplainer(model)
explainer_X_test = explainer(X_test)

shap_values = explainer_X_test.values
base_values = explainer_X_test.base_values


exp = shap.Explanation(shap_values, base_values, data=X_test.values, feature_names=X_test.columns)

idx=0

shap.plots.waterfall(exp[idx])

# Force plot example for a record
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:])

# Global explainability
shap.plots.bar(explainer(X_test))

# Local explanation summary
shap.summary_plot(shap_values, X_test)

# Dependence plot between variables (automatic)
shap.dependence_plot({variable}, shap_values, X_test)

# Dependence plot between variables (assigned)
shap.dependence_plot({variable}, shap_values, X_test, alpha=0.2, interaction_index={variable})

# Sort the features indexes by their importance in the model
# (sum of SHAP value magnitudes over the validation dataset)
top_inds = np.argsort(-np.sum(np.abs(shap_values), 0))

# Make SHAP plots of the three most important features
for i in range(3):
    shap.dependence_plot(top_inds[i], shap_values, X_test)


# Play with plot variables
shap.dependence_plot(top_inds[9], shap_values, X_test, x_jitter=10, alpha=0.8, dot_size=5)
```