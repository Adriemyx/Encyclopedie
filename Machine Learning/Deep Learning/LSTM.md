## 🧠 Objectif : Pourquoi un LSTM ?

Les **réseaux de neurones récurrents (RNN)** sont faits pour traiter des **données séquentielles** (texte, audio, signaux, séries temporelles...). Ils **mémorisent des informations** d’une étape à l’autre.

Mais les RNN classiques ont un gros défaut : ils **oublient rapidement** (ils n’arrivent pas à gérer des dépendances longues). Le **LSTM** (Long Short-Term Memory) a été inventé pour **corriger ça**, grâce à une **mémoire contrôlée par des "portes"**.

---

## ⚙️ Structure d’un LSTM : ce qui entre, ce qui sort

À chaque **pas de temps `t`** (par exemple chaque point de ta séquence radar), l'unité LSTM reçoit :

* `xₜ` : l’entrée actuelle (ex: un vecteur \[amplitude, distance])
* `hₜ₋₁` : la **sortie précédente**
* `cₜ₋₁` : la **mémoire précédente** (c’est le cœur du système)

Elle produit :

* `hₜ` : la **nouvelle sortie** (qu’on passe à la couche suivante ou au pas suivant)
* `cₜ` : la **nouvelle mémoire interne**

---

## 🔑 1. Les **portes** de l’unité LSTM

Chaque porte est un petit réseau de neurones (souvent un produit matriciel + une activation `sigmoïde` ou `tanh`). Elle contrôle le **flux d'information** :

### 🧽 **Porte d’oubli** `fₜ`

Elle décide **quoi oublier** de l’ancienne mémoire `cₜ₋₁`.

```math
fₜ = sigmoid(W_f · [hₜ₋₁, xₜ] + b_f)
```

* Si `fₜ ≈ 1` → on garde l’info dans `cₜ₋₁`
* Si `fₜ ≈ 0` → on oublie cette info

---

### 🧰 **Porte d’entrée** `iₜ` + `ĉₜ`

Elle décide **quelle nouvelle info ajouter** à la mémoire.

```math
iₜ = sigmoid(W_i · [hₜ₋₁, xₜ] + b_i)
ĉₜ = tanh(W_c · [hₜ₋₁, xₜ] + b_c)
```

* `iₜ` dit **quelles dimensions sont mises à jour**
* `ĉₜ` est la **nouvelle info candidate**


#### 🔍 Ce que ça veut dire concrètement :

`iₜ` est un **vecteur de la même taille que `ĉₜ`** (la nouvelle "info candidate" à ajouter à la mémoire). Chacune de ses composantes (entre 0 et 1 grâce à la sigmoïde) agit comme un **interrupteur doux** :

* Si `iₜ[j]` est proche de **1**, cela veut dire :
  👉 **“Oui, on veut mettre à jour la dimension `j` de la mémoire avec `ĉₜ[j]`.”**

* Si `iₜ[j]` est proche de **0**, cela veut dire :
  👉 **“Non, on ne touche pas à cette dimension `j` de la mémoire.”**



#### 🧠 Exemple simple :

Imaginons que `ĉₜ = [0.4, -0.7, 0.2]` (la nouvelle info)
et que `iₜ = [1.0, 0.0, 0.5]` (le filtre "quoi ajouter").

Alors `iₜ * ĉₜ = [0.4, 0.0, 0.1]`

➡️ La **1ère dimension** est complètement ajoutée
➡️ La **2ème dimension** est ignorée
➡️ La **3ème dimension** est partiellement prise en compte



#### 📌 En résumé :

Quand on dit que `iₜ` "dit quelles dimensions sont mises à jour", ça signifie :

> Chaque élément de `iₜ` décide **dans quelle mesure** on ajoute la nouvelle information `ĉₜ` **dans chaque case** de la mémoire `cₜ`.

Tu peux imaginer que la mémoire a plein de petits tiroirs (une par dimension), et `iₜ` choisit **quels tiroirs ouvrir plus ou moins grand** pour y glisser la nouvelle info.


---

### 🧠 **Mise à jour de la mémoire** `cₜ`

On combine l’oubli et l’ajout :

```math
cₜ = fₜ * cₜ₋₁ + iₜ * ĉₜ
```

* On oublie une partie du passé (`fₜ * cₜ₋₁`)
* On ajoute une partie du présent (`iₜ * ĉₜ`)

---

### 📤 **Porte de sortie** `oₜ`

Elle contrôle ce qu’on sort à ce pas de temps :

```math
oₜ = sigmoid(W_o · [hₜ₋₁, xₜ] + b_o)
hₜ = oₜ * tanh(cₜ)
```

* `hₜ` est la sortie visible (transmise à la couche suivante)
* `cₜ` est stocké en mémoire pour l’étape suivante

---

## 🧾 En résumé : ce qui se passe dans une cellule LSTM à l’instant `t`

```text
Entrée: xₜ, hₜ₋₁, cₜ₋₁
Sortie: hₜ, cₜ

1. fₜ = sigmoid(W_f · [hₜ₋₁, xₜ])        ← quoi oublier ?
2. iₜ = sigmoid(W_i · [hₜ₋₁, xₜ])        ← quoi ajouter ?
3. ĉₜ = tanh(W_c · [hₜ₋₁, xₜ])           ← valeur à ajouter
4. cₜ = fₜ * cₜ₋₁ + iₜ * ĉₜ              ← mémoire mise à jour
5. oₜ = sigmoid(W_o · [hₜ₋₁, xₜ])        ← quoi sortir ?
6. hₜ = oₜ * tanh(cₜ)                    ← sortie

```

---

## 💡 Intuition d’ingénieur

On peut voir ça comme un **filtre adaptatif intelligent** :

* `cₜ` est une **bande passante mémoire** : elle conserve l’essentiel.
* Les portes `fₜ`, `iₜ`, `oₜ` sont des **circuits logiques flous** : elles apprennent à ouvrir/fermer le passage à l'information selon le contexte.
* Grâce à ça, le LSTM peut **retenir une info longtemps** (ex: une bosse dans le signal radar) ou **l’oublier vite** (ex: du bruit ou un pic isolé).

---

## 🔧 En pratique (PyTorch)

Tu n’as pas à coder tout ça à la main. PyTorch fournit une implémentation :

```python
lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=True)
```

Mais **comprendre ce qu’il se passe dans chaque cellule t’aide à mieux configurer le modèle**, à choisir le bon pooling, ou à interpréter les erreurs.

---

## 🚀 À quoi sert la bidirectionnalité ?

Une **LSTM bidirectionnelle** lit la séquence **dans les deux sens** :

* une LSTM “avant” lit de `t=0` à `t=T`
* une LSTM “arrière” lit de `t=T` à `t=0`

Elle capte donc :

* le **contexte passé** (`←`) comme une LSTM normale
* **et aussi le futur** (`→`) dans la séquence

### 👉 Exemple :

Dans une phrase :

> “Il a **glissé** sur une **peau de banane**.”

Pour prédire ou classer "**glissé**", savoir que "**peau de banane**" vient **après** peut aider énormément — mais une LSTM classique ne le sait pas encore. Une bidirectionnelle, oui.

---

## 📊 Est-ce que c’est toujours mieux ?

### ✅ **Oui**, si :

* Tu travailles sur une tâche où **tout le contexte est disponible en avance** (ex: classification de séquence, compréhension globale, NLP, séries temporelles **non causales**).
* Tu veux **plus de contexte global** pour prendre une décision à la fin (ex: sentiment global, détection d’anomalies sur fenêtre glissante...).

### ❌ **Non**, si :

* Tu fais de la **prédiction temps réel / séquentielle** (ex: prédire le futur en temps réel, traitement de flux, génération en ligne, etc.).
* Tu ne peux **pas utiliser d’info du futur** (c'est interdit dans le cadre métier, ex: finance en ligne, robotique, etc.).



Oui, et c’est un sujet passionnant ! 🎯 L’interprétabilité des modèles LSTM (et plus généralement des modèles séquentiels profonds) est **un défi**, mais il existe plusieurs **outils et méthodes** pour **comprendre ce que le modèle a appris** ou **pourquoi il prédit ce qu’il prédit**.

---

## 🔍 Outils & méthodes d’interprétabilité pour les LSTM

### 1. 🧠 **Attention (self-attention ou mécanisme externe)**

* Ajoute un **poids à chaque pas de temps** de la séquence.
* Tu peux **visualiser les poids d’attention** pour savoir **quels moments du signal ont influencé la prédiction**.

> **Idéal** pour les tâches où certains instants clés du signal portent plus d’info que d’autres (ex : anomalies, pics, motifs localisés).

#### 📦 Outils : implémenté manuellement, ou via modules comme `torch.nn.MultiheadAttention`

---

### 2. 📊 **LIME / SHAP (adaptés aux séquences)**

* Techniques **agnostiques** au modèle (boîte noire) qui mesurent l’effet de perturbations locales.
* Te disent **quels éléments de la séquence ont le plus pesé** dans la décision.

> ⚠️ Peut être coûteux en temps de calcul, et nécessite parfois d'adapter la granularité temporelle (par ex. segmenter la séquence en blocs).

#### 📦 Outils :

* `LIME`: [`lime.lime_tabular`](https://github.com/marcotcr/lime)
* `SHAP`: [`DeepExplainer` ou `GradientExplainer`](https://github.com/slundberg/shap)

---

### 3. 🔎 **Gradient-based methods (saliency, integrated gradients, etc.)**

* Calculent le **gradient de la sortie par rapport à l’entrée** : "si je bouge ce point du signal, est-ce que la sortie change beaucoup ?"
* Permettent de **visualiser les zones sensibles du signal**.

> Très utilisé en NLP et vision, **transposable aux séries temporelles**.

#### 📦 Outils :

* `captum` (lib PyTorch pour l’interprétabilité)
  → [https://github.com/pytorch/captum](https://github.com/pytorch/captum)
  → propose : saliency maps, integrated gradients, DeepLIFT, etc.

---

### 4. 🛠️ **Hidden state analysis**

* Inspecte manuellement les `hₜ` ou `cₜ` au fil du temps.
* Peut être visualisé comme un "électroencéphalogramme" du modèle.
* Si tu observes des pics ou des activations fortes à certains instants, cela **révèle que le modèle "réagit" à certaines zones du signal.**

#### 👉 Pratique :

```python
out, (hn, cn) = model.lstm(x)
# out: (batch, seq_len, hidden_size)
plt.plot(out[0].detach().cpu())  # Affiche l'évolution des activations
```

---

### 5. 📚 **Feature occlusion / masking**

* Masquer des morceaux du signal (par ex. remplacer par du bruit ou des zéros) et observer la variation de la prédiction.
* Cela montre **quelles zones sont critiques** pour la décision.

---

## 🧠 En résumé : quelle méthode choisir ?

| Objectif                                        | Méthode                    | Facilité            | Interprétation                |
| ----------------------------------------------- | -------------------------- | ------------------- | ----------------------------- |
| Voir **quand** le modèle s'active               | Attention                  | ✅ Facile à intégrer | 🌟 Très visuel                |
| Voir **quels éléments** impactent la prédiction | LIME / SHAP                | ⚠️ Plus lourd       | 🌟 Explicite                  |
| Voir **où le gradient est fort**                | Saliency / IG (Captum)     | ✅ Moyen             | 🔍 Précis mais parfois bruité |
| Voir **ce que le modèle "ressent"**             | Analyse des états internes | ✅ Facile            | 🔧 Diagnostic                 |
| Tester l’impact d’un bloc du signal             | Masquage / occlusion       | ✅ Simple            | 🧪 Empirique                  |

