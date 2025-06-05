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
