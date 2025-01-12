<h1 align='center'> Reinforcement learning - Chapitre 1 : Modélisation de problèmes de décision séquentielle avec des processus de décision de Markov </h1>

L'idée de ce chapitre est de développer une théorie générale pour décrire des problèmes qui contiennent:
- un ensemble d'états $\mathcal{S}$ décrivant le système à contrôler,
- un ensemble d'actions $\mathcal{A}$ à appliquer.


<br>

## I. Propriété de Markov

À chaque pas de temps, l'état du système est $S_t$ et auquel une action $A_t$ est appliquée. Cela se traduit par l'observation d'un nouvel état $S_{t+1}$ et la réception d'un signal de récompense scalaire $R_t$ pour cette transition.   
$R_t$ indique à quel point il faut être satisfait de la dernière transition.   

*A noter: $S_t$, $A_t$, $S_{t+1}$ et $R_t$ sont des variables aléatoires.*


<br>

**Hypothèse fondamentale (propriété de Markov)**
$$\mathbb{P}(S_{t+1},R_t|S_t, A_t, S_{t-1}, A_{t-1}, \ldots, S_0, A_0) = \mathbb{P}(S_{t+1},R_t|S_t, A_t)$$

Un système qui suit cette propriété sera appelé processus de décision de Markov (MDP).

On sépare généralement la dynamique des états et les récompenses par :
$$\mathbb{P}(S_{t+1},R_t|S_t, A_t) = \mathbb{P}(S_{t+1}|S_t, A_t)\cdot \mathbb{P}(R_t|S_t, A_t, S_{t+1})$$

Ce qui conduit à son tour à la définition générale d'un MDP :

- Un ensemble d'états $\mathcal{S}$
- Un ensemble d'actions $\mathcal{A}$
- Un modèle de transition (markovien) $\mathbb{P}\left(S_{t+1} | S_t, A_t \right)$, noté $p(s'|s,a)$
- Un modèle de récompense $\mathbb{P}\left( R_t | S_t, A_t, S_{t+1} \right)$, noté $r(s,a)$ ou $r(s,a,s')$
- Un ensemble **d'époques** de décision discrètes $\mathcal{T}=\{0,1,\ldots,h\}$

Si $h\rightarrow\infty$, c'est un problème de contrôle à horizon infini.   
Comme dans la plupart du temps avec les problèmes sont d'horizon infini, le MDP est identifié avec le 4-uplet $(\mathcal{S},\mathcal{A},p,r)$.

Dans le cas général, $\mathcal{S}$ et $\mathcal{A}$ peuvent être chacun :
- des ensembles finis arbitraires,
- des ensembles infinis dénombrables arbitraires,
- des sous-ensembles compacts d'un espace euclidien de dimension finie, ou
- des sous-ensembles boréliens non vides d'espaces métriques complets et séparables.

Le modèle de récompense peut être écrit indifféremment $r(s,a)$ ou $r(s,a,s')$, selon les auteurs, avec $r(s,a) = \mathbb{E}_{s'\sim p(\cdot|s,a)} [r(s,a,s')]$.      

Ainsi, en RL, le but est de contrôler la trajectoire d'un système qui, par hypothèse, se comporte comme un processus de décision de Markov.


<br>

## II. Politiques

Soit $\Delta_\mathcal{A}$ l'ensemble des mesures de probabilité sur l'espace d'action $\mathcal{A}$. Alors la loi $\pi_t$ de $A_t$ appartient à $\Delta_\mathcal{A}$.
$\pi_t$ est appelée la **règle de décision** à l'étape $t$, c'est une distribution sur l'espace d'action $\mathcal{A}$.
La collection $\pi = \left(\pi_t \right)_{t\in T}$ est appelée une **politique**.

Une politique $\pi$ est une séquence de règles de décision $\pi_t$ : $\pi = \{\pi_t\}_{t\in T}$, avec $\pi_t \in \Delta_\mathcal{A}$.


<br>

### 1. Evaluataion des politiques

Certaines politiques sont parfois meilleures que d'autres. Il faut alors décider d'un critère pour comparer les politiques. Intuitivement, ce critère devrait refléter l'idée qu'une bonne politique accumule autant de récompenses que possible le long d'une trajectoire.   
Dans le cas général, cette somme de récompenses sur un horizon infini pourrait être illimitée. Il faut alors donc la variable aléatoire **$\gamma$-somme actualisée des récompenses** (à partir d'un état de départ $s$, sous la politique $\pi$) :
$$G^\pi(s) = \sum\limits_{t = 0}^\infty \gamma^t R_t \quad \Bigg| \quad \begin{array}{l}S_0 = s,\\ A_t \sim \pi_t,\\ S_{t+1}\sim p(\cdot|S_t,A_t),\\R_t = r(S_t,A_t,S_{t+1}).\end{array}$$

$G^\pi(s)$ représente ce qu'il est possible de gagner à long terme en appliquant les actions de $\pi$.

Ensuite, étant donné un état de départ $s$, il est possible de définir la valeur de $s$ sous la politique $\pi$ :
$$v^\pi(s) = \mathbb{E} \left[ G^\pi(s) \right]$$

Cela définit la fonction de valeur $v^\pi$ de la politique $\pi$ sous un critère d'actualisation $\gamma$ :
$$
v^\pi : 
\begin{cases} 
\mathcal{S} \rightarrow \mathbb{R}, \\ 
s \mapsto v^\pi(s) = \mathbb{E}\left( \sum\limits_{t = 0}^\infty \gamma^t R_t \Bigg| S_0 = s, \pi \right)
\end{cases}
$$



Et, étant donné une distribution $\rho_0$ sur les états de départ, il est possible de "mapper" $\pi$ sur la valeur scalaire :
$$J(\pi) = \mathbb{E}_{s \sim \rho_0} \left[ v^\pi(s) \right]$$

*Il est à noter que cette définition est assez arbitraire : au lieu de la somme attendue (actualisée) des récompenses, il aurait été possible de prendre la récompense moyenne sur tous les pas de temps, ou un autre critère de comparaison (plus ou moins exotique) entre les politiques:*

- **Horizon fini**  
  $v(s) = \mathbb{E}\left( \sum\limits_{t = 0}^h R_t \bigg| s_0 = s \right)$, avec $h \in \mathbb{N}$

- **Récompense moyenne**  
  $v(s) = \mathbb{E}\left( \lim\limits_{h\rightarrow\infty} \frac{1}{h} \sum\limits_{t = 0}^h R_t \bigg| s_0 = s \right)$  

- **Récompense totale**  
  $v(s) = \mathbb{E}\left( \lim\limits_{h\rightarrow\infty} \sum\limits_{t = 0}^h R_t \bigg| s_0 = s \right)$  

- **Récompense actualisée**  
  $v(s) = \mathbb{E}\left( \lim\limits_{h\rightarrow\infty} \sum\limits_{t = 0}^h \gamma^t R_t \bigg| s_0 = s \right)$, avec $0 \leq \gamma < 1$


<br>


- Le critère de récompense moyenne: caractérise la récompense moyenne par pas de temps que l'agent reçoit.
- Le critère de récompense totale: maximise les récompenses cumulées obtenues au cours d'un épisode. Mais il ne fait pas de distinction entre celles obtenues au début ou à la fin de l'épisode. De plus, il souffre d'un défaut majeur : pour les problèmes à horizon infini, même si le modèle de récompense est borné, cette somme peut diverger.
- Le critère de récompense actualisée: le facteur d'actualisation ($0\leq \gamma<1$) garantit qu'avec les modèles de récompenses bornés $r$, la somme converge toujours.   

La plupart des travaux sur le RL utilisent ce critère rajusté (dans certains cas avec $\gamma=1$), certains travaux utilisent un critère d'horizon fini, d'autres utilisent le critère de récompense moyenne, et quelques travaux s'aventurent dans des critères plus exotiques. 

<br>

### 2. Politiques optimales

Maintenant qu'il est possible comparer des politiques à partir d'un état initial (ou d'une distribution d'états initiaux), il est possible de déterminer la meilleure politique pour un état initial donné. Cette politique est la politique **optimale**.

$\pi^*$ est dite optimale ssi $\pi^* \in \arg\max\limits_{\pi} v^\pi$.<br>

Une politique est optimale si elle **domine** toute autre politique dans chaque état :
$$\pi^* \textrm{ est optimale}\Leftrightarrow \forall s\in S, \ \forall \pi, \ v^{\pi^*}(s) \geq v^\pi(s)$$

*A noter: il est possible qu'il y ait plusieurs politiques optimales. Elles partagent toutes la même fonction de valeur $v^* = v^{\pi^*}$.*

<br>

**Théorème : famille de politiques optimales**   
Pour $\left\{\begin{array}{l}
\gamma\textrm{-critère actualisé}\\
\textrm{horizon infini}
\end{array}\right.$,
il existe toujours au moins une politique optimale stationnaire, déterministe et sans mémoire.


- Sans mémoire (aussi appelé Markovien) : **toutes les règles de décision sont uniquement conditionnées par le dernier état vu**. Mathématiquement :
$\left.\begin{array}{l}
\forall \left(s_i,a_i\right)_{i\leq t-1}\in \left(\mathcal{S}\times \mathcal{A}\right)^{t-1}\\
\forall \left(s'_i,a'_i\right)_{i\leq t-1}\in \left(\mathcal{S}\times \mathcal{A}\right)^{t-1}\\
\forall s \in \mathcal{S}
\end{array}\right\}, \pi_t\left(A_t|S_0=s_0, A_0=a_0, \ldots, S_t=s\right) = \pi_t\left(A_t|S'_0=s'_0, A'_0=a'_0, \ldots, S_t=s\right)$.   
On écrit alors $\pi_t(A_t|S_t=s)$, ou plus simplement $\pi_t(\cdot | s)$ ou $\pi_t(s) \in \Delta_\mathcal{A}$.

- Stationnaire (et sans mémoire) : **toutes les règles de décision sont les mêmes au cours du temps**. Mathématiquement :
$\forall (t,t')\in \mathbb{N}^2, \pi_t(A_t|S_t=s) = \pi_{t'}(A_{t'}|S_{t'}=s)$.   
Cette distribution unique s'écrit $\pi(\cdot | s) = \pi_t( \cdot | s)$ ou $\pi(s) \in \Delta_\mathcal{A}$.

- Déterministe : **toutes les règles de décision placent toute la masse de probabilité sur un seul élément de l'espace d'action** $\mathcal{A}$.
$\pi_t(A_t|history) = \left\{\begin{array}{l}
1\textrm{ pour un seul }a\\
0\textrm{ sinon}
\end{array}\right.$.

Donc, en termes plus simples, parmi toutes les manières optimales possibles de choisir $A_t$, au moins une est une fonction $\pi:\mathcal{S}\rightarrow \mathcal{A}$.   
Il est alors possible de simplement rechercher une fonction $\pi(s)=a$ qui associe les états aux actions.


<br>

### 3. Optimalité en moyenne sur les états

Si tous les états ont une probabilité non nulle d'être visités par une politique, alors une politique optimale doit choisir les actions optimales le long de toute trajectoire commençant dans (n'importe quel) $s_0$, donc elle doit choisir les actions optimales dans tous les états. Alors trouver une politique qui maximise $v^\pi$ dans tous les états revient en fait à trouver une politique qui maximise la valeur espérée $v^\pi(s_0)$ d'un état initial fixe $s_0$, ou la valeur espérée sur n'importe quelle distribution sur les états initiaux $\mathbb{E}_{s_0\sim \rho_0}[v^\pi(s_0)]$.

Soit $J(\pi) = \mathbb{E}_{s_0\sim \rho_0}[v^\pi(s_0)]$. Alors :


**Le problème d'optimisation de la politique :**
À condition que tous les états soient accessibles depuis n'importe quel autre état, une politique optimale est une solution à $\max_\pi J(\pi) = \mathbb{E}_{s_0\sim \rho_0}[v^\pi(s_0)]$.


En supposant que tous les états sont accessibles depuis n'importe quel autre le problème de maximisation de la valeur dans *chaque état* (optimalité ponctuelle) devient un problème de maximisation de la valeur *en moyenne* sur une distribution $\rho_0$. 

L'hypothèse selon laquelle tous les états sont accessibles depuis n'importe quel autre est très forte. Lorsqu'elle n'est pas vérifiée (ce qui se produira dans de nombreux cas réels), le problème d'optimisation $\max_\pi J(\pi) = \mathbb{E}_{s_0\sim \rho_0}[v^\pi(s_0)]$ pourrait ne pas donner lieu à une politique totalement optimale. Il fournira une politique qui maximise son résultat attendu en moyenne sur tous les états, sur un ensemble spécifique d'états de départ distribués selon $\rho_0$.

Il est intéressant de noter que $\rho_0$ n'a pas besoin d'être interprété comme une distribution d'états de départ (même s'il est mentalement pratique de le voir de cette façon). Cela conduit à une formulation alternative du problème d'optimisation de politique comme la recherche d'une politique qui a la fonction de valeur la plus élevée en moyenne sur tous les états.
<br>

**Le problème d'optimisation de politique :**
À condition que $\rho_0$ ait une masse de probabilité non nulle sur tous les états, une politique optimale est une solution à $\max_\pi J(\pi) = \mathbb{E}_{s_0\sim \rho_0}[v^\pi(s_0)]$.

<br>
<br>


Les deux notions d'optimalité introduites ci-dessus ouvrent la voie à deux familles différentes de méthodes en RL. Soit on recherche la fonction de valeur optimale (optimalité ponctuelle), puis on en déduit une politique optimale comme sous-produit, soit on optimise directement pour de bonnes politiques de valeur moyenne (optimalité en moyenne). 

En pratique, aucune approche n'est plus justifiée que l'autre et toutes deux méritent d'être étudiées. 

Optimisation des politiques, résoudre le problème $\max_\pi$ directement :
- par des méthodes sans dérivées (RL évolutionnaire)
- par des méthodes basées sur les dérivées (méthodes du gradient de politique et leurs extensions modernes)

Optimisation de la valeur, résoudre pour $v^*$ :
- par programmation dynamique (approximative) (et la vaste gamme des algorithmes d'itération de valeur approximative les plus récents)
- par programmation linéaire ou formulations alternatives


<br>

## III. Limites du modèle MDP

Que se passe-t-il si le système est un MDP mais que son état n'est pas entièrement observable ?
$\rightarrow$ C'est le domaine des MDP partiellement observables. Notre résultat clé d'avoir une politique optimale markovienne ne tient plus. Il existe des moyens d'obtenir encore des politiques optimales (mais c'est souvent très coûteux en calcul) ou de les approximer avec des politiques markoviennes.

Que se passe-t-il si plusieurs actions sont effectuées en même temps par différents agents ?
$\rightarrow$ Cela entre dans la catégorie des jeux stochastiques multijoueurs. Ces jeux peuvent être antagonistes, coopératifs ou un mélange des deux. Bien sûr, ils peuvent aussi avoir une observabilité partielle.

Et si le modèle de transition n'était pas markovien ?
$\rightarrow$ Attention, il y a des dragons ici ! Tout le beau cadre ci-dessus s'effondre si ses hypothèses sont violées. Il faut donc faire très attention lors du choix des variables d'état pour un problème donné. Dans un sens, un MDP est une version à temps discret d'une équation différentielle du premier ordre. Écrire un système comme $\dot{x} = f(x,u, noise)$, comme c'est courant en théorie du contrôle, est une bonne pratique pour garantir la propriété de Markov.

<br>

## IV. Résumé

En résumé, il a été possible de formaliser le **problème général de contrôle optimal stochastique à temps discret** :
- Environnement (temps discret, non déterministe, non linéaire, Markov) $\leftrightarrow$ MDP.
- Politique de contrôle du comportement $\leftrightarrow$ $\pi : \mathcal{S}\rightarrow \mathcal{A}$ ou $\Delta_\mathcal{A}$.
- Critère d'évaluation de la politique $\leftrightarrow$ Critère actualisé $\gamma$.
- Objectif $\leftrightarrow$ Maximiser la fonction de valeur $v^\pi(s)$.


Le système à contrôler est un processus de décision de Markov $(\mathcal{S}, \mathcal{A}, p, r)$, et à contrôler avec une politique $\pi:s\mapsto a$, afin d'optimiser $\mathbb{E} \left( \sum_t \gamma^t R_t\right)$


<br>

## V. Gymnasium

**Gymnasium** est une bibliothèque Python utilisée pour créer et interagir avec des environnements d'apprentissage par renforcement. C'est un fork de la populaire bibliothèque OpenAI Gym, offrant des environnements standardisés où les systèmes peuvent apprendre à effectuer des tâches en interagissant avec eux. Elle fournit des outils pour la simulation, l'observation et l'évaluation des performances d'un système.

### Concepts de Base

#### 1. **Environnements**
Un environnement dans Gymnasium est un cadre où un système interagit en exécutant des actions. Le système reçoit une observation de l'état de l'environnement et un signal de récompense pour chaque action.

#### 2. **Fonctionnalités principales**
- `make`: Crée un environnement.
- `reset`: Réinitialise l'environnement à son état initial.
- `step`: Fait progresser l'environnement d'un pas de temps en fonction d'une action.
- `observation_space`: Définit l'espace des observations possibles.
- `action_space`: Définit l'espace des actions possibles.

---

### Fonctionnalités de Base

#### **1. `gymnasium.make`**
La fonction `make` est utilisée pour créer un environnement.

```python
import gymnasium as gym

env = gym.make("CartPole-v1")
```

Dans cet exemple, `"CartPole-v1"` est le nom de l'environnement. Gymnasium propose une large gamme d'environnements, chacun adapté à des tâches spécifiques.

#### **2. `env.observation_space`**
Cette propriété décrit l'espace des observations possibles que le système peut recevoir de l'environnement.

Exemple pour `"CartPole-v1"` :
```python
print(env.observation_space)
```

Cela pourrait renvoyer :
```
Box([-4.8, -inf, -0.42, -inf], [4.8, inf, 0.42, inf], (4,), float32)
```

Cela signifie que :
- Les observations sont un tableau (de type `Box`) de taille 4.
- Chaque élément de l'observation a des bornes spécifiques (par exemple, la position du chariot, l'angle du poteau, etc.).

#### **3. `env.action_space`**
Cette propriété définit l'espace des actions possibles que le système peut prendre.

Exemple pour `"CartPole-v1"` :
```python
print(env.action_space)
```

Cela pourrait renvoyer :
```
Discrete(2)
```

Cela signifie que le système peut choisir parmi **deux** actions discrètes (par exemple, pousser le chariot à gauche ou à droite).

#### **4. `env.reset`**
Réinitialise l'environnement à son état initial et retourne une observation initiale.

Exemple :
```python
observation, info = env.reset()
print(observation)
```

Cela renvoie l'observation initiale de l'environnement sous forme d'un tableau.

#### **5. `env.step`**
Cette méthode permet à le système de prendre une action, et l'environnement renvoie les éléments suivants :
- Une observation résultante.
- Une récompense associée à l'action.
- Une valeur booléenne indiquant si l'épisode est terminé.
- Des informations supplémentaires.

Exemple :
```python
action = env.action_space.sample()  # Choisir une action aléatoire
observation, reward, done, truncated, info = env.step(action)
print(observation, reward, done, info)
```

Ici :
- `action_space.sample()` choisit une action aléatoire.
- `done` est `True` si l'épisode est terminé.
- `info` peut contenir des métadonnées utiles.

---

### Exemple Complet

Voici un exemple de base qui montre comment utiliser Gymnasium pour entraîner un système dans l'environnement `"CartPole-v1"` :

```python
import gymnasium as gym

# Créer l'environnement
env = gym.make("CartPole-v1", render_mode="human")

# Réinitialiser l'environnement
observation, info = env.reset()

for _ in range(1000):  # Effectuer 1000 étapes
    env.render()  # Affiche l'environnement (si applicable)
    
    # Sélectionner une action aléatoire
    action = env.action_space.sample()
    
    # Effectuer l'action
    observation, reward, done, truncated, info = env.step(action)
    
    # Si l'épisode est terminé, réinitialiser l'environnement
    if done or truncated:
        observation, info = env.reset()

env.close()  # Fermer l'environnement
```