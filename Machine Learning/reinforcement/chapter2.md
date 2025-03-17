<h1 align='center'> Reinforcement learning - Chapitre 2 : Caractérisation des fonctions de valeur - les équations de Bellman </h1>

L'idée de ce chapitre est de s'appuyer sur les propriétés MDP de l'environnement à contrôler, afin de caractériser les politiques à travers leurs fonctions de valeur.
<br>

## I. L'équation d'évaluation
Il est possible de définir des fonctions de valeur état-action, également appelées Q-fonctions, qui constituent un objet central en RL. Une Q-fonction est une fonction $\mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$.

La fonction de valeur état-action associée à la politique $\pi$, notée $q^\pi$, représente **le rendement attendu en effectuant l'action $a$ dans l'état $s$, puis en suivant la politique $\pi$ dans tous les états ultérieurs.** En d'autres termes, elle correspond à la somme de la récompense immédiate et de la récompense future espérée obtenue en suivant la politique $\pi$.

**Fonction valeur état-action**

$$q^\pi(s,a) = \mathbb{E}\left( \sum\limits_{t=0}^\infty \gamma^t r\left(S_t, A_t, S_{t+1}\right) \bigg| S_0 = s, A_0=a, \pi \right)$$


Pour être précis et réutiliser les notations complètes de la définition MDP :


$$q^\pi(s,a) =\mathbb{E}\left[ \sum\limits_{t = 0}^\infty \gamma^t R_t \quad \Bigg| \quad \begin{array}{l}S_0 = s, A_0=a,\\ A_t \sim \pi(S_t)\textrm{ for }t>0,\\ S_{t+1}\sim p(\cdot|S_t,A_t),\\R_t = r(S_t,A_t,S_{t+1})\end{array} \right]$$

 $$= \mathbb{E}_{s'} \left[ r(s,a,s') + \gamma v^\pi(s') \right]$$

 $$= r(s,a) + \gamma \mathbb{E}_{s'} \left[ v^\pi(s') \right]$$


<br>

La variable aléatoire *retour* $G^\pi(s)$ pour une politique $\pi$, dans chaque état $s$, était définie comme la somme des récompenses actualisées avec le facteur $\gamma$ :  
$$G^\pi(s) = \sum\limits_{t = 0}^\infty \gamma^t R_t \quad \Bigg| \quad \begin{array}{l} S_0 = s,\\ A_t \sim \pi(S_t),\\ S_{t+1} \sim p(\cdot|S_t, A_t),\\ R_t = r(S_t, A_t, S_{t+1}).\end{array}$$

Avec donc $v^\pi(s) = \mathbb{E}[G^\pi(s)]$.

De manière similaire, il est possible de définir la variable aléatoire *retour état-action* $G^\pi(s,a)$ pour une politique $\pi$, dans chaque paire état-action $(s,a)$, comme la somme des récompenses actualisées avec le facteur $\gamma$ :  
$$G^\pi(s,a) = \sum\limits_{t = 0}^\infty \gamma^t R_t \quad \Bigg| \quad \begin{array}{l} S_0 = s, A_0 = a,\\ A_t \sim \pi(S_t) \text{ pour } t > 0,\\ S_{t+1} \sim p(\cdot|S_t, A_t),\\ R_t = r(S_t, A_t, S_{t+1}).\end{array}$$

Ainsi, $q^\pi(s,a) = \mathbb{E}[G^\pi(s,a)]$. 


*<u>Remarque:</u>* Cette définition des Q-fonctions utilise le critère d'actualisation $\gamma$, mais elle peut être étendue de manière directe à tout autre critère. :
Soit $c((R_t)_{t \in \mathbb{N}})$ un critère défini sur la séquence des variables aléatoires de récompense.  
Alors, la variable aléatoire de retour $G^\pi(s,a)$ est définie comme la variable aléatoire $c((R_t)_{t \in \mathbb{N}})$, sachant que $S_0 = s$, $A_0 = a$, $A_t \sim \pi(S_t)$ pour $t > 0$, $S_{t+1} \sim p(\cdot|S_t, A_t)$, et $R_t = r(S_t, A_t, S_{t+1})$.  
La Q-fonction correspondante pour ce critère est simplement donnée par :  
$$q^\pi(s,a) = \mathbb{E}\left[ c((R_t)_{t \in \mathbb{N}}) \quad \Bigg| \quad \begin{array}{l} S_0 = s, A_0 = a,\\ A_t \sim \pi(S_t) \text{ pour } t > 0,\\ S_{t+1} \sim p(\cdot|S_t, A_t),\\ R_t = r(S_t, A_t, S_{t+1}). \end{array} \right].$$ 


<br>

Dans ce cas, $v^\pi(s) = q^\pi(s,\pi(s))$. Or, comme $a = \pi(s)$, le résultat permet d'obtenir une équation importante pour caractériser $v^\pi$:

$$v^\pi(s) = r(s,\pi(s)) + \gamma \mathbb{E}_{s'\sim p(s'|s,\pi(s))} \left[ v^\pi(s') \right]$$

Cette équation utilise $v^\pi(s')$ dans tous les $s'$ accessibles depuis $s$ pour définir $v^\pi(s)$.
Puisque cette équation est vraie dans tous les $s$, cela fournit autant d'équations qu'il y a d'états.

<br>


$v^\pi$ obéit alors au système d'équations linéaires :
$$
v^\pi\left(s\right) = r(s,\pi(s)) + \gamma \mathbb{E}_{s'\sim p(s'|s,\pi(s))} \left[ v^\pi(s') \right]\\
$$
De même :
$$
q^\pi\left(s,a\right) = r(s,a) + \gamma \mathbb{E}_{s'\sim p(s'|s,a)} \left[ q^\pi(s',\pi(s')) \right]
$$


Cela conduit à l'introduction de l'**opérateur d'évaluation de Bellman** $\mathbb{T}^\pi$:

$\mathbb{T}^\pi$ est un opérateur sur les fonctions de valeur, qui transforme une fonction $v:\mathcal{S}\rightarrow \mathbb{R}$ en :


$$\mathbb{T}^\pi v\left(s\right) = r(s,\pi(s)) + \gamma \mathbb{E}_{s'\sim p(s'|s,\pi(s))} \left[ v(s') \right]$$

$$= r\left(s,\pi\left(s\right)\right) + \gamma \sum\limits_{s'\in \mathcal{S}} p\left(s'|s,\pi\left(s\right)\right) v\left(s'\right)$$


De la même manière, nous pouvons introduire une évaluation opérateur (avec le même nom $\mathbb{T}^\pi$) sur les fonctions de valeur d'état-action. <br>
$\mathbb{T}^\pi$ est un opérateur sur les fonctions de valeur état-action, qui transforme une fonction $q:\mathcal{S}\times \mathcal{A}\rightarrow \mathbb{R}$ en :

$$\mathbb{T}^\pi q\left(s,a\right) = r(s,a) + \gamma \mathbb{E}_{s'\sim p(s'|s,a)} \left[ q(s',\pi(s')) \right]$$
$$= r\left(s,a\right) + \gamma \sum\limits_{s'\in \mathcal{S}} p\left(s'|s,a\right) q\left(s', \pi\left(s'\right)\right)$$


Ainsi, trouver $v^\pi$ (resp. $q^\pi$) revient à résoudre l'équation d'évaluation $v= \mathbb{T}^\pi v$ (resp. $q = \mathbb{T}^\pi q$).



<br>


Pour une politique déterministe sans mémoire et un modèle de récompense de la forme $r(s,a)$, ces équations deviennent: 
$$\mathbb{T}^\pi v\left(s\right) = \mathbb{E}_{\substack{a \sim \pi(a|s) \\ s'\sim p(s'|s,a)}} \left[ r(s,a,s') + \gamma v(s') \right]$$

$$\mathbb{T}^\pi q\left(s,a\right) = \mathbb{E}_{\substack{s'\sim p(s'|s,a) \\ a' \sim \pi(a'|s')}} \left[ r(s,a,s') + \gamma q(s',a') \right]$$


Il est possible de définir un *retour bootstrappé* $G^\pi_m(s,a,q)$, pour $m\geq 1$, comme variable aléatoire :
$$G^\pi_m(s,a,q) = \sum\limits_{t = 0}^{m-1} \gamma^t R_t + \gamma^m q(S_m, A_m) \quad \Bigg| \quad \begin{array}{l}S_0 = s, A_0=a\\ A_t \sim \pi(S_t)\textrm{ pour }t>0,\\ S_{t+1}\sim p(\cdot|S_t,A_t),\\R_t = r(S_t,A_t,S_{t+1}).\end{array}$$


En particulier :
$$G^\pi_1(s,a,q) = R_0 + \gamma q(S_1, A_1) \quad \Bigg| \quad \begin{array}{l}S_0 = s, A_0=a\\ A_1 \sim \pi(S_1),\\ S_{1}\sim p(\cdot|S_0,A_0),\\ R_0 = r(S_0,A_0,S_{1}).\end{array}$$

*<u>Remarque:</u>* $(\mathbb{T}^\pi q)(s,a) = \mathbb{E} \left[ G^\pi_1(s,a,q) \right]$

<br>

### Résumé

- Une politique $\pi$ est le comportement d'un agent.
- Dans chaque état $s$, on peut espérer gagner $v^\pi(s)$ à long terme en appliquant $\pi$.
- $v^\pi(s)$ est la somme de la récompense sur la première étape $r(s,\pi(s))$ et du rendement à long terme attendu de l'état suivant $\gamma \mathbb{E}_{s'} \left[v^\pi(s')\right]$.
- La fonction $v^\pi$ obéit en fait au système d'équations linéaires ci-dessus qui relie simplement la valeur d'un état aux valeurs de ses successeurs dans un épisode.


**Propriétés de $\mathbb{T}^\pi$:**

- $\mathbb{T}^\pi$ est un opérateur affine, il définit un système linéaire d'équations.
- $\mathbb{T}^\pi$ est une application de contraction.   
Plus précisément, avec $\gamma<1$, $\mathbb{T}^\pi$ est un $\| \cdot \|_\infty$-cartographie des contractions sur l'espace de Banach $\mathbb{R}^\mathcal{S}$ (resp. $\mathbb{R}^{\mathcal{S} \mathcal{A}}$).<br>
$\Rightarrow$ Avec $\gamma<1$, $v^\pi$ (resp. $q^\pi$) est l'unique solution à l'équation (linéaire) du point fixe :<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$v=\mathbb{T}^\pi v$ (resp. $q=\mathbb{T}^\pi q$).


## Programmation dynamique pour l'équation d'évaluation

Soit $q_0(s,a) = 0$ pour tous les $(s,a)$.

L'application de $\mathbb{T}^\pi$ une fois donne :
$$q_1(s,a) = \sum_{s'} p(s'|s,a) \left[ r(s,a,s') + \gamma q_0(s',\pi(s')) \right]$$

En termes simples, $q_1$ est le rendement attendu en une étape sous la politique $\pi$.

L'application de $\mathbb{T}^\pi$ deux fois donne :
$$q_2(s,a) = \sum_{s'} p(s'|s,a) \left[ r(s,a,s') + \gamma q_1(s',\pi(s')) \right]$$

Il s'agit du rendement attendu en deux étapes.

Et ainsi de suite.

Ainsi, comme $\mathbb{T}^\pi$ est une application de contraction, si $\mathbb{T}^\pi$ est appliquée suffisamment de fois, $q_n$ devrait se rapprocher de $q^\pi$, quelle que soit la valeur choisie pour $q_0$, car la séquence $q_{n+1} = \mathbb{T}^\pi q_n$ converge vers le point fixe de $\mathbb{T}^\pi$.

Étant donné que la procédure d'application répétée de $\mathbb{T}^\pi$ décompose le problème d'évaluation de $\mathbb{E} [\sum_t \gamma^t R_t]$ en sous-problèmes d'horizon croissant, il s'agit d'une procédure de **[programmation dynamique](https://www.jstor.org/stable/j.ctv1nxcw0f)**.

La suite de $\| q_n - q_{n-1} \|_\infty$ converge bien vers 0 et la décroissance est logarithmique (car à chaque itération, il y a une multiplication par $\gamma$).


<br>

## L'équation d'optimalité
De même, pour la valeur d'une politique optimale:
$$v^{\pi^*} = v^*, \quad q^{\pi^*} = q^*$$


**Théorème : Politique optimale gloutonne**

Toute politique $\pi$ définie par $\pi(s) \in \arg\max\limits_{a\in \mathcal{A}} q^*(s,a)$ est une politique optimale.

Pour les politiques stochastiques: toute politique $\pi$ définie par $\pi(s) \in \arg\max\limits_{\delta \in \Delta_\mathcal{A}} \mathbb{E}_{a\sim \delta} [q^*(s,a)]$ est une politique optimale.

Et $q^*$ obéit au même type de relation de récurrence :

**Théorème : équation d'optimalité de Bellman**

La fonction de valeur optimale obéit à :

$$v^*(s) = \max\limits_{a\in \mathcal{A}} \left[ r(s,a) + \gamma \mathbb{E}_{s'\sim p(s'|s,a)} v^*(s') \right]$$
$$= \max\limits_{a\in \mathcal{A}} \left[ r(s,a) + \gamma \sum\limits_{s'\in \mathcal{S}} p(s'|s,a) v^*(s') \right]$$

ou en termes de Fonctions Q :
$$q^*(s,a) = r(s,a) + \gamma \mathbb{E}_{s'\sim p(s'|s,a)} \left[ \max_{a'\in \mathcal{A}} q^*(s',a') \right]$$
$$= r(s,a) + \gamma \sum\limits_{s'\in \mathcal{S}}p(s'|s,a) \max\limits_{a'\in \mathcal{A}} q^*(s',a')$$


**Opérateur d'optimalité de Bellman:**

$$\left(\mathbb{T}^*v\right)(s) = \max\limits_{a\in \mathcal{A}} \left[ r(s,a) + \gamma \mathbb{E}_{s'\sim p(s'|s,a)} v(s') \right]$$
$$\left(\mathbb{T}^*q\right)(s,a) = r(s,a) + \gamma \mathbb{E}_{s'\sim p(s'|s,a)} \left[ \max_{a'\in \mathcal{A}} q(s',a') \right]$$


Ainsi, trouver $v^*$ (resp. $q^*$) revient à résoudre $v= \mathbb{T}^* v$ (resp. $q = \mathbb{T}^* q$).


**Propriétés de $\mathbb{T}^*$:**

- $\mathbb{T}^*$ est non linéaire.
- $\mathbb{T}^*$ est une application de contraction.   
Avec $\gamma<1$, $\mathbb{T}^*$ est une application de $\| \cdot \|_\infty$-cartographie des contractions sur l'espace de Banach $\mathbb{R}^\mathcal{S}$ (resp. $\mathbb{R}^{\mathcal{S} \mathcal{A}}$).<br>
$\Rightarrow$ Avec $\gamma<1$, $v^*$ (resp. $q^*$) est l'unique solution à l'équation du point fixe :<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$v=\mathbb{T}^* v$ (resp. $q=\mathbb{T}^* q$).

<br>

### Programmation dynamique pour l'équation d'optimalité


L'application répétée de $\mathbb{T}^*$ à une fonction initiale $q_0$ donne la séquence $q_{n+1} = \mathbb{T}^* q_n$ qui converge vers $q^*$.


**Itération de valeur**
L'algorithme qui calcule la séquence $q_{n+1} = \mathbb{T}^* q_n$ pour un nombre fini d'itérations est appelé **itération de valeur**.


En pratique, pour calculer $q_{n+1}$ dans un MDP d'états et d'actions fini, on parcourt tous les états $s$ et actions $a$ et on définit $q_{n+1}(s,a) = r(s,a) + \gamma \mathbb{E}_{s'\sim p(s'|s,a)} [ \max_{a'\in \mathcal{A}} q_n(s',a') ]$ pour chaque paire $(s,a)$. 
Chacune de ces affectations nécessite de parcourir à nouveau les états pour calculer l'espérance.

La complexité temporelle d'une itération dans l'itération de valeur, en termes de $|\mathcal{S}|$ et $|\mathcal{A}|$  est $O(|\mathcal{S}|^2 |\mathcal{A}|)$.

<br>

### Résumé

Dans cette section, le but était de s'appuyer sur les propriétés MDP de l'environnement à contrôler, afin de caractériser les politiques à travers leurs fonctions de valeur.

- $q^\pi$ est une solution à l'équation d'évaluation de Bellman:
$$q = \mathbb{T}^\pi q$$
- $q^*$ est une solution à l'équation d'optimalité de Bellman:
$$q = \mathbb{T}^* q$$
- L'itération de valeur construit la séquence de fonctions de valeur $q_{n+1} = \mathbb{T}^* q_n$


**Qu'est-ce qu'une stratégie optimale ?**
Une politique optimale est une politique qui produit des récompenses cumulées optimales. C'est une politique qui est *gloutonne* par rapport à une fonction de valeur optimale $q^*$. Une telle fonction de valeur obéit à l'équation d'optimalité de Bellman et peut être calculée via la programmation dynamique.

Ceci dépend néanmoins d'une caractérisation de $\pi^*$ qui repose sur la connaissance du MDP.

Mais il serait possible d'imaginer une procédure qui *apprend* $q_{n+1}$ à partir d'échantillons de $\mathbb{T}^* q_n$. Si de tels échantillons peuvent être obtenus à partir de l'interaction avec le système à contrôler, il pourrait être possible d'apprendre $q^*$ sans connaître le MDP...