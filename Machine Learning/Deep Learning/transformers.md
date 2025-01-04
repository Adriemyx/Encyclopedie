<h1 align='center'> Deep Learning - Transformers 🤖 </h1>

Les transformers sont une architecture clé en deep learning, introduite dans l'article *"Attention is All You Need"* (Vaswani et al., 2017). Ils ont révolutionné des domaines comme le traitement automatique du langage naturel (NLP), la vision par ordinateur et même la modélisation moléculaire.

### Avantages
- **Parallélisation**: Contrairement aux RNNs, les transformers traitent tous les éléments de la séquence simultanément.
- **Flexibilité**: Fonctionnent avec des séquences de longueurs variées.
- **Performance**: Modélisent efficacement les dépendances à longue distance.

### Limites
- **Complexité quadratique**: La self-attention a une complexité de $O(n^2)$ en temps et en mémoire.
- **Sensibilité à la quantité de données**: Nécessitent de grandes bases d'entraînement.

<br>

## I. **Concepts de base**
Les transformers utilisent une approche d'auto-attention qui permet de modéliser efficacement les relations entre les éléments d'une séquence, quelle que soit leur distance relative.

### Architecture globale
L'architecture typique des transformers se compose de:
- **Encodeurs** (*Encoder*): Traitent la séquence d'entrée.
- **Décodeurs** (*Decoder*): Génèrent la séquence de sortie (utile pour des tâches de traduction, par exemple).

Chaque encodeur ou décodeur est composé de:
1. Une couche de **multi-head self-attention**.
2. Une couche **feed-forward fully connected**.
3. Des mécanismes de **normalisation** et des **résidus**.


<br>

## 2. **Mathématiques sous-jacentes**

### 2.1 Self-Attention

L'auto-attention est le mécanisme clé. Chaque élément d'une séquence est représenté par une **requête** ($Q$), une **clé** ($K$) et une **valeur** ($V$):
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$
où:
- $X \in \mathbb{R}^{n \times d}$ est la matrice des vecteurs d'entrée ($n$: nombre d'éléments, $d$: dimension).
- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ sont des matrices de poids apprises.

La similarité entre $Q$ et $K$ est calculée via un produit scalaire:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### Étapes détaillées:
1. Calculer les scores $\frac{QK^T}{\sqrt{d_k}}$: indique à quel point les éléments de la séquence s'influencent mutuellement.
2. Appliquer une normalisation $\text{softmax}$: transforme les scores en probabilités.
3. Pondérer les valeurs $V$ en fonction des probabilités obtenues.

---

### 2.2 Multi-Head Attention

Pour améliorer la capacité de modélisation, plusieurs têtes d'attention ($h$) sont calculées en parallèle:
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O$$
où chaque tête est une instance indépendante de self-attention.

---

### 2.3 Positionnal Encoding

Puisque les transformers ne capturent pas directement l'ordre séquentiel, on ajoute un encodage positionnel à l'entrée:
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$
Cela injecte des informations sur la position dans les vecteurs d'entrée.

---

### 2.4 Feed-Forward Network (FFN)

Chaque encodeur/décodeur comprend une couche totalement connectée:
$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$


<br>

## 4. **Implémentation en Python**

Voici une implémentation simplifiée avec PyTorch.

### 4.1 Importation des bibliothèques
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

### 4.2 Self-Attention
```python
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads

        assert self.head_dim * heads == embed_dim, "Embed dim must be divisible by heads"

        self.values = nn.Linear(embed_dim, embed_dim, bias=False)
        self.keys = nn.Linear(embed_dim, embed_dim, bias=False)
        self.queries = nn.Linear(embed_dim, embed_dim, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        N, seq_length, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        values = self.values(x)
        keys = self.keys(x)
        queries = self.queries(x)

        values = values.view(N, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        keys = keys.view(N, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)
        queries = queries.view(N, seq_length, self.heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / (self.head_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, seq_length, embed_dim)

        return self.fc_out(out)
```

### 4.3 Encodeur
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, heads, forward_expansion, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = SelfAttention(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention = self.self_attention(x)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        x = self.norm2(forward + x)
        return self.dropout(x)
```

### 4.4 Transformer complet
```python
class Transformer(nn.Module):
    def __init__(self, embed_dim, num_layers, heads, forward_expansion, dropout, vocab_size, max_length):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim
        self.word_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, heads, forward_expansion, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(x.device)
        x = self.word_embedding(x) + self.position_embedding(positions)

        for layer in self.layers:
            x = layer(x)

        return self.fc_out(x)
```

---

Ce modèle peut être entraîné avec une tâche de prédiction comme un modèle de langage.