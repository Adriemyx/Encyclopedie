<h1 align='center'> Deep learning - PyTorch 🔦</h1>

PyTorch est une bibliothèque logicielle Python open source d'apprentissage automatique qui s'appuie sur Torch développée par [Meta](https://fr.wikipedia.org/wiki/Meta_(entreprise)). 

<br>

## **1. Les Bases de PyTorch**
### Installation de PyTorch
Pour installer `PyTorch`, il suffit de taper la commande:
```bash
pip install torch torchvision
```

### Importation des Bibliothèques
Les bibliothèques principales dans `PyTorch` sont les suivantes:
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
```

<br>

## **2 Différents Types de Couches**
PyTorch fournit différentes couches pour construire vos modèles.

### Couche dense (Fully Connected)
Une **couche dense** est une couche où chaque neurone est connecté à tous les neurones de la couche précédente. Elle est souvent utilisée dans les parties finales des réseaux pour traiter des données tabulaires ou pour prendre des décisions finales.

#### Calcul mathématique:   
Pour une couche dense avec $n$ neurones:
$$y = W \cdot x + b$$
- $W$: matrice de poids ($n \times m$, où $m$ est le nombre d'entrées).
- $x$: vecteur d'entrée.
- $b$: vecteur de biais.
- $y$: vecteur de sortie.

```python
fc_layer = nn.Linear(in_features=128, out_features=64)  # Couche dense
x = torch.randn(32, 128)  # Batch de 32 avec 128 caractéristiques
y = fc_layer(x)
print(y.shape)  # torch.Size([32, 64])
```


### Couche de Pooling
Les couches de **pooling** sont utilisées pour **réduire la dimensionnalité spatiale** (largeur et hauteur) tout en préservant les informations importantes. Elles permettent de diminuer le coût de calcul et de rendre le modèle robuste aux variations mineures.

#### Types courants:
- **Max Pooling**: Prend la valeur maximale dans une région donnée.
- **Average Pooling**: Calcule la moyenne des valeurs dans une région donnée.

```python
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
x = torch.randn(8, 16, 32, 32)  # Batch de 8 images avec 16 canaux
y = pool_layer(x)
print(y.shape)  # torch.Size([8, 16, 16, 16])
```


### Couche convolutionnelle
Les **couches convolutionnelles** (Convolutional Layers) sont utilisées pour extraire des caractéristiques locales des données, principalement dans des données structurées comme les images, les signaux ou les séquences.   
Une convolution applique un **filtre (kernel)** sur une portion locale de l'entrée (patch) en effectuant un produit scalaire. Les filtres sont appris pendant l'entraînement pour extraire des motifs utiles comme des bords, des textures ou des formes.

```python
# Une couche de convolution avec 3 canaux d'entrée, 16 canaux de sortie, et un kernel 3x3
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
input_image = torch.randn(8, 3, 32, 32)  # Batch de 8 images RGB de taille 32x32
output_image = conv_layer(input_image)
print(output_image.shape)  # torch.Size([8, 16, 32, 32])
```
#### Paramètres importants:
- **`in_channels`**: Nombre de canaux dans l'entrée (e.g., 3 pour RGB).
- **`out_channels`**: Nombre de filtres à apprendre.
- **`kernel_size`**: Taille du filtre (e.g., 3x3, 5x5).
- **`stride`**: Pas de la convolution.
- **`padding`**: Ajout de pixels autour de l'entrée pour préserver les dimensions.



### Couche Récurrente (RNN, LSTM, GRU)
Les **couches récurrentes** sont utilisées pour traiter des données séquentielles, comme des séries temporelles, du texte, ou des signaux audio. Elles permettent de conserver une **mémoire** des étapes précédentes grâce à des états cachés.

#### Types principaux:
- **RNN** (Recurrent Neural Network): Basique, mais souffre de problèmes de gradient.
- **LSTM** (Long Short-Term Memory): Gère les dépendances à long terme.
- **GRU** (Gated Recurrent Unit): Plus léger que LSTM, mais souvent performant.

```python
rnn_layer = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)
x = torch.randn(5, 15, 10)  # Batch de 5 séquences, longueur 15, dimension 10
output, (hn, cn) = rnn_layer(x)
print(output.shape)  # torch.Size([5, 15, 20])
```


### Couche de Normalisation
Les couches de normalisation, comme **Batch Normalization** ou **Layer Normalization**, sont utilisées pour accélérer l'entraînement et stabiliser le modèle en normalisant les activations des couches.

```python
bn_layer = nn.BatchNorm1d(num_features=128)
x = torch.randn(32, 128)  # Batch de 32
y = bn_layer(x)
print(y.shape)  # torch.Size([32, 128])
```


### Couche Dropout
La couche **Dropout** est utilisée pour régulariser le modèle et réduire le sur-apprentissage en mettant aléatoirement à zéro un pourcentage des neurones pendant l'entraînement.

```python
dropout_layer = nn.Dropout(p=0.5)  # 50% des neurones mis à zéro\nx = torch.randn(32, 128)\ny = dropout_layer(x)
print(y.shape)  # torch.Size([32, 128])
```

---

### Couche d'Activation
Les couches d'activation appliquent une transformation non linéaire pour permettre au réseau d'apprendre des relations complexes.

#### Types courants:
- **ReLU**: $f(x) = \max(0, x)$
- **Sigmoid**: $f(x) = \\frac{1}{1 + e^{-x}}$
- **Tanh**: $f(x) = \\tanh(x)$

```python
activation = nn.ReLU()  # Ou nn.Sigmoid(), nn.Tanh()
x = torch.tensor([-1.0, 0.0, 1.0])
y = activation(x)
print(y)  # tensor([0., 0., 1.])
```



<br>

## **3. Définition de la Fonction de Perte**
Les fonctions de perte mesurent l'erreur ou l'écart entre les prédictions du modèle et les valeurs cibles attendues. Leur objectif principal est de guider l'apprentissage en fournissant un signal clair à l'optimiseur pour ajuster les paramètres du modèle (les poids et les biais). Voici quelques critères d’erreur standard:

### Cross-Entropy Loss
```python
criterion = nn.CrossEntropyLoss()
predictions = torch.tensor([[2.5, 0.3, 2.1], [1.0, 3.2, 0.1]], requires_grad=True)
labels = torch.tensor([0, 1])  # Classes correctes
loss = criterion(predictions, labels)
print(loss.item())
```

### Mean Squared Error (MSE)
```python
criterion = nn.MSELoss()
predictions = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
targets = torch.tensor([1.0, 3.0, 5.0])
loss = criterion(predictions, targets)
print(loss.item())
```

<br>

## **4. Optimisation**
Les **optimiseurs** servent à **ajuster les paramètres** (poids et biais) d'un modèle de machine learning ou de deep learning afin de minimiser une **fonction de perte**. Ils jouent un rôle central dans la phase d'entraînement des modèles en effectuant la mise à jour itérative des paramètres pour améliorer les performances du modèle.

L'objectif principal d'un optimiseur est de résoudre le problème d'optimisation suivant :
$$\min_{\theta} L(\theta)$$
où :
- $\theta$ représente les **paramètres** du modèle (poids et biais).
- $L(\theta)$ est la **fonction de perte** qui mesure l'erreur du modèle.

Pour cela, les optimiseurs utilisent la **descente de gradient** ou ses variantes. La descente de gradient consiste à ajuster $\theta$ dans la direction opposée au gradient de la perte $\nabla L(\theta)$, car cette direction réduit la perte.

---

### **Fonctionnement général d'un optimiseur**
1. **Calcul du gradient** : 
   - Le gradient de la fonction de perte par rapport aux paramètres $\theta$ est calculé grâce à la rétropropagation.
   - Ce gradient indique dans quelle direction et de combien il faut ajuster les paramètres pour réduire l'erreur.

2. **Mise à jour des paramètres** : 
   - Les paramètres sont mis à jour en suivant la règle :
     $$\theta = \theta - \eta \cdot \nabla L(\theta)$$
     où :
     - $\eta$ est le **taux d'apprentissage** (*learning rate*), qui contrôle l'amplitude des mises à jour.
     - $\nabla L(\theta)$ est le gradient.

---

### **Différents types d'optimiseurs**
Voici quelques optimiseurs populaires, chacun ayant ses avantages et ses inconvénients.

#### **1. Stochastic Gradient Descent (SGD)**
Le **gradient descent stochastique** met à jour les paramètres en utilisant un **échantillon aléatoire** des données à chaque itération, ce qui rend les mises à jour rapides et peu coûteuses.

- **Formule de mise à jour** :
  $$\theta = \theta - \eta \cdot \nabla L(\theta)$$
- **Caractéristiques** :
  - Simplicité et efficacité.
  - Sensible au choix du taux d'apprentissage ($\eta$).
  - Peut osciller autour du minimum en raison de la nature stochastique.

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

#### **2. Momentum**
Le **momentum** est une amélioration de SGD. Il accélère les mises à jour dans les directions consistantes et réduit les oscillations.

- **Formule de mise à jour** :
  $$v_t = \beta \cdot v_{t-1} + \nabla L(\theta)$$
  $$\theta = \theta - \eta \cdot v_t$$
  où $\beta$ est un facteur de "momentum" ($0 < \beta < 1$).

- **Caractéristiques** :
  - Meilleure convergence que le SGD pur.
  - Réduit les oscillations dans des ravins étroits de la surface d'erreur.

#### **3. RMSProp (Root Mean Square Propagation)**
RMSProp adapte le taux d'apprentissage pour chaque paramètre en divisant le gradient par une moyenne glissante de ses valeurs passées.

- **Formule de mise à jour** :
  $$E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2$$
  $$\theta = \theta - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \cdot g_t$$

- **Caractéristiques** :
  - Bien adapté pour les problèmes non convexes et les réseaux profonds.
  - Convergence plus stable grâce à l'ajustement dynamique des taux d'apprentissage.

#### **4. Adam (Adaptive Moment Estimation)**
Adam combine les avantages de **Momentum** et **RMSProp**, en utilisant des moyennes glissantes des gradients (premier moment) et de leurs carrés (second moment).

- **Formule de mise à jour** :
  $$m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) g_t$$
  $$v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\hat{m_t} = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v_t} = \frac{v_t}{1 - \beta_2^t}$$
  $$\theta = \theta - \frac{\eta}{\sqrt{\hat{v_t}} + \epsilon} \cdot \hat{m_t}$$

- **Caractéristiques** :
  - Convergence rapide.
  - Ajustement dynamique des taux d'apprentissage pour chaque paramètre.
  - Très populaire pour les réseaux neuronaux profonds.

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

#### **Comparaison des optimiseurs**
| **Optimiseur** | **Avantages**                                   | **Inconvénients**                          |
|----------------|-----------------------------------------------|--------------------------------------------|
| **SGD**        | Simple, efficace, faible coût.                | Convergence lente, oscille autour du minimum. |
| **Momentum**   | Convergence plus rapide que SGD.              | Besoin d'un hyperparamètre supplémentaire ($\beta$). |
| **RMSProp**    | Taux d'apprentissage adaptatif, efficace pour les réseaux profonds. | Moins performant sur certains problèmes convexes. |
| **Adam**       | Rapide, taux adaptatifs, fonctionne bien en pratique. | Peut ne pas converger vers le minimum global. |


#### Exemple d’optimisation:
```python
optimizer.zero_grad()  # Réinitialiser les gradients
predictions = model(x)  # Passer les données dans le modèle
loss = criterion(predictions, y)  # Calculer la perte
loss.backward()  # Calculer les gradients
optimizer.step()  # Mettre à jour les poids
```




<br>

## **5. Gestion de l’Apprentissage**

### **5.1 Boucle d’Entraînement**
Voici une boucle typique:
```python
trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(validset, batch_size=512, shuffle=True, num_workers=2)
criterion = nn.CrossEntropyLoss()

def validation(neuralNetwork):
    valid_loss = 0
    with torch.no_grad():
        for data in validloader:
            images, labels = data
            outputs = neuralNetwork(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
    return valid_loss

def train(neuralNetwork):
    optimizer = torch.optim.SGD(neuralNetwork.parameters(), lr=0.1, momentum=0.9)
    train_history = []
    valid_history = []
    for epoch in range(30):
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = neuralNetwork(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        valid_loss = validation(net)
        train_history.append(train_loss)
        valid_history.append(valid_loss)
        print('Epoch %02d: train loss %0.5f, validation loss %0.5f' % (epoch, train_loss, valid_loss))
    return train_history, valid_history
```

Il est possible de visualiser les performances d'entraînement et de validation d'un modèle au cours des époques:
```python
def plot_train_val(train, valid):
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_ylabel('Training', color=color)
    ax1.plot(train, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation', color=color)
    ax2.plot(valid, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

plot_train_val(train_history, valid_history)
```

### **5.2 Early Stopping**
L'arrêt précoce consiste à interrompre l'entraînement si la performance sur les données de validation ne s'améliore pas.
```python
def train(net, earlystopping=True):
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    train_history = []
    valid_history = []
    if earlystopping:
        estop = EarlyStopping(patience=2)

    for epoch in range(30):
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        valid_loss = validation(net)
        train_history.append(train_loss)
        valid_history.append(valid_loss)
        print('Epoch %02d: train loss %0.5f, validation loss %0.5f' % (epoch, train_loss, valid_loss))

        if earlystopping:
            estop.step(valid_loss)

        if earlystopping and estop.early_stop:
            break

    return train_history, valid_history
```

---

### **5.3 Gestion du Surapprentissage**
- **Dropout**: Introduire de la régularisation avec `nn.Dropout`.
- **Batch Normalization**: Normaliser les sorties des couches cachées.
```python
batch_norm = nn.BatchNorm1d(num_features=64)  # Pour données tabulaires
```

<br>

## **6. Chargement des Données**
PyTorch facilite la gestion des données via `torch.utils.data`:
```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

<br>

## **7. Exemple complet**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Préparation des données
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalisation : moyenne 0.5, écart-type 0.5
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. Définition du modèle
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)  # Couche fully connected
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten (batch_size, 28*28)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()

# 3. Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 4. Boucle d'entraînement
num_epochs = 50
patience = 5  # Nombre d'époques à attendre avant d'arrêter si aucune amélioration
best_loss = float('inf')
patience_counter = 0

train_losses = []
test_losses = []

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(data_loader), correct / total

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    test_loss, test_accuracy = evaluate(model, test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}]\t" \
          f"Train Loss: {train_loss:.4f}\tTest Loss: {test_loss:.4f}\tTest Accuracy: {test_accuracy:.4f}")

    # Early Stopping Logic
    if test_loss < best_loss:
        best_loss = test_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"EarlyStopping counter: {patience_counter}/{patience}")
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# 5. Affichage des courbes d'erreur
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Courbes de perte (Train/Test)')
plt.show()
```