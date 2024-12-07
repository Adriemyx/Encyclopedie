<h1 align='center'> GPU 🖥</h1>


L'utilisation des GPU (unités de traitement graphique) pour accélérer les calculs en Python est devenue courante grâce à l'intégration de CUDA (Compute Unified Device Architecture) de NVIDIA. CUDA permet de tirer parti de la puissance de calcul massive des GPU pour effectuer des tâches qui seraient autrement lentes avec des CPU seuls. Cette technologie est particulièrement utile dans le contexte de l'apprentissage automatique, du traitement d'image, de la vision par ordinateur, du traitement du langage naturel et d'autres domaines nécessitant des calculs intensifs. Il y a donc plusieurs avantages à utiliser des GPU:

- **Performance accrue**: Les GPU offrent des performances bien supérieures aux CPU pour de nombreux types de calculs, notamment les multiplications de matrices, les réseaux de neurones profonds et les autres opérations mathématiques lourdes.
- **Parallélisation**: Les GPU possèdent de nombreux cores de traitement capables d'effectuer des opérations en parallèle, ce qui est idéal pour les calculs vectoriels et matriciels massifs utilisés dans le deep learning.
- **Réduction de la latence**: Utiliser un GPU peut considérablement réduire le temps de calcul, accélérant ainsi le processus d'entraînement et d'inférence pour des modèles complexes.

<br>
<br>

## I. **Prérequis pour l'utilisation de CUDA avec Python**
Avant d'utiliser CUDA avec Python, il faut s'assurer que:
- **CUDA**: Avoir installé CUDA sur votre machine. CUDA est une technologie de NVIDIA permettant aux développeurs d'utiliser le GPU comme unité de calcul parallèle.
- **CuDNN**: Pour le deep learning, il faut avoir également besoin de CuDNN (CUDA Deep Neural Network), qui est une librairie optimisée pour les réseaux neuronaux profonds avec CUDA.
- **Pip**: il faut s'assurer d'avoir `pip` installé pour installer les librairies Python nécessaires.

<br>
<br>

## II. **Installation des librairies Python pour CUDA**
Pour utiliser CUDA avec Python, les principales librairies à installer sont:
- `torch` pour PyTorch
- `tensorflow-gpu` pour TensorFlow
- `cupy` pour des calculs numpy-like accélérés par CUDA
- `pycuda` pour le calcul GPU direct à travers CUDA

Voici un exemple pour installer PyTorch avec CUDA support:
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu11.0
```

<br>
<br>

## III. **Code Exemple avec PyTorch**
L'exemple suivant montre comment utiliser PyTorch avec CUDA pour entraîner un simple réseau de neurones sur un GPU.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Génération de données fictives
X = np.random.randn(1000, 20)  # 1000 échantillons, 20 features
y = (np.random.randn(1000) > 0).astype(int)  # Classes binaires

# Conversion en tensreurs PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.int64)

# Création du jeu de données et dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Définition du modèle
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(20, 50)
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Envoi du modèle sur le GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN().to(device)

# Optimiseur et fonction de perte
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entraînement du modèle
num_epochs = 100
for epoch in range(num_epochs):
    for batch in dataloader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch.float())
        
        # Backward pass et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prédiction sur le GPU
model.eval()
with torch.no_grad():
    outputs = model(X_tensor.to(device))
    predicted = (outputs.squeeze() > 0.5).int()
    accuracy = (predicted == y_tensor).float().mean()
    print(f'Accuracy on GPU: {accuracy:.4f}')
```

## IV. **Interopérabilité avec CPU/GPU**
- PyTorch, TensorFlow et d'autres librairies modernes permettent une transparence totale entre CPU et GPU. Vous pouvez facilement transférer des tensors entre le CPU et le GPU à l'aide de fonctions comme `.to('cuda')` pour transférer vers GPU et `.to('cpu')` pour transférer vers CPU.
- Cela permet d'utiliser un GPU pour les opérations intensives et de basculer vers le CPU pour les tâches nécessitant moins de puissance de calcul.

## V. **Utilisation de `torch.cuda` pour les opérations spécifiques au GPU**
- `torch.cuda.is_available()`: Vérifie si CUDA est disponible sur le système.
- `torch.cuda.get_device_name(0)`: Retourne le nom du GPU disponible.
- `torch.cuda.memory_allocated(device)`: Retourne la mémoire GPU actuellement allouée.
- `torch.cuda.empty_cache()`: Libère la mémoire GPU.