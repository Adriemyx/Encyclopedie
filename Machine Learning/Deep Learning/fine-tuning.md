<h1 align='center'> Fine-Tuning pour le DeepLearning </h1>

Le **Fine-Tuning** est une technique d'apprentissage profond qui consiste à ajuster un modèle pré-entraîné sur une nouvelle tâche spécifique.  


## 1. Freezing & Unfreezing Progressif
- **Phase 1**: Geler toutes les couches du modèle sauf les couches de classification.  
- **Phase 2**: Débloquer progressivement certaines couches du réseau et fine-tuner avec un LR plus faible.  

Cela permet d'empêcher d'apprendre des poids inutiles au début et de conserver les features pré-entraînées.  

### **Implémentation (TensorFlow/Keras)**
```python
# Phase 1: Geler toutes les couches du modèle
base_model.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=5)

# Phase 2: Débloquer certaines couches et fine-tuner
base_model.trainable = True
for layer in base_model.layers[:100]:  # Geler les 100 premières couches
    layer.trainable = False

# Diminuer le LR pour éviter d'endommager les poids pré-entraînés
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=5)
```

<br>


## 2. Ajustement du Learning Rate (LR) et Schedulers
Le **learning rate** (taux d’apprentissage) détermine à quelle vitesse le modèle ajuste ses poids.  
- **Trop grand** 🏃💨 → Convergence instable, risque d’oscillations.  
- **Trop petit** 🐢 → Convergence très lente, risque de rester bloqué dans un minimum local.  

Dans le **fine-tuning**, il est commun d'utiliser un **LR plus faible** que celui utilisé lors du pré-entraînement.    
Un *scheduler* ajuste **dynamiquement** le LR pendant l'entraînement pour améliorer la convergence.  

###  **2.1. ReduceLROnPlateau (Adaptatif)**
- Réduit le LR lorsque la validation loss **cesse de diminuer**.  
- Évite un LR trop élevé lorsque le modèle stagne.  

**🔹 Implémentation (TensorFlow/Keras):**  
```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Réduit le LR si la validation loss ne diminue pas après 3 epochs
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# Ajout dans model.fit()
model.fit(train_generator, validation_data=val_generator, epochs=20, callbacks=[lr_scheduler])
```

**🔹 Implémentation (PyTorch):**  
```python
import torch.optim.lr_scheduler as lr_scheduler

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

for epoch in range(20):
    train_loss = train_one_epoch()  # Fonction d'entraînement
    val_loss = validate()  # Fonction de validation
    
    scheduler.step(val_loss)  # Ajuste le LR si val_loss stagne
```

---

### **2.2. StepLR (Décroissance en paliers)**
- Diminue le LR après un nombre d'epochs fixe (ex. chaque 10 epochs).  
- Simple mais efficace !  

**🔹 Implémentation (PyTorch):**  
```python
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

for epoch in range(30):
    train_one_epoch()
    scheduler.step()  # Diminue le LR toutes les 10 epochs
```

---

### **3. Cosine Annealing (Scheduler avancé)**
- **Diminution progressive** du LR en suivant une courbe cosinus.  
- Utilisé souvent avec **SGD + Momentum**.  

**🔹 Implémentation (PyTorch):**  
```python
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

for epoch in range(50):
    train_one_epoch()
    scheduler.step()  # LR suit une courbe cosinus
```

<br>

## **3. Utilisation du Dropout**
- Technique qui **désactive aléatoirement** des neurones pendant l'entraînement.  
- Empêche le réseau de **trop s’adapter** aux données d'entraînement (**overfitting**).  
- Généralement utilisé **avant les couches fully connected**.

**Dans le fine-tuning, on peut augmenter le dropout pour compenser le risque d’overfitting sur peu de données.**  

**Implémentation (TensorFlow/Keras)**
```python
from tensorflow.keras.layers import Dropout

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Désactive 50% des neurones
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)  # Désactive 30% des neurones
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
```

**Implémentation (PyTorch)**
```python
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, base_model):
        super(CustomModel, self).__init__()
        self.base = base_model
        self.dropout = nn.Dropout(0.5)  # Dropout de 50%
        self.fc = nn.Linear(512, 2)  # Dernière couche pour classification

    def forward(self, x):
        x = self.base(x)
        x = self.dropout(x)  # Appliquer le dropout
        x = self.fc(x)
        return x

model = CustomModel(models.resnet18(pretrained=True))
```


<br>

## 4. Label Smoothing
- **Empêche le modèle d’être trop sûr de ses prédictions.**  
- Modifie légèrement les labels:  
  - Au lieu d’avoir **1** pour la classe correcte, on met **0.9**.  
  - Au lieu d’avoir **0** pour les autres classes, on met **0.1 / (nombre de classes - 1)**.  

**Implémentation (PyTorch)**
```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

<br>

## 5. Data augmentation
La *data augmentation* (augmentation des données) est une technique utilisée pour améliorer la performance des modèles d'apprentissage automatique en augmentant artificiellement la taille et la diversité du jeu de données d’entraînement. Elle consiste à appliquer diverses transformations sur les images originales, ce qui permet d'éviter le surapprentissage (overfitting) et de rendre le modèle plus robuste face aux variations des données réelles.

Les transformations classiques incluent:
- **Rotation**: Faire tourner l'image d'un certain angle.
- **Zoom**: Appliquer un zoom avant ou arrière sur l'image.
- **Translation**: Déplacer l'image horizontalement ou verticalement.
- **Miroir**: Retourner l'image horizontalement ou verticalement.
- **Perturbation de couleur**: Modifier la luminosité, le contraste ou la saturation des images.

Ces transformations permettent au modèle d'apprendre à reconnaître les objets sous différentes perspectives et conditions, rendant ainsi le modèle plus généraliste et mieux adapté à des données réelles. L'utilisation de la data augmentation est particulièrement bénéfique lorsqu'il y a peu de données d’entraînement disponibles.


### Implémentation (PyTorch)
```python
import torch
from torchvision import transforms
from PIL import Image

# Définition des transformations
data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Retourner l'image horizontalement
    transforms.RandomRotation(30),      # Rotation aléatoire de l'image entre -30 et 30 degrés
    transforms.RandomResizedCrop(224),  # Recadrage aléatoire de l'image avec mise à l'échelle
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Perturbation des couleurs
    transforms.ToTensor(),              # Conversion de l'image en tensor
])

# Chargement de l'image
image = Image.open('image.jpg')

# Application des transformations
augmented_image = data_augmentation(image)

# Affichage de l'image transformée (optionnel)
import matplotlib.pyplot as plt
plt.imshow(augmented_image.permute(1, 2, 0))  # Permute les dimensions pour l'affichage
plt.show()
```

### Explication des transformations:
1. **RandomHorizontalFlip()**: Retourne l'image horizontalement avec une probabilité de 50%.
2. **RandomRotation(30)**: Fait une rotation aléatoire de l'image entre -30 et +30 degrés.
3. **RandomResizedCrop(224)**: Recadre l'image de manière aléatoire tout en conservant une taille de 224x224 pixels après le recadrage.
4. **ColorJitter()**: Modifie de manière aléatoire la luminosité, le contraste, la saturation et la teinte de l'image pour diversifier les couleurs.
5. **ToTensor()**: Convertit l'image PIL en un tensor PyTorch pour l'entraînement.


<br>

## 6. Mixup & CutMix (Augmentation Avancée)
- **Mixup**: Mélange deux images et leurs labels correspondants.  
- **CutMix**: Remplace une partie d’une image par une autre image du dataset.  

### **Implémentation (TensorFlow/Keras)**
```python
import tensorflow as tf

def mixup(image, label, alpha=0.2):
    batch_size = tf.shape(image)[0]
    weights = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1, 1, 1])
    image_shuffled = tf.random.shuffle(image)
    label_shuffled = tf.random.shuffle(label)
    
    mixed_image = weights * image + (1 - weights) * image_shuffled
    mixed_label = weights * label + (1 - weights) * label_shuffled
    return mixed_image, mixed_label
```

<br>

## 7. Stochastic Weight Averaging (SWA)
- Au lieu de prendre les poids du dernier epoch, **moyenner plusieurs modèles entraînés**.  
- Donne de meilleures performances et une meilleure généralisation.  

### **Implémentation (PyTorch)**
```python
from torch.optim.swa_utils import AveragedModel

swa_model = AveragedModel(model)
```

<br>

## 8. Knowledge Distillation
- **Un gros modèle (teacher)** entraîne un **modèle plus petit (student)** en lui transférant son "savoir".  
- Permet d’utiliser un modèle **léger** avec des performances comparables.  

### **Implémentation**
```python
# Loss avec la sortie du teacher et celle du student
loss = alpha * student_loss + (1 - alpha) * distillation_loss
```


<br>

## 9. **Utilisation de l'Ensemble Learning**
Combiner plusieurs modèles pour améliorer la robustesse des prédictions, car en combinant les prédictions de plusieurs modèles (par exemple, via un *ensemble voting* ou une *moyenne* des prédictions), on peut réduire la variance et éviter le sur-apprentissage sur des données spécifiques.

**Implémentation (PyTorch)**: 
```python
class EnsembleModel(nn.Module):
    def __init__(self, model_list):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(model_list)

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

# Usage avec plusieurs modèles
ensemble_model = EnsembleModel([model1, model2, model3])
```

<br>

## 10. **Early Stopping**
Arrêter l’entraînement si la performance sur les données de validation cesse de s’améliorer pendant plusieurs epochs. Cela permet d'éviter l'overfitting en arrêtant l'entraînement avant que le modèle ne commence à surajuster les données d'entraînement.

**Implémentation (TensorFlow/Keras)**:
```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

model.fit(train_data, validation_data=val_data, epochs=50, callbacks=[early_stopping])
```

## 11. **Batch Normalization (BN)**
Ajouter la **normalisation par lots** pour améliorer la stabilité et accélérer la convergence. BN aide à maintenir la variance des activations dans un certain intervalle, ce qui peut améliorer la vitesse de convergence et la stabilité de l'entraînement.

**Implémentation (TensorFlow/Keras)**:
```python
from tensorflow.keras.layers import BatchNormalization

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)  # Ajout de la normalisation par lots
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
```

## 12. **Adversarial Training**
Introduire des **exemples adversariaux** dans l'entraînement pour rendre le modèle plus robuste. Cela permet d'améliorer la capacité du modèle à généraliser en l'exposant à des exemples légèrement modifiés qui peuvent tromper un modèle sans défense.

**Implémentation (PyTorch)**:
```python
def adversarial_training(model, inputs, labels, epsilon=0.1):
    inputs.requires_grad = True
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = inputs.grad.data
    perturbed_data = inputs + epsilon * data_grad.sign()
    return perturbed_data
```

## 13. **Progressive Resizing (Redimensionnement Progressif)**
Commencer l’entraînement avec des images de faible résolution et augmenter progressivement la taille des images pendant l'entraînement. Cela peut améliorer l'efficacité de l'entraînement en réduisant le coût computationnel au début, puis en affinant progressivement les détails à des résolutions plus élevées.

**Implémentation (Keras)**:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Phase 1: Images de basse résolution
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory('data/train', target_size=(128, 128))

# Phase 2: Images de haute résolution
datagen = ImageDataGenerator(rescale=1./255)
train_data = datagen.flow_from_directory('data/train', target_size=(224, 224))
```

## 14. **Self-ensembling (Self-ensemblage)**
Créer plusieurs versions d’un modèle via des techniques comme la *dropout* pendant l'inférence, puis agréger les prédictions. Cela permet de tirer parti de la variance des prédictions et de réduire l'overfitting.

**Implémentation (PyTorch)**:
```python
model.eval()  # Passage en mode évaluation pour utiliser dropout durant l'inférence
outputs = []
for _ in range(5):  # Prédictions multiples
    outputs.append(model(inputs))
final_prediction = torch.mean(torch.stack(outputs), dim=0)
```

### 15. **Hyperparameter Optimization (Optimisation des hyperparamètres)**
Utiliser des techniques comme la **recherche aléatoire** ou les **algorithmes bayésiens** pour optimiser les hyperparamètres du modèle (par exemple, le taux d'apprentissage, le nombre de couches, etc.). L'optimisation des hyperparamètres peut améliorer de manière significative les performances en ajustant les paramètres du modèle à leurs valeurs optimales.

**Implémentation (Hyperopt avec PyTorch)**:
```python
from hyperopt import fmin, tpe, hp, Trials

def objective(params):
    model = YourModel(params['learning_rate'])
    train_loss = train(model)  # Fonction d'entraînement
    return train_loss

space = {
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'batch_size': hp.choice('batch_size', [16, 32, 64])
}

best = fmin(objective, space, algo=tpe.suggest, max_evals=50)
print(best)
```

### 16. **Weighted Loss Function**
Attribuer des poids plus élevés aux classes sous-représentées dans le jeu de données. Cela peut être très utile lorsque les données sont déséquilibrées (par exemple, classification binaire avec un déséquilibre entre les classes).

   **Implémentation (PyTorch)**:
   ```python
   class_weights = torch.tensor([1.0, 5.0])  # Poids plus élevés pour la classe minoritaire
   criterion = nn.CrossEntropyLoss(weight=class_weights)
   ```