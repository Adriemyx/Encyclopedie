<h1 align='center'> Deep learning - TensorFlow Keras</h1>


**Keras** est une bibliothèque de haut niveau pour le deep learning, construite au-dessus de frameworks comme TensorFlow. Elle est facile à utiliser, flexible et rapide pour développer des modèles de machine learning.   
Pour utiliser Keras, il faut installer TensorFlow (Keras est inclus dans TensorFlow à partir de sa version 2.0):
```bash
pip install tensorflow
```

<br>

## **1. Création du modèle**
Un modèle est créé en ajoutant des couches via l'API séquentielle ou fonctionnelle.

### **API Séquentielle**
Le modèle ci-dessous a 3 couches denses avec ReLU et une couche de sortie Softmax:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Création du modèle
model = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),  # Couche d'entrée
    Dropout(0.5),                                       # Dropout pour réduire le surapprentissage
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')                    # Couche de sortie pour classification
])
```

### **API Fonctionnelle**
L'API fonctionnelle est utilisée pour des modèles complexes comme des réseaux multi-branches:
```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

inputs = Input(shape=(784,))
x = Dense(256, activation='relu')(inputs)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
```

<br>

## **2. Compilation du modèle**
Une fois le modèle défini, il doit être compilé avec un optimiseur, une fonction de perte et des métriques d'évaluation.

```python
model.compile(
    optimizer='adam',                           # Optimiseur
    loss='categorical_crossentropy',            # Fonction de perte
    metrics=['accuracy']                        # Métrique de performance
)
```

> Les principaux paramètres à connaître pour le modèle Keras incluent:   
- **optimizer**: Algorithme utilisé pour l'entraînement. Les choix courants incluent `'sgd'` (Stochastic Gradient Descent), `'adam'` (Adaptive Moment Estimation), `'rmsprop'` (Root Mean Square Propagation) et `'adamax'`.   
- **loss**: Fonction de perte pour évaluer la qualité du modèle. Par exemple:   
    - `'mean_squared_error'` pour la régression.   
    - `'binary_crossentropy'` pour la classification binaire.   
    - `'categorical_crossentropy'` pour la classification multiclass.    
- **metrics**: Listes de métriques pour évaluer le modèle pendant l'entraînement et le test. Exemple: `'accuracy'`, `'precision'`, `'recall'`, `'f1-score'`, `'auc'`.



<br>

## **3. Entraînement et Évaluation**
L'entraînement se fait via la méthode `fit()`, et l'évaluation via `evaluate()`.

### **Entraînement avec validation**
```python
history = model.fit(
    x_train, y_train,
    validation_split=0.2,   # Utiliser 20% des données d'entraînement pour la validation
    epochs=20,              # Nombre d'époques
    batch_size=64,          # Taille des lots
    verbose=1               # Afficher la progression
)
```

### **Évaluation**
```python
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
```

### **Courbes d'apprentissage**
Il est possible de tracer les courbes d'entraînement et de validation pour suivre la performance:
```python
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Courbes de perte')
plt.show()
```

<br>

## **4. Early Stopping**
Keras propose une méthode intégrée pour arrêter l'entraînement si la performance sur les données de validation cesse de s'améliorer.

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',   # Surveiller la perte de validation
    patience=5,           # Nombre d'époques à attendre avant d'arrêter
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    verbose=1,
    callbacks=[early_stopping]  # Ajouter EarlyStopping
)
```

<br>

> Les hyperparamètres clés de l'apprentissage incluent:
- **learning_rate**: Taux d'apprentissage pour la descente de gradient.
- **batch_size**: Nombre d'exemples formés à chaque itération.
- **epochs**: Nombre de passes à travers l'ensemble d'entraînement.
- **dropout**: Pour éviter le surapprentissage, ajoute une couche Dropout qui fixe certains neurones à zéro pendant chaque itération d'entraînement.
- **activation**: Fonction d'activation utilisée dans les couches (e.g., `'relu'`, `'sigmoid'`, `'tanh'`).



<br>

## **5. Prédiction**
Utiliser le modèle pour prédire les classes ou probabilités pour de nouvelles données:

```python
predictions = model.predict(x_test[:5])  # Prédire les 5 premières images
print("Probabilités: ", predictions)
print("Classes prédites: ", predictions.argmax(axis=1))
```

<br>

## **6. Sauvegarde et Chargement du Modèle**
Il est possible de sauvegarder le modèle et le recharger plus tard.

### **Sauvegarde**
```python
model.save('mnist_model.h5')  # Sauvegarde au format HDF5
```

### **Chargement**
```python
from tensorflow.keras.models import load_model

loaded_model = load_model('mnist_model.h5')
```

<br>
<br>

## **Exemple Complet**

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Charger et préparer les données
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Définir le modèle
model = Sequential([
    Dense(256, activation='relu', input_shape=(784,)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Entraîner le modèle
history = model.fit(x_train, y_train, validation_split=0.2, epochs=50, batch_size=64, callbacks=[early_stopping])

# Évaluer le modèle
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Afficher les courbes
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Courbes de perte')
plt.show()
```