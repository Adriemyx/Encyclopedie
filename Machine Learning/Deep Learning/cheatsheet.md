### 📌 **1. Perceptron Multicouche (MLP / ANN - Artificial Neural Network)**
- **Utilité** : Réseau de neurones dense entièrement connecté. Base des réseaux profonds.  
- **Cas d'usage** :
  - Classification et régression
  - Prédiction de séries temporelles simples
  - Recommandation et segmentation  
- **Intérêt** :  
  ✅ Simple et efficace pour des données tabulaires  
  ❌ Mauvaise gestion des relations spatiales ou temporelles  

---

### 🎨 **2. Réseaux Convolutifs (CNN - Convolutional Neural Networks)**  
- **Utilité** : Idéal pour traiter des données ayant une structure spatiale (ex : images).  
- **Cas d'usage** :
  - Vision par ordinateur (détection d'objets, reconnaissance faciale)
  - Traitement d’images médicales
  - Analyse de vidéos et de scènes  
- **Intérêt** :  
  ✅ Excellente performance sur les images grâce aux convolutions  
  ❌ Moins performant pour des séquences temporelles longues  

---

### ⏳ **3. Réseaux Récurrents (RNN - Recurrent Neural Networks)**  
- **Utilité** : Adapté aux séquences et aux données temporelles.  
- **Cas d'usage** :
  - Analyse de séries temporelles (prédiction météo, finance)
  - Traitement du langage naturel (NLP)
  - Génération de texte et de musique  
- **Intérêt** :  
  ✅ Garde la mémoire du passé pour contextualiser l’analyse  
  ❌ Problème de disparition du gradient pour des séquences longues  

---

### 🔁 **4. LSTM (Long Short-Term Memory) et GRU (Gated Recurrent Unit)**  
- **Utilité** : Amélioration des RNN pour gérer les longues séquences.  
- **Cas d'usage** :
  - Traduction automatique et synthèse vocale
  - Chatbots et assistants vocaux (Siri, Alexa)
  - Analyse de sentiments  
- **Intérêt** :  
  ✅ Mémorise mieux les longues dépendances  
  ❌ Plus coûteux en calcul que les RNN classiques  

---

### 🎭 **5. Autoencodeurs (AE - Autoencoders & VAE - Variational Autoencoders)**  
- **Utilité** : Compression et génération de données.  
- **Cas d'usage** :
  - Réduction de dimension (compression de données)
  - Dénoising d’images ou de sons
  - Génération de nouvelles données réalistes  
- **Intérêt** :  
  ✅ Utile pour la détection d’anomalies et le transfert de style  
  ❌ Moins interprétable que les autres modèles  

---

### 🔥 **6. GAN (Generative Adversarial Networks)**  
- **Utilité** : Génération de données réalistes à partir de bruit aléatoire.  
- **Cas d'usage** :
  - Création d’images (DeepFake, AI Art)
  - Augmentation de données pour l’apprentissage
  - Génération de visages, de musique, de textes  
- **Intérêt** :  
  ✅ Produits des résultats ultra-réalistes  
  ❌ Instable à entraîner  

---

### ⚡ **7. Transformers (ex: BERT, GPT)**  
- **Utilité** : Traitement avancé des séquences et NLP.  
- **Cas d'usage** :
  - Traduction et résumé automatique
  - Chatbots avancés et IA conversationnelle
  - Analyse de texte (BERT pour la compréhension du langage)  
- **Intérêt** :  
  ✅ Capte mieux le contexte grâce à l’attention  
  ❌ Très coûteux en calcul  
