<h1 align='center'> 🐍 Python 🐍 </h1>
Python est un langage de programmation interprété, multiparadigme et multiplateformes. Il favorise la programmation impérative structurée, fonctionnelle et orientée objet.    


___

Le but de ce document est de présenter l'installation et l'utilisation de python.


### 1. **Installer Python**  

####  1.1 **Sous Linux**  
Sur Linux, Python est souvent préinstallé, mais pour avoir une version plus récente si nécessaire, voici les commandes pour l'installer.

**a. Vérifier si Python est déjà installé:**
```bash
python3 --version
```
Si la commande retourne quelque chose comme `Python 3.x.x`, Python est déjà installé. Sinon, passer à l'étape suivante.

**b. Installer Python avec apt (pour les distributions basées sur Debian/Ubuntu):**
Il est possible d'installer Python avec `apt`:
```bash
sudo apt update
sudo apt install python3
sudo apt install python3-pip  # Pour installer pip (gestionnaire de paquets Python)
```

**c. Vérifier l'installation:**
Une fois installé, il est possible de vérifier avec:
```bash
python3 --version
```




#### 1.2 **Sous Windows**

**a. Télécharger l'installateur:**
- Aller sur le site officiel de Python: [python.org/downloads](https://www.python.org/downloads/)
- Cliquer sur **Download Python** pour télécharger l'installateur pour Windows.



**b. Lancer l'installateur:**
- Exécuter le fichier `.exe` téléchargé.
- **IMPORTANT:** Il faut s'assurer de cocher l'option **"Add Python to PATH"** avant de cliquer sur "Install Now". Cela rendra Python accessible depuis la ligne de commande.



**c. Vérifier l'installation:**
Une fois l'installation terminée, ouvrir l'Invite de commandes (cmd) et taper:
```cmd
python --version
```
Si tout est bon, cela affichera la version de Python.


### 2. **Installer Conda (Miniconda ou Anaconda)**

#### 2.1 **Sous Linux**

**a. Installer **Anaconda**:**

   1. Aller sur le site de Anaconda: [Télécharger Anaconda](https://www.anaconda.com/products/distribution).
   2. Choisir la version pour Linux (généralement un script `.sh`).

   3. Ouvrir un terminal et naviguer vers le dossier où le fichier a été téléchargé.
   4. Exécuter la commande suivante pour démarrer l'installation:
   ```bash
   bash aconda3-latest-Linux-x86_64.sh
   ```

   5. Accepter les termes de la licence en tapant `yes`, puis choisir l'emplacement d'installation.

   6. Une fois l'installation terminée, redémarrer le terminal ou exécuter cette commande pour activer Conda dans le shell:
   ```bash
   source ~/.bashrc
   ```

   7. Vérifie que `conda` est bien installé avec la commande:
   ```bash
   conda --version
   ```

#### 2.2 **Sous Windows**

**a. Télécharger Anaconda:**

   - **Anaconda:** Aller sur le site d'Anaconda: [Anaconda Download Windows](https://www.anaconda.com/products/distribution) et choisir l'installateur pour Windows.

**b. Installer Anaconda:**

   1. Exécuter le fichier `.exe` téléchargé.
   2. Suivre les étapes de l'installateur. Choisir d'ajouter Anaconda au PATH pour y accéder depuis n'importe quel terminal.
   
**c. Vérifier l'installation:**
   Une fois l'installation terminée, ouvrir **Anaconda Prompt** et taper:
   ```cmd
   conda --version
   ```
   Cela devrait donner la version de Conda installée.



### 3. **Mettre à jour Conda**
Une fois Conda installé, il est recommandé de le mettre à jour.   
Utiliser cette commande pour s'assurer de disposer de la dernière version de Conda:
   ```bash
   conda update conda
   ```

### 4. **Créer un environnement virtuel**
   ```bash
   conda create --name ${env_name} python=${python_version}
   conda activate ${env_name}
   ```

### 5. **Installer des bibliothèques**  
   Maintenant que l'environnement est prêt et activé, il faut installer les bibliothèques nécessaire:
   ```bash
   conda install -c conda-forge ${bib}
   ```

   Il est aussi possible de le faire avec `pip`:
   ```bash
   pip install ${bib}
   ```


### 6. **Créer un fichier `requirements.txt` ou `environment.yml` (optionnel)**  
   Pour rendre le projet reproductible ou pour partager les dépendances avec d'autres personnes, il est utile de générer un fichier `requirements.txt` (via `pip freeze > requirements.txt`) ou un fichier `environment.yml` (via `conda list --export > environment.yml`).
