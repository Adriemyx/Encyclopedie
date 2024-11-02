<h1 align='center'> Git </h1>

Git est un système de contrôle de version distribué, gratuit et open source, conçu pour gérer tous les projets, qui aide à suivre les versions et les changements dans le code.

GitHub est une plateforme en ligne qui facilite la collaboration et le partage de projets, tout en utilisant Git comme base. L'intérêt principal de GitHub réside dans ses fonctionnalités sociales et collaboratives, qui permettent aux développeurs du monde entier de contribuer facilement à des projets open-source, de réviser du code via des pull requests, et de gérer des branches multiples pour un développement parallèle.

___


<h1 align='center'> I. Les bases de Github 🐱 </h1> 
<h2 align='center'> 1. Configurations compte Github </h2> 
<h3 align='center'> Création de compte 👤 </h3> 

1. Accéder au site de GitHub:
    Ouvrir son navigateur web et aller sur [github](https://github.com).

2. Cliquer sur *"Sign up"* (s'inscrire):
    Sur la page d'accueil de GitHub, cliquer sur le bouton <kbd>Sign up</kbd> situé en haut à droite.

3. Saisir les informations de base:
    - Adresse e-mail.
    - Mot de passe.
    - Nom d'utilisateur.

4. Cliquer ensuite sur *"Create account"*.
   
5. Effectuer les vérifications de sécurité nécessaires.

<h3 align='center'> Génération d'une clé SSH 🔑 </h3> 

**SSH** (*Secure Shell*) est un protocole de communication sécurisé qui permet d'établir une connexion chiffrée entre deux ordinateurs, généralement pour permettre l'accès à distance et l'administration de serveurs ou d'ordinateurs. SSH est couramment utilisé par les administrateurs systèmes et les développeurs pour se connecter à des machines distantes, exécuter des commandes à distance, transférer des fichiers et gérer des systèmes à distance de manière sécurisée.

**Exemple d'utilisation:**

Pour se connecter à un serveur distant via SSH:
```bash
ssh user@serveur_ip_address
```

<br>

Une clé SSH est un mécanisme d'authentification utilisé dans SSH pour se connecter à des machines distantes sans avoir à utiliser de mots de passe. Une paire de clés SSH est composée de deux parties :   
1.  Clé publique : Elle est stockée sur le serveur distant.
2. Clé privée : Elle reste sur l'ordinateur local et doit être gardée secrète.

Lorsqu'une connexion SSH est initiée, le serveur vérifie si le client a la clé privée correspondante à la clé publique qu'il possède. Si la vérification est réussie, la connexion est établie sans avoir besoin d'un mot de passe.   

Pour générer une paire de clés SSH, il suffit de taper la commande : 
```bash
ssh-keygen -t ed25519 -C "email"
```

Il est également possible de spécifier le nombre de bits lors de la génération d'une clé SSH grâce au flag `-b`. 

<br>

Ensuite pour ajouter la clé privée à l'agent SSH:  
1. Démarrer l'agent ssh en arrière-plan:
    ```bash
    eval "$(ssh-agent -s)"
    ```
2. Ajouter la clé privée SSH à l'agent ssh.
    ```bash
    ssh-add ~/.ssh/id_ed25519
    ```
3. Ajouter la clé publique SSH à son compte sur GitHub:
    - Copier la clé publique SSH dans votre presse-papiers.
        ```bash
        cat ~/.ssh/id_ed25519.pub
        ```
        Ensuite, sélectionner et copier le contenu du fichier id_ed25519.pub affiché dans le terminal dans le presse-papiers.
    - Aller sur son compte Github.
    - Cliquer sur votre photo de profil, puis sur <kbd>Parameters</kbd>.
    - Dans la section *"Access"* de la barre latérale, cliquer sur *"SSH and GPG keys"*.
    - Cliquer sur *"New SSH key"* ou A*"Add SSH key"*.
    - Dans le champ *"key"*, coller la clé publique.
    - Cliquer sur <kbd>Add SSH key</kbd>.



Pour plus d'informations: [generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)


Il est possible de vérifier la connection SSH via la commande: 
```bash
ssh -T git@github.com
```
Si la clé est correctement configurée, un message de confirmation de GitHub devrait apparaitre.



*<u>Remarque</u>: Il est parfois utile de redémarrer l'agent ssh et lui redonner la clé privée avant de pouvoir pousser des modifications.*   



<h3 align='center'> Configurer l'espace de travail local 💻​ </h3> 

Il est nécessaire de configurer son espace de travail pour pouvoir associer les commits à un auteur spécifique. Il est possible de configurer ces informations globalement (pour tous les dépôts Git sur son système) ou localement (pour le dépôt actuel uniquement):   
### Configurer les informations globales:
1. Ouvrir un terminal.
2. Taper les commandes suivantes en remplaçant les valeurs par ses propres informations.
    ```bash
    git config --global user.name "username"
    git config --global user.email "email"
    ```

### Configurer les informations globales:
1. Ouvrir un terminal.
2. Accéder au répertoire du dépôt spécifique.
3. Taper les commandes suivantes en remplaçant les valeurs par ses propres informations.
    ```bash
    git config user.name "username"
    git config user.email "email"
    ```





<h2 align='center'> 2. Travailler sur des répertoires Github </h2> 

<h3 align='center'> Répertoires 📁 </h3> 

Pour créer un nouveau répertoire dans Github, il suffit de:

1. Aller en haut à droite  sur "Create new...".
2. Sélectionner "New repository".

*<u>Remarque</u>: Il est également possible de créer un nouveau répertoire sur Github via des lignes de commandes sur un terminal. En effet, il suffit de taper la commande `git init`.*

Et si le travail à modifier appartient à un autre utilisateur, il faut d'abord le *forker* (le copier dans son propre compte GitHub). 

<h3 align='center'> Fork 🍴 </h3> 

Dans ce cas, il suffit de:   

1. Aller sur la page GitHub du dépôt d'origine.
2. Cliquer sur le bouton <kbd>Fork</kbd> en haut à droite.

Ainsi, GitHub va créer une copie du dépôt dans son compte personnel.

<h3 align='center'> Clone 👬 </h3> 

Un fois que le répertoire est dans le compte personnel, il faut:
1. Ouvrir le répertoire sur Github.
2. Cliquer sur <kbd>Code</kbd>.
3. Copier l'URL du répertoire.
4. Ouvrir un terminal.
5. Se déplacer dans le dossier où cloner le répertorie.
6. Taper la commande suivante: (avec l'URL à coller)
   ```bash
    git clone URL
    ```

Le répertoire sera alors cloné en local.   

*<u>Remarque</u>: Il est possible de cloner un répertoire qui ne se trouve pas dans son espace personnel. Il ne sera juste pas possible de pousser ses modifications personnelles.*


<h3 align='center'> Branches 🌳 </h3> 

Avant d'apporter des modifications, il est recommandé de créer une branche pour organiser son travail:
1. Créer une nouvelle branche localement:
    ```bash
    git checkout -b new_branch_name
    ```
2. Vérifier que l'on est bien sur la nouvelle branche:
    ```bash
    git branch
    ```
    La branche active sera précédée d'une étoile <kbd>*</kbd>.



**À ce stade, il est possible d'apporter les modifications nécessaires aux fichiers du projet en local, via un éditeur de code (ex: VS code).**


<h3 align='center'> Commit 🙌 </h3> 

Un fois les modifications faites, il faut partager ces nouvelles modifications sur son profil Github:
1. Ajouter des fichiers modifiés:
    ```bash
    git add .
    ```
    *<u>Remarque</u>: La commande `git add .` permet d'ajouter tout les fichiers modifiés. Pour ajouter uniquement un fichier spécifique taper la commande `git add 'filename'`.*   
    <br>

    A ce stade, les modifications se trouvent dans la *staging area* (zone de préparation ou index). La staging area permet de sélectionner les modifications que l’on souhaite inclure dans un prochain commit. Elle sert à séparer les fichiers modifiés qui sont prêts à être commités de ceux qui ne le sont pas encore.   
    <br>

2. Enregistrer les modifications via un commit:
   Une fois que les fichiers sont dans la *staging area*, il faut les enregistrer définitivement en créant un commit avec la commande:
    ```bash
    git commit -m "Commit message"
    ```
    À ce moment-là, Git prend une "photo" de l'état actuel de la staging area et crée un commit. Les modifications sont ainsi enregistrés dans l'historique local. 
    <br>

    *<u>Remarque</u>: Grâce au flag `-m`, il est possible d'ajouter un message de commit. Ce message est très utile pour suivre l'historique des modifications dans un projet Github. Il est donc important de ne pas le négliger et de mettre des messages pertinents.*

    <br>

    En Git, chaque commit est identifié par un hash unique, également appelé *"commit ID"*. Il existe deux formats pour ce *commit ID*:
    - La version longue : un hash SHA-1 de 40 caractères.
    - La version courte : une abréviation du début du hash, typiquement les 7 premiers caractères.   
   
        Pour passer du *commit ID* court au *commit ID* long: 
        ```bash
        git rev-parse <short_commit_id>
        ```
        Pour passer du *commit ID* long au *commit ID* court: 
        ```bash
        git rev-parse --short <long_commit_id>
        ```

<h3 align='center'> Push 🫸 </h3> 

Enfin, pour voir le travail modifié arriver sur son répertoire Github en ligne, il faut envoyer les commits vers le dépôt distant:
```bash
git push origin new_branch_name
```


<h3 align='center'> Pull request (PR) </h3> 

Dans le cas où le le travail modifié appartient à un autre utilisateur, il est possible de lui suggérer d'intégrer les modifications apportées via une *pull request*:
1. Aller sur sa page personnelle de son dépôt.
2. Cliquer sur <kbd>Contribute</kbd>.
3. Cliquer sur <kbd>Open pull request</kbd>.

Une fois la PR approuvée et fusionnée, les modifications seront intégrées dans la branche principale du dépôt GitHub d'origine.



*<u>Remarque</u>: Pour se familiariser avec les commandes et l'arborescence de Github, voici un site pour visualiser ces actions: [learngitbranching](https://learngitbranching.js.org/?locale=fr_FR).*




<h1 align='center'> II. Manipulations de commandes Git </h1> 

La commande `git` permet d'interagir avec le système de gestion de versions Git. Certaines de ces commandes ont été vu précedemment comme `clone`, `commit`...   
Voici quelques autres commandes `git` utiles:
- `pull`: permet d'incorporer les modifications d’un dépôt distant dans la branche courante.
  *Par exemple: Si un répertoire est cloné, pour récupérer les modifications ajoutées par le propriétaire du répertoire après le clone, il suffit de faire:*
    ```bash
    git pull
    ```
- `status`: permet d'afficher l'état actuel de l'arborescence de travail et de la *staging area* en aidant à voir les modifications en cours dans le dépôt, notamment celles qui ont été ajoutées à l'index et celles qui ne le sont pas encore.
- `ls-tree`: permet de lister le contenu d'un répertoire ou d'une branche Git. Elle affiche les fichiers et les dossiers associés à un commit spécifique ou à une branche dans Git.
    ```bash
    git ls-tree [options] <branch_name_or_commit_id>
    ```
    Quelques options importantes:
    - `-r`: (récursif), pour lister les fichiers dans tous les sous-répertoires.
    - `-d`: pour ne lister que les répertoires.
- `ls-files`: permet d'afficher tous les fichiers suivis (fichiers ajoutés à l'index) dans le dépôt actuel. Contrairement à `git ls-tree`, elle ne montre pas les répertoires, uniquement les fichiers qui sont suivis dans Git.
    ```bash
    git ls-files
    ```
    *<u>Remarque</u>: Il est possible de rechercher des fichiers selon un pattern spécifique*   
    Exemple: Pour recherche les fichiers ayant un extension .py
    ```bash
    git ls-files "./*.py"
    ```
- `log`: permet d'afficher l'historique des commits d'un dépôt Git. 
    ```bash
    git log [options]
    ``` 
    *<u>Remarque</u>: Les commits sont présentés par ordre **antéchronologique**, c'est-à-dire que les commits les plus récents apparaissent en premier.* 
  Quelques options courantes de `git log`: 
  - `-n`: pour limiter l'affichage à un nombre donné de commits.
  - `-p`: pour afficher les différences apportées par chaque commit.
  - `--reverse`: pour inverser l'ordre d'affichage des commits: **du plus ancien au plus récent**.
  - `--oneline`: pour afficher chaque commit sur une seule ligne avec un hash **abrégé**.
  - `--stat`: pour afficher un résumé des changements pour chaque commit, notamment les fichiers modifiés et combien de lignes ont été ajoutées ou supprimées. 
  - `--graph`: pour afficher un graphe visuel de l'historique des branches et des fusions.
  Il est possible également de filtrer les commits:
  - `--author="Name"`: pour filtrer les commits par auteur.
  - `--since` et `--until`: pour filtrer les commits par date.
    Exemple: 
    ```bash
    git log --since="2 weeks ago"
    git log --until="2023-09-01"
    ```
  - `--grep="keyword"`: pour filtrer les commits par message, en recherchant un mot-clé dans les messages de commit. 
    <br>
  - `--format`: pour personnaliser la manière dont les informations sur les commits sont affichées:
    - `--format="%H"`: pour afficher uniquement les *commit ID* longs.
    - `--format="%h"`: pour afficher uniquement les *commit ID* courts.
    - `--format="%s"`: pour afficher uniquement les messages de commit.
    - `--format="%an"`: pour afficher uniquement les noms des auteurs.
    - `--format="%ae"`: pour afficher uniquement les emails des auteurs.
    - `--format="%ad"`: pour afficher uniquement les dates de commit.
    - ...
- `shortlog`: pour générer un résumé condensé de l'historique des commits, groupé par auteur. Contrairement à `git log`, qui affiche chaque commit en détail, `git shortlog` offre un aperçu plus concis en affichant le nombre de commits par auteur, suivi de la liste des messages de commit correspondants.   
  Quelques flags utiles:
  - `-s`: pour afficher uniquement un résumé du nombre de commits par auteur (sans les messages).
  - `-s`: pour trier les auteurs par nombre de commits (du plus grand au plus petit).
  - `-e`: pour afficher les adresses e-mail des auteurs.
  *<u>Remarque</u>: Le flag `-sne` permet de combiner les 3 options. Donc `git shortlog -sne` permet d'obtenir une vue résumée, triée et avec les e-mails et le nombre de commit des contributeurs d'un dépôt Git.*

- `branch`:  pour lister, créer, renommer ou supprimer des branches dans un dépôt Git:
  - Pour lister les branches:
    ```bash
    git branch
    ```
  - Pour créer un nouvelle branche:
    ```bash
    git branch <new_branch>
    ```
  - Pour renommer un branche locale: 
    ```bash
    git branch -m <new_name>
    ```
  - Pour supprimer un branche: 
    ```bash
    git branch -d <branch_to_delete>
    ```
  - `-r`: (remote), pour lister uniquement les branches distantes (utiles pour suivre l'état du dépôt distant et voir sur quelles branches travailler).
- `checkout`: pour changer de branche dans un dépôt et créer une nouvelle branche (comme vu précédemment). Mais `checkout` permet également de **restaurer des fichiers**
    ```bash
    git checkout <commit_id> -- <fichier>
    ```
    Cette commande permet de ramener le fichier sélectionné à l'état où il était lors d'un commit particulier. 

- `tag`: pour créer, lister, supprimer ou gérer des tags dans un dépôt Git. Un **tag** est un marqueur qui identifie un point spécifique de l'historique d'un projet, généralement utilisé pour marquer des versions importantes, telles que des versions de production ou des jalons. 
  - Pour lister les tags:
    ```bash
    git tag
    ```
  - Pour créer un tag **léger**. Un tag léger est simplement un nom donné à un commit, sans autres métadonnées (comme un message de tag ou la date):
    ```bash
    git tag <tag_name>
    ```
  - Pour créer un tag **annoté**. Un tag annoté est une version plus complète d'un tag, contenant des métadonnées comme l'auteur, la date, et un message de tag. Les tags annotés sont recommandés pour marquer des versions officielles:
    ```bash
    git tag -a <tag_name> -m "Message"
    ```
  - Pour afficher les détails d'un tag annoté: 
    ```bash
    git show <tag_name>
    ```
  - Pour supprimer un tag: 
    ```bash
    git tag -d <tag_name>
    ```
- `remote`: pour gérer les dépôts distants dans Git. (un dépôt distant est une version du projet qui est hébergée sur un serveur, comme GitHub, GitLab...).
  Quelques flags et commandes utiles:
  - `-v`: pour vérifier les URL des dépôts distants.
  - `add`: pour ajouter un dépôt distant:
    ```bash
    git remote add origin https://github.com/username/repository.git
    ```
    Cette commande crée un lien entre votre dépôt local et un dépôt distant.   
    *<u>Remarque</u>: Si le dépôt est issu d'un `git clone`, Git configure déjà le dépôt distant. Il n'est donc pas nécessaire d'utiliser `git remote add`.*   
    *Mais si le dépôt est d'abord créé en local avec puis `git init`, il faut utiliser `git remote add` pour connecter le projet à un dépôt distant, comme GitHub, afin de le pousser en ligne.*
  - `remove`: pour supprimer un dépôt distant:
    ```bash
    git remote remove origin
    ```
- ... 