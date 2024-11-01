<h1 align='center'> 🐋 Docker 🐋 </h1>

**Docker** est une plateforme permettant de lancer certaines applications dans des <u>conteneurs</u> logiciels lancée. Docker est un outil qui peut empaqueter une application et ses dépendances dans un conteneur isolé, qui pourra être exécuté sur n'importe quel serveur.    
**Il ne s'agit pas de virtualisation, mais de conteneurisation**, une forme plus légère qui s'appuie sur certaines parties de la machine hôte pour son fonctionnement. Cette approche permet d'accroître la flexibilité et la portabilité d’exécution d'une application, laquelle va pouvoir tourner de façon fiable et prévisible sur une grande variété de machines hôtes, que ce soit sur la machine locale, un cloud privé ou public, une machine nue, etc. 

___

<h1 align='center'> I. Introduction </h1> 

En Docker, la différence principale entre une **image** et un **conteneur** est la suivante :

### 1. L'image
Une image Docker est un **modèle statique**. Elle contient tout ce qui est nécessaire pour exécuter une application, comme le système d'exploitation, les bibliothèques, les dépendances, et le code de l'application. Une image est donc une sorte de "photo" **immuable** d'un environnement qui peut être réutilisée pour créer plusieurs conteneurs. Elle ne peut pas être modifiée pendant son exécution, et elle est souvent stockée dans un registre (comme *Docker Hub*) pour être téléchargée ou partagée.

### 2. Le conteneur
Un conteneur est une **instance en cours d'exécution** d'une image Docker. Lorsqu'une image est démarrée, elle devient un conteneur, qui est un environnement isolé et en fonctionnement. Contrairement aux images, les conteneurs sont dynamiques: ils peuvent stocker des données et subir des changements d’état (écriture de fichiers, modification des variables d’environnement, etc.). Chaque conteneur est indépendant et peut être démarré, arrêté, ou supprimé.

### Exemples d'utilisation
- **Image** : Une image de base de données MySQL prête à l'emploi.
- **Conteneur** : Une instance active de MySQL démarrée à partir de cette image, configurée et contenant des données spécifiques.

<br>

<h1 align='center'> II. Utilisation d'images</h1> 

Voici dans ce chapitre quelques commandes Docker utiles ainsi que leur signification:
- `docker pull img`: permet de récupérer l'image *img* depuis le registre Docker et l'enregistre dans le système de l'utilisateur.
- `docker images`: permet de voir la liste de toutes les images présentes sur son système.
- `docker run img`: permet de lancer un conteneur Docker basé sur l'image *img*.
  Lorsque la commande `run` est appellée:
  1. Le client Docker contacte le démon Docker.
  2. Le démon Docker vérifie dans le magasin local si l'image (*img* dans ce cas) est disponible <u>localement</u>, et si ce n'est pas le cas, il la <u>télécharge</u> depuis le magasin Docker. (Si la commande `docker pull img` a déjà été appellée, l'étape de téléchargement n'est pas nécessaire).
  3. Le démon Docker crée le conteneur, puis exécute une commande dans ce conteneur.
  4. Le démon Docker transmet la sortie de la commande au client Docker.
    Ex:
    `docker run img ls`: ici la commande `ls` est donnée à exécuter. Cela affichera la liste des fichiers et répertoires présents dans le répertoire de travail par défaut de l'image *img*.   
    `docker run img echo "hello from img"`: ici la commande `echo "hello from img"` est donnée à exécuter. Cela affichera "hello from img" dans le terminal.

    *<u>Remarque</u>: Le client Docker exécute la commande dans le conteneur, puis le quitte. (C'est équivalent à démarrer une machine virtuelle, exécuter une commande, puis la tuer.)*   
    <br>

    Quelques options utiles avec la commande `docker run`:
    - `-it`: permet d'exécuter les commandes dans un terminal interactif, pour que le shell ne se termine pas après l'éxecution.
    - `-d`: permet de démarrer un conteneur en arrière-plan (mode *détaché*). Cela permet au terminal de rester disponible pour d'autres commandes, sans être "bloqué" par l'exécution du conteneur.
    - `-P`: permet de publier tous les ports exposés d'un conteneur vers des ports aléatoires disponibles sur l'hôte. Elle est utile pour rendre accessibles les ports définis dans le Dockerfile avec la commande EXPOSE, sans avoir à spécifier manuellement chaque port.
    - `-e`: permet de définir des variables d'environnement pour un conteneur. Elle est utilisée pour passer des informations de configuration, comme des identifiants, des chemins ou d'autres paramètres, directement dans l'environnement du conteneur au moment de son lancement.
    Ex: `docker run -d -e NOM_UTILISATEUR=user -e MOT_DE_PASSE=secret img`
    - `--name`: permet de définir un nom de conteneur (si jamais l'original est trop long ou compliqué).

    <br>

- `docker ps`: permet d'afficher tous les conteneurs **en cours d'exécution.**
  *<u>Remarque</u>: L'option `-a` vous permet de voir la liste de tous les conteneurs exécutés: `docker ps -a`.*

- `docker stop <container-id>`: permet d'arrêter le conteneur.
- `docker rm <container-id>`: permet de supprimer le conteneur du système.
  *<u>Remarque</u>: Le <container-id> est visible via la commande `docker ps` ou `docker ps -a`.*

<br>

<h1 align='center'> III. Création d'images</h1> 

Pour créer une image Docker, il faut écrire un *Dockerfile*. Voici une présentation des quelques commandes de base à utiliser dans un Dockerfile:

- `FROM` : permet de commencer le Dockerfile. Il est obligatoire que le Dockerfile débute par la commande `FROM`. Les images sont créées en couches, ce qui signifie qu'il est possible d'utiliser une autre image comme image de base pour. La commande `FROM` définit la couche de base et prend en argument le nom de l'image. Il est possible d'ajouter en option le nom d'utilisateur Docker Cloud, du mainteneur ainsi que la version de l'image, au format `nom_utilisateur/nom_image:version`.

- `RUN` : permet de construire l'image. Pour chaque commande `RUN`, Docker exécute la commande puis crée une nouvelle couche de l'image. Cela permet de revenir facilement à un état antérieur de l'image. La syntaxe pour une instruction `RUN` est de placer la commande complète du shell après `RUN` (par exemple, `RUN mkdir /user/local/foo`). Par défaut, la commande s’exécute dans un shell `/bin/sh`, mais il est possible de définir un autre shell ainsi : `RUN /bin/bash -c 'mkdir /user/local/foo'`.

- `COPY` : permet de copier les fichiers locaux dans le conteneur.

- `CMD` : permet de définir les commandes qui seront exécutées au démarrage de l'image. Contrairement à `RUN`, cela ne crée pas de nouvelle couche pour l'image, mais exécute simplement la commande. **Il ne peut y avoir qu'une seule instruction `CMD` par Dockerfile/image**. S'il faut exécuter plusieurs commandes, la meilleure façon de le faire est d'utiliser `CMD` pour exécuter un script. `CMD` nécessite d'indiquier l'emplacement où exécuter la commande, contrairement à `RUN`. Par exemple, des commandes `CMD` pourraient être :
  `CMD ["python", "./app.py"]` ou `CMD ["/bin/bash", "echo", "Hello World"]`

- `ENTRYPOINT`: permet de définir le binaire ou le script principal qui doit toujours s’exécuter, même si l’utilisateur fournit des arguments supplémentaires. Il rend l’image plus directive et moins flexible dans l’usage.
- `EXPOSE`: permet de créer une indication pour les utilisateurs de l’image sur les ports qui fournissent des services. Cette information est incluse dans les détails que l’on peut récupérer avec `docker inspect <container-id>`.

    *<u>Remarque</u>: La commande `EXPOSE` ne rend pas réellement les ports accessibles à l’hôte! Pour cela, il est nécessaire de publier les ports en utilisant l’option `-p` lors de l’exécution de `docker run`.*

- `PUSH` : permet d'envoyer une image vers Docker Cloud ou, en alternative, vers un registre privé.

    *<u>Remarque</u>: : Pour en savoir plus sur les Dockerfiles: [best-practices](https://docs.docker.com/build/building/best-practices/).*