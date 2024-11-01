<h1 align='center'> SSH </h1>

**SSH** (Secure Shell) est un protocole de communication sécurisé. Le protocole de connexion impose un échange de clés de chiffrement en début de connexion. Par la suite, tous les segments TCP sont authentifiés et chiffrés. Il devient donc impossible d'utiliser un analyseur de paquets pour voir ce que fait l'utilisateur. 

___


**Exemple d'utilisation:**

Pour se connecter à un serveur distant via SSH:
```bash
ssh user@serveur_ip_address
```

<br>

Il arrive que le serveur distant refuse la connexion:
- Si le refus est lié à une clé privée incorrecte ou manquante (**authentification**):
  Utilisation du flag `-i`: 
    ```bash
    ssh -i /path/to/private_key user@hostname
    ```

- Si le refus est lié à un nom d'utilisateur incorrect (**authentification**):
  Utilisation du flag `-l`: 
    ```bash
    ssh -l correct_user hostname
    ```

<br>

Le port de communication par défaut de SSH est le **22**.


<br>

Une fois authentifié l'utilisateur est connecté au serveur à distance et peut effectuer des commandes sur un pseudoterminal lié au serveur distant.   
Pour sortir de ce pseudoterminal et revenir sur sa machine: `~.`.


<br>

**Depuis le pseudoterminal**, il est possible (entre autres) de:
- déterminer le nom d'hôte du serveur: 
    ```bash
    hostname
    ```
- déterminer le nom du noyau du serveur: 
    ```bash
    uname -s
    ```
- déterminer la version du noyau du serveur: 
    ```bash
    uname -r
    ```
    *<u>Remarque</u>: La commande `uname -a` permet de généraliser et d'afficher toutes les informations disponibles sur le système d'exploitation de la machine distante: Le nom et la version du noyau, le nom d'hôte, l'architecture matérielle...*
- faire toutes les commandes classiques sur linux (grep, find, ls...)