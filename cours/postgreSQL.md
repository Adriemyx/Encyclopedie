<h1 align='center'> 🐘 postgreSQL 🐘 </h1>

**postgreSQL** est un **système de gestion de base de données relationnelle et objet** (SGBD). C'est un outil libre disponible selon les termes d'une licence de type BSD. Ce système est comparable à d'autres systèmes de gestion de base de données, qu'ils soient libres, ou propriétaires. 

___

<h1 align='center'> I. Installation </h1> 

Pour installer PostgreSQL, il suffit d'entrer la commande:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```   

Ensuite, il est important de vérifier que PostgreSQL est bien lancé:
```bash
sudo service postgresql start
```

Pour vérifier l'état du service:
```bash
sudo service postgresql status
```

<h1 align='center' id="init"> II. Initialisation </h1> 

La procédure d'installation a créé un compte utilisateur appelé **postgres** associé au rôle par défaut de PostgreSQL. Il existe plusieurs façons d'utiliser ce compte pour accéder à PostgreSQL. Une méthode consiste à passer au compte **postgres** sur son serveur en exécutant la commande suivante:
```bash
sudo -i -u postgres
``` 

Une fois cette commande exécutée, l'utilisateur est dans une session en tant que **postgres** et peut exécuter des commandes PSQL directement. Depuis le compte `postgres`, lancer le client en ligne de commande de PostgreSQL:
```bash
psql
```

Dans `psql` créer un nouvel utilisateur avec un mot de passe:
```sql
CREATE USER my_user WITH PASSWORD 'my_pwd';
```

Voici quelques commandes utiles dans `psql`:
- `\l`: permet de voir les bases de données disponibles.
- `\dt`: permet de lister les tables** de la base de données courante.
- `\du`: permet de voir les utilisateurs.
- `\q`: permet de quitter `psql`.



<h1 align='center'> II. Création de bases de données </h1> 

Créer une nouvelle base de données et lui attribuer un propriétaire:
```sql
CREATE DATABASE my_db WITH OWNER my_user;
```


Accorder des privilèges à l’utilisateur:
```sql
GRANT ALL PRIVILEGES ON DATABASE my_db TO my_user;
```

Ensuite, pour se connecter à la base de données avec le nouvel utilisateur, il faut quitter `psql` avec la commande `\q` et se connecteravec le nouvel utilisateur:
```bash
psql -U my_user -d my_db
```

Si une authentification est nécessaire, PostgreSQL demandera le mot de passe.


Dans cette base de données, il faut alors créer une table et insérer des données dans la table:
```sql
CREATE TABLE table (
    ...
    ...
    ...
);

INSERT INTO table (X, Y) VALUES (x1, y1), ..., (xn, yn);
```
ex: 
```sql
CREATE TABLE personnes (
    id SERIAL PRIMARY KEY,
    nom VARCHAR(50),
    age INT
);

INSERT INTO personnes (nom, age) VALUES ('Alice', 30), ('Bob', 25);
```




<h1 align='center'> IV. Accès distant </h1> 

Pour définir un mot de passe pour l’utilisateur **postgres**, ce qui permet de protéger l'accès à l'administration de la base de données et donc empêche les utilisateurs non autorisés d'accéder à ce compte puissant qui a des privilèges d'administrateur.   
Il faut se connecter en tant que **postgres** (cf. [II. Initialisation](#init)) et entrer la commande suivante dans `psql`:

```sql
ALTER USER postgres WITH PASSWORD 'new_pwd';
```

Enfin, pour permettre l'accès distant à PostgreSQL, il faut modifier le fichier de configuration PostgreSQL (`postgresql.conf`). Pour y accéder:
```bash
sudo nano /etc/postgresql/XX/main/postgresql.conf
```

Dans ce fichier, trouver `listen_addresses` et le changer en:
```conf
listen_addresses = '*'
```

Puis, dans le fichier `pg_hba.conf`, ajouter une règle d’autorisation:
```bash
sudo nano /etc/postgresql/XX/main/pg_hba.conf
```

Ajouter la ligne suivante pour autoriser les connexions TCP/IP pour l'utilisateur spécifié depuis toutes les adresses IP :
```conf
host    all             all             0.0.0.0/0               md5
```

Redémarrer PostgreSQL pour appliquer les modifications:
```bash
sudo service postgresql restart
```


*<u>Remarque:</u> **`XX`** dans le chemin doit être remplacé par la version spécifique de PostgreSQL nstallée sur le système. Pour déterminer la version de PostgreSQL installée, il est possible  d'utiliser la commande suivante:*
```bash
psql --version
```
Cette commande affichera la version de PostgreSQL. Par exemple, si la sortie est `psql (PostgreSQL) 14.1`, cela signifie qu'il faut utiliser `14` comme `XX`.