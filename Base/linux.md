<h1 align='center'> 🐧 Linux commands 🐧 </h1>
Linux ou GNU/Linux est un système d'exploitation open source de type Unix fondé sur le noyau Linux créé en 1991 par Linus Torvalds.

___

<h1 align='center'> I. Linux </h1> 

<h3 align='center'> man </h3> 

La commande `man` pert d'affiche le manuel utilisateur d'un outil.     
Ex:   
   ```bash
   man apt
   ```

Quelques raccourcis utiles pour la navigation dans un manuel:
- <kbd>b</kbd>: pour remonter d'une page dans une page de manuel.
- <kbd>/</kbd>: pour effectuer une recherche dans une page de manuel. ⚠️ sensible à la casse.  
- <kbd>n</kbd>: pour passer à l'occurrence suivante du terme recherché.
- <kbd>N</kbd>: pour passer à l'occurrence précédente du terme recherché.
- <kbd>g</kbd>: pour aller au début d'un page de manuel.
- <kbd>q</kbd>: pour quitter le manuel


<h3 align='center'> ls </h3> 

La commande `ls` est utilisée pour lister le contenu d'un répertoire.   
Quelques flags utiles:
- `-la`: pour lister aussi les fichiers cachés.
- `-lt`: pour afficher en plus la date de modification.
- `-h`: (« human-readable ») pour afficher la taille du fichier dans un format plus compréhensible.



<h3 align='center'> cd </h3> 

La commande ̀`cd` permet de changer de répertoire de travail.   
*<u>Remarque</u>: La commande `cd ~/` permet de se déplacer directement dans le répertoire personnel (`/home/username`), peu importe où l'on est dans la hiérarchie des fichiers.*   



<h3 align='center'> pwd </h3> 

La commande `pwd` ("print working directory") permet d'afficher le chemin d'accès vers le répertoire où se situe l'utilisateur qui a entré la commande.  



<h3 align='center'> ps </h3> 

La commande `ps` (*process status*) permet d'afficher des informations sur les processus en cours d'exécution sur le système. Elle permet notamment d'afficher les PID (identifiants de processus), les statuts, les consommations de ressources telles que la mémoire et le CPU, les utilisateurs propriétaires et bien plus encore.



<h3 align='center'> df </h3> 

La commande `df` (*disk full*) est un utilitaire puissant qui fournit des informations précieuses sur l'utilisation de l'espace disque. La commande df affiche des informations sur l'utilisation de l'espace disque du système de fichiers monté.
```bash
df [options] [filesystems]
``` 



<h3 align='center'> ip </h3> 

La commande `ip` est utilisée pour attribuer une adresse à une interface réseau et/ou configurer les paramètres d'une interface réseau sur les systèmes d'exploitation Linux. Cette commande remplace l'ancienne commande `ifconfig`, désormais obsolète sur les distributions Linux modernes.   
Pour afficher les adresses IP de toutes les interfaces réseau:
```bash
ip a # ou ip addr show
``` 
Cette commande fournit des informations détaillées sur toutes les interfaces réseau et les adresses IP qui leur sont associées, y compris les adresses IPv4 et IPv6.   
L'adresse IP se trouve sur la ligne commençant par *inet*.
<br>

Pour obtenir des informations sur l'adresse MAC de la machine:
```bash
ip link show
``` 
L'adresse MAC se trouve sur la ligne commençant par *link/ether*.

Il est possible que plusieurs lignes commencent par *link/ether* dans la sortie de `ip link show`, cela signifie qu'il y a plusieurs interfaces réseau, chacune ayant sa propre adresse MAC. 



<h3 align='center'> mv </h3> 

La commande ̀`mv` permet de déplacer un élément d'un emplacement spécifié par l’utilisateur.
Ex:
   ```bash
   mv file.txt /path/to/destination_directory
   ```  
*<u>Remarque</u>: Le flag `-i` permet de garantir que le système « demande » si un fichier ou un répertoire doit ou non être écrasé.*  



<h3 align='center'> cat </h3> 

Elle tire son nom du mot concaténer (concatenate en anglais) et permet d'afficher le contenu d'un fichier en sortie standard mais aussi de créer ou de fusionner des fichiers.   
Quelques flags utiles:
- `-n`: pour afficher les numéros de ligne.
- `-v`: pour afficher les caractères invisibles .
 


<h3 align='center'> less </h3> 

La commande ̀`less` est un visionneur de fichiers en mode texte qui permet d'afficher le contenu d'un fichier ou d'une sortie de commande, page par page.  
Ex:
   ```bash
   less file.txt
   ```
*<u>Remarque</u>: Contrairement à `cat`, qui affiche tout le fichier d'un coup, less permet de lire le fichier page par page sans le charger entièrement en mémoire.*   
Pour rechercher du texte dans le fichier, taper `/` suivi du terme de recherche, puis sur Entrée. Utiliser n pour aller à l'occurrence suivante et N pour l'occurrence précédente.
Pour naviguer dans le fichier rapidement:
- **Défilement vers le bas d'un demi-écran**: <kbd>ctrl</kbd> + <kbd>D</kbd> 
- **SDéfilement vers le haut d'un demi-écran**: <kbd>ctrl</kbd> + <kbd>U</kbd> 



<h3 align='center'> cp </h3> 

La commande ̀`cp` permet de copier un ou plusieurs fichiers vers un emplacement spécifié par l’utilisateur. 
Ex:
   ```bash
   cp file.txt /path/to/destination_directory
   ```    
Pour copier un dossier entier:
   ```bash
   cp -r /path/to/source_directory /path/to/destination_directory
   ```



<h3 align='center'> find </h3> 

La commande `find` est utilisée pour rechercher des fichiers.
   ```bash
   find <chemin> <option>
   ```
*<u>Remarques</u>:* 
- *Pour rechercher dans le répertoire courant, on utilise le point « . » comme chemin d’accès au répertoire:*  
   ```bash
   find . <option>
   ```
- *Pour rechercher dans son répertoire personnel, on utilise le tilde « ~ » comme chemin d’accès au répertoire:*  
   ```bash
   find ~ <option>
   ```
- *Pour rechercher dans tout le système, on utilise la barre oblique « / » comme chemin d’accès au répertoire:*
   ```bash
   find / <option>
   ```

Quelques flags utiles:   
- `-type`: pour sélectionner le type de fichier à rechercher:
  - ̀`-f`: pour les fichiers. 
  - ̀`-d`: pour les dossiers. 
- `-name`, `-iname`: pour rechercher par nom de fichier. ⚠️ `-name` est sensible à la casse pas `-iname` (d'où le 'i' de insensible).  
  Ex:
   ```bash
   find . -name .gitignore 
   find . -name "*.jpEg" # Risque de ne rien retourner à cause du 'E'.
   find . -type f -iname "*.jpeg"
   ```
- `and`, `or`, `not`: pour ajouter des cas à la requête.
- `-maxdepth` / `-mindepth`: pour contrôler la profondeur de la recherche dans l'arborescence des répertoires.

*<u>Remarque</u>:* 
*Pour compter le nombre de résultats, on transmet la sortie de la commande find à la commande wc avec l’option « -l »:*
   ```bash
   find <chemin> <option> | wc -l
   ```



<h3 align='center'> wc </h3> 

La commande `wc` permet de compter le nombre de lignes, de mots et de caractères d’un fichier. 
```bash
wc file.txt
```
renvoie le nombre de lignes, de mots et de caractères dans le fichier 'file.txt'. 

- `-l`: pour compter uniquement les lignes au sein d’un fichier.
- `-m`: pour compter uniquement les caractères au sein d’un fichier.
- `-w`: pour compter uniquement les mots au sein d’un fichier.



<h3 align='center'> diff </h3> 

La commande ̀`diff` permet de comparer le contenu de deux fichiers et d'afficher les différences entre eux.
```bash
diff file_a.txt file_b.txt
```
*<u>Remarque</u>: L'ajout du flag `-i` permet de rendre la comparaison insensible à la casse.* 



<h3 align='center'> grep </h3> 

La commande `grep` ("Global Regular Expression Print") est utilisée pour rechercher des lignes qui correspondent à un motif donné. 
   ```bash
   grep [options] [motif] [fichier]
   ```
⚠️ grep est sensible à la casse.   

*<u>Remarque</u>: Comme pour `find`, pour prendre tous les motifs, on utilise le point « . » comme motif:*  
   ```bash
   grep [options] . [fichier]
   ```

Quelques flags utiles:  
- `-v` ou `--invert-match`: pour sélectionner les lignes qui ne correspondent pas au motif.
- `-c` ou `--count`: pour afficher uniquement le nombre de lignes correspondantes.
- `-i` ou `--ignore-case`: pour rendre grep insensible à la casse.
- `-n`: pour afficher les numéros de ligne.
- `-r` ou `--recursive`: pour rechercher récursivement dans les sous-répertoires et non plus uniquement dans un seul fichier.
  Exemple: 
   ```bash
   grep -r "pattern" /path/to/repo
   ```
- `-o` ou `--only-matching`: pour afficher uniquement les parties du texte qui correspondent au motif de recherche, au lieu de toute la ligne où la correspondance a été trouvée.   
  *<u>Remarque</u>: Sans le flag `-o`, si le motif à trouver apparaît plusieurs fois sur la même ligne, cette ligne sera considérée comme une seule occurrence du motif dans le contexte de `wc -l` et donc comptée qu'une seule fois.*
- `-E`: pour ativer l'utilisation des expressions régulières étendues (Extended Regular Expressions, ou ERE), qui permettent d'utiliser des motifs plus complexes et avancés que les expressions régulières de base.

   Sans le flag `-E`, `grep` utilise les **Basic Regular Expressions** (BRE), où certains caractères spéciaux doivent être échappés avec un `\` pour fonctionner. Avec le flag `-E`, ces caractères sont traités comme spéciaux directement, sans avoir besoin d'être échappés.

   Voici les principales différences et pourquoi le flag `-E` est souvent utile :

   | Caractère | **BRE** (sans `-E`)      | **ERE** (avec `-E`)   |
   |-----------|--------------------------|-----------------------|
   | `+`       | Représente un ou plusieurs occurrences, mais doit être échappé : `\+` | Directement utilisable : `+` |
   | `?`       | Représente zéro ou une occurrence, mais doit être échappé : `\?` | Directement utilisable : `?` |
   | `|`       | Représente une alternance, mais doit être échappé : `\|` | Directement utilisable : `|` |
   | `()`      | Parenthèses pour grouper des motifs, doivent être échappées : `\(` et `\)` | Directement utilisables : `()` |

   Ex:
   ```bash
   grep '\(cat\|dog\)' fichier.txt
   grep -E '(cat|dog)' fichier.txt
   ```

Quelques motifs utiles:  
- Chercher des mots ou sous-chaînes qui **commencent** par une lettre spécifique:   
  Ex: rechercher des mots ou sous-chaînes qui commencent par la lettre 'd': 
   ```bash
   grep -E '\bd' fichier.txt
   ```
   *<u>Remarque</u>: `\b` marque une limite de mot (début ou fin d'un mot).*

- Chercher des mots ou sous-chaînes qui **finissent** par une lettre spécifique:   
  Ex: rechercher des mots ou sous-chaînes qui finissent par la lettre 'd': 
   ```bash
   grep -E '\bd' fichier.txt
   ```
   *<u>Remarque</u>: le mot 'sud-ouest' sera compté comme terminant par la lettre 'd'. Pour éviter cela, il faut supprimer les mots contenant 'd-'.*
   ```bash
   grep -E '\bd' fichier.txt | grep -v 'd-'
   ```
   *L'opérateur `|` est un **opérateur de pipe** qui redirige la sortie d'une commande vers l'entrée d'une autre commande. En d'autres termes, il permet de chaîner plusieurs commandes ensemble.*

- Chercher des lignes qui **commencent** par une lettre spécifique:   
  Ex: rechercher des lignes qui commencent par la lettre 'd': 
   ```bash
   grep -E '^d' fichier.txt
   ```
   *<u>Remarque</u>: `^` représente le début d'une ligne.*

- Chercher des lignes qui **finissent** par une lettre spécifique:   
  Ex: rechercher des lignes qui finissent par la lettre 'd': 
   ```bash
   grep -E 'd$' fichier.txt
   ```
   *<u>Remarque</u>: `$` représente la fin d'une ligne.*

- Chercher des sous-chaînes qui contiennent les lettres de l'alphabet:
   ```bash
   grep -E '\b[a-zA-Z]\b' fichier.txt
   ```

- Chercher des sous-chaînes qui contiennent des chiffres de 0 à 9:
   ```bash
   grep -E '\b[0-9]\b' fichier.txt
   ```

- Chercher des sous-chaînes qui contiennent des caractères spécifiques
   ```bash
   grep -E '\b[_!@#$%^&*()-+=]\b' fichier.txt
   ```

- Chercher des mots ou sous-chaînes qui ont un nombre donné de caractères
  Ex: rechercher des chaînes de 5 caractères exactement
   ```bash
   grep -E '\b[a-zA-Z0-9_!@#$%^&*()-+=]{5}\b' fichier.txt
   ```

Exemples particuliers:
- Chercher des mots qui ont exactement 15 caractères, dont les 5e et 10e sont des chiffres, et le reste sont des lettres 
   ```bash
   grep -E '\b[a-zA-Z]{4}[0-9][a-zA-Z]{4}[0-9][a-zA-Z]{5}\b' fichier.txt
   ```

- Trouver des mots qui commencent par "p", finissent par "g", ont une longueur entre 6 et 12 caractères, et qui contiennent au moins deux chiffres
   ```bash
   grep -E '\bp[a-zA-Z0-9]{4,10}g\b' fichier.txt | grep -E '[0-9].*[0-9]'
   ```

- Trouver des mots de 10 caractères qui commencent par "m", contiennent au moins une majuscule et finissent par un chiffre
   ```bash
   grep -E '\bm[a-zA-Z]*[A-Z]+[a-zA-Z]*[0-9]\b' fichier.txt
   ```

- Trouver des mots de 7 caractères qui commencent par une voyelle et finissent par une consonne
   ```bash
   grep -E '\b[aeiouAEIOU][a-zA-Z]{5}[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\b' fichier.txt
   ```



<h3 align='center'> Opérations sur les lignes </h3> 

Pour récupérer les caractères contenus à une ligne spécifique d'un fichier texte, différentes commandes existent:
- `sed`:   
  La commande `sed` (stream editor) est très utile pour extraire des lignes spécifiques.
  Exemple: Extraction de la ligne 82 d'un fichier:
   ```bash
   sed -n '82p' file.txt
   ```
   *<u>Remarque</u>: Le flag `-n` permet de ne pas afficher toutes les lignes par défaut et `82p` permet d'afficher (p) la ligne 82.*

- `awk`:   
  La commande `awk` est également très pratique pour traiter des lignes spécifiques:
  Exemple: Extraction de la ligne 82 d'un fichier:
   ```bash
   awk 'NR==82' file.txt
   ```

- `head` et `tail`:   
  Une combinaison des commandes `head` et `tail` peut également être utilisée pour extraire une ligne spécifique :
  Exemple: Extraction de la ligne 82 d'un fichier:
   ```bash
   head -n 82 file_a.txt | tail -n 1
   ```
   *<u>Remarques</u>:*
   -  *`head -n 82 file_a.txt` : Affiche les 82 premières lignes du fichier.*
   -  *`| tail -n 1 file_a.txt` : Affiche la dernière ligne parmi les 82 premières lignes du fichier selectionnées précedemment.*



<h3 align='center'> variables </h3>   

Pour déclarer une variable, il suffit de l'affecter à un nom de varaible grâce à `=`:
   ```bash
   variable=valeur
   ```
*<u>Remarque</u>:* **Il n'y a pas d'espace entre le nom de la variable et le signe égal, ni entre le signe égal et la valeur.**

*Si un espace est mis l'interpréteur de commandes traitera la variable comme s'il s'agissait d'une commande et, comme cette commande n'existe pas, il affichera une erreur.*  

Ex: 
   ```bash
   name="Adrien MYX"
   age=22
   third_year=("SDD" "COS")
   ```

   ```bash
   echo "Je m'appelle ${name} (mon surnom est echo ${name:0:2}), j'ai ${age} ans. Je suis en filière ${third_year[0]} et en domaine ${third_year[1]}"
   ```

*<u>Remarque</u>: Une variable de shell peut être exportée pour devenir une variable d'environnement grâce à la commande `export`.*   

### Les variables d'environnement:   
Les variables d'environnement sont essentielles pour la configuration et le fonctionnement des systèmes d'exploitation et des applications. Elles permettent de personnaliser l'environnement d'exécution et de gérer les paramètres de manière flexible et centralisée.   
Quelques exemples:
- `PATH`: Contient une liste de répertoires séparés par des deux-points (sur Unix/Linux) ou des points-virgules (sur Windows). Le système recherche des exécutables dans ces répertoires lorsque vous tapez une commande.
- `HOME`: Stocke le chemin vers le répertoire personnel de l'utilisateur.
- `USER` ou `USERNAME`: Contient le nom de l'utilisateur actuel.



<h3 align='center'> cURL </h3>    

La commande `cURL` (abréviation de Client URL) est un outil en ligne de commande utilisé pour transférer des données depuis ou vers un serveur. Elle prend en charge plusieurs protocoles réseau tels que HTTP, HTTPS, FTP, SFTP...   
Voici quelques usages principaux de `cURL` :

a. **Télécharger des fichiers** :
   ```bash
   curl -O http://exemple.com/fichier.txt
   ```

b. **Envoyer des requêtes HTTP** :   
- GET:
   ```bash
   curl https://exemple.com
   ```
- POST: (utile pour pour soumettre des formulaires)
   ```bash
   curl -X POST -d "param1=valeur1&param2=valeur2" http://exemple.com
   ```






<h1 align='center'> II. Linux tools </h1> 

<h3 align='center'> apt </h3>   

**A**dvanced **P**ackaging **T**ool est un système complet et avancé de gestion de paquets, permettant une recherche facile et efficace et surtout une installation simple et une désinstallation propre de logiciels et utilitaires.

```bash
apt [méthode] [paramètres]
```

*<u>Remarque</u>: Certaines méthodes requièrent l'utilisation de la commande `sudo`, d'autres pas, selon qu'elles influent ou non sur les fichiers du système.*   


Quelques méthodes courantes:  
- `install 'package'`: pour installer un paquet.
- `remove 'package'`: pour desinstaller un paquet.
- `purge 'package'`: pour desinstaller un paquet ainsi que ses fichiers de configuration.
- `show 'package'`: pour afficher les détails du paquet.
- `update`: pour mettre à jour la liste des paquets disponibles.
- `upgrade`: pour mettre à jour le système en installant/mettant à jour les paquets.

*<u>Remarque</u>: `apt` est une version plus conviviale et moderne d'`apt-get`, offrant une meilleure interface utilisateur pour la gestion des paquets. Pour un usage quotidien, `apt` est suffisant et recommandé, tandis que `apt-get` reste utile pour les scripts et les tâches plus techniques.*  


The `apt-get` command flag to perform a test run of an installation without actually making any changes to the system is `-s` or `--simulate`.  
This flag allows you to simulate what would happen during the installation, removal, or upgrade process, without actually applying any changes. It's useful to see the effect of the command before executing it.

```bash
apt-get install -s package_name
```



<h3 align='center'> top </h3>  

La commande `top` est utilisée pour afficher les processus Linux actifs. Elle fournit une vue **dynamique en temps réel du système en cours d'exécution**. En général, cette commande affiche les informations récapitulatives du système et la liste des processus ou des fils d'exécution actuellement gérés par le noyau Linux.

*<u>Remarque</u>: La commande `ps` vue précedemment ne fournit qu'une vue statique des processus et affiche un instantané des processus en cours d'exécution à un moment donné.*  


Le nom de la première colonne du tableau retourné par la commande `top` est par défaut PID, qui signifie Process ID.   
Pour tuer un processus, il suffit d'appuyer sur `k` (*kill*) en étant dans `top`, puis taper le numéro du PID à tuer.

La commande `htop` permet également de surveiller les processus en cours d'exécution et les ressources système sous Linux. Cependant, `htop` est plus visuel, convivial, et interactif, avec des fonctionnalités avancées.
En général, `htop` est préféré pour une surveillance plus confortable et intuitive des processus.



<h3 align='center'> tmux </h3>   

La commande `tmux` (terminal multiplexer) est un multiplexeur de terminaux, outil permettant d'exploiter plusieurs terminaux au sein d'un seul et même affichage.    
Pour lancer une session `tmux`:
```bash
tmux
```

La commande permettant de lister toutes les sessions `tmux` en cours est la suivante:
```bash
tmux ls
```

Pour créer une nouvelle session avec un nom spécifique:
```bash
tmux new-session -s <session_name>
```

Pour réutiliser une session existante: 
```bash
tmux attach -t <session_name>
```

Pour renommer une session existante: 
```bash
tmux rename-session -t <old_session_name> <new_session_name>
```

`tmux` fait appel à l'ensemble de touches <kbd>ctrl</kbd> + <kbd>b</kbd>.   
Quelques commandes de base utiles dans une session `tmux`:
- <kbd>ctrl</kbd> + <kbd>b</kbd> suivi de <kbd>c</kbd>: pour créer une nouvelle fenêtre (avec un seul terminal) dans la session `tmux` active.
- <kbd>ctrl</kbd> + <kbd>b</kbd> suivi de <kbd>X</kbd>: pour choisir un terminal spécifique (où X est le numéro du terminal).
- <kbd>ctrl</kbd> + <kbd>b</kbd> suivi de <kbd>,</kbd>: pour renommer la fenêtre actuelle.
- <kbd>ctrl</kbd> + <kbd>b</kbd> suivi de <kbd>n</kbd>: pour se déplacer entre les différents fenêtres de la session.
- <kbd>ctrl</kbd> + <kbd>b</kbd> suivi de <kbd>&</kbd>: pour supprimer la **fenêtre courante**.
- <kbd>ctrl</kbd> + <kbd>b</kbd> suivi de <kbd>w</kbd>: pour afficher les terminaux disponibles.   

Quelques commandes utiles dans un split:
- <kbd>ctrl</kbd> + <kbd>b</kbd> suivi de <kbd>"</kbd>: pour faire une coupe horizontale du terminal courant en deux + ouverture d’un terminal dans le nouveau panel.
- <kbd>ctrl</kbd> + <kbd>b</kbd> suivi de <kbd>%</kbd>: pour faire une coupe verticale du terminal courant en deux + ouverture d’un terminal dans le nouveau panel.
- <kbd>ctrl</kbd> + <kbd>b</kbd> suivi de <kbd>o</kbd>: pour changer entre les panneaux.
- <kbd>ctrl</kbd> + <kbd>b</kbd> suivi de <kbd>(flèches directionnelles)</kbd>: pour se déplacer entre les panneaux.

*<u>Remarque</u>: Il ne faut pas confondre <u>fenêtre</u> et <u>panneau</u>. Chaque fenêtre est un espace de travail complet avec plusieurs panneaux possibles. Une seule session `tmux` peut contenir plusieurs fenêtres et une seule fenêtre peut contenir plusieurs panneaux.*  

L'avantage d'utiliser `tmux` et qu'il est possible d'éxecuter plusieurs tâches sur **un serveur distant** qui continuera de tourner si l'utilisateur se déconnecte.   
Pour quitter le client actuel sans l'arrêter: <kbd>ctrl</kbd> + <kbd>b</kbd> suivi de <kbd>d</kbd>.



<h3 align='center'> vim </h3>  

`vim` est un éditeur de fichier texte pour le terminal sous GNU/Linux. Très complet et peu gourmand en ressources, Vim est une version améliorée du classique Vi. Il est particulièrement adapté pour la coloration syntaxique. Vim est un éditeur modal au contraire de la plupart des éditeurs.   
Voici un tutoriel pour apprendre vim: [vim-adventures](https://vim-adventures.com/)   


Par défaut, `vim` démarre en mode *'Normal'*. Il est utilisé pour naviguer dans le texte, manipuler le texte, et exécuter des commandes.  

Pour ouvrir un fichier avec `vim`: 
```bash
vim file_name
```

*<u>Remarque</u>: Si le fichier n'existe pas, `vim` créera un nouveau fichier avec le nom spécifié et permettra de commencer à éditer ce fichier. Si le fichier existe, `vim` ouvrira le fichier pour que pouvoir le modifier.*     

Quelques commandes de base utiles dans `vim`:
- <kbd>i</kbd>: pour passer en mode insertion.
- <kbd>:</kbd>: pour passer en mode commande.
- <kbd>Esc</kbd>: pour sortir des autres modes et revenir au mode normal.
- <kbd>:wq</kbd> puis <kbd>Enter</kbd>: pour enregistrer les modifications et quitter depuis le mode normal.
- <kbd>:q!</kbd> puis <kbd>Enter</kbd>: pour quitter sans enregistrer depuis le mode normal.


Les commandes de navigation dans `vim` en mode normal:
- <kbd>h</kbd>: pour déplacer le curseur à gauche.
- <kbd>j</kbd>: pour déplacer le curseur vers le bas.
- <kbd>k</kbd>: pour déplacer le curseur vers le haut.
- <kbd>l</kbd>: pour déplacer le curseur à droite.
  

Les commandes de manipulation du texte dans `vim` en mode normal:
- <kbd>dd</kbd>: pour supprimer la ligne courante.
- <kbd>yy</kbd>: pour copier la ligne courante.
- <kbd>p</kbd>: pour coller le texte copié ou coupé.




<!---
<style>
kbd {
    background-color: #eee;
    border: 1px solid #ccc;
    border-radius: 3px;
    box-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    padding: 5px;
    font-family: "Courier New", Courier, monospace;
    color: #333;
}
</style>
-->
