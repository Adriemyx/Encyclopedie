<h1 align='center'> R </h1>

**R** est un langage de programmation et un logiciel libre destiné aux statistiques et à la science des données soutenu par la *R Foundation for Statistical Computing*. Il fait partie de la liste des paquets GNU3 et est écrit en C, Fortran et R.

Le langage R est largement utilisé par les statisticiens, les *data miners*, *data scientists* pour le développement de logiciels statistiques et l'analyse des données. 

___


<h2 align='center'> I. Initiation à R </h2> 
<h3 align='center'> 1. Structure de données </h3>

- Calculatrice
  Il est possible d'excecuter des cellules de calcul:   
   ```r
   # Opérations élémentaires
    2 + 2

    # Fonctions usuelles
    exp(10)

    # Booléens
    5 < 2
   ``` 

- Affectation de varaiables
  Pour sauvegarder une valeur dans une variable, il faut utiliser l’opérateur <code>&lt;-</code>: 
   ```r
    a <- log(2)
    b <- 10
    a + b
   ``` 
  retourne `10.6931471805599`.

- Chaînes de caractères
  Grâce à la fonction `cat`, il est possible de concaténer les chaînes de caratère est d'imprimer le résultat:
   ```r
    nom <- "Adrien"
    cat("Mon nom est", nom)
   ``` 
  retourne 'Mon nom est Adrien'.

- Vecteurs
  Pour créer un vecteur, il est possible d'utiliser la fonction `c`:
   ```r
    vec_1 <- c(1, 2, 5, 8, 12)
    vec_2 <- c("Lundi", "Jeudi", "Mercredi")
   ``` 

   Il est également possible de créer un vecteur de séquence:
   ```r
    vec_1 <- 1:10
    print(vec_1)
   ``` 
    Retourne `[1]  1  2  3  4  5  6  7  8  9 10`.

    De même, 
   ```r
    vec_1 <- seq(from=1, to=10, by=1)
    print(vec_1)
   ``` 
    Retourne `[1]  1  2  3  4  5  6  7  8  9 10`.
    
    
    
    Il est également possible de créer un vecteur par répétition:
    - Soit d'un nombre:
    ```r
    vec_1 <- rep("x", times=5)
    print(vec_1)
    ``` 
    Retourne `[1]  "x" "x" "x" "x" "x"`.

    - Soit d'un vecteur:
    ```r
    vec_1 <- rep(c(2,4), times=6)
    print(vec_1)
    ``` 
    Retourne ` [1] 2 4 2 4 2 4 2 4 2 4 2 4`.
   
   
   
   
   Opérations sur les vecteurs: 
   - Opérations terme à terme
    ```r
    vec_1 <- c(2,3,8,6,7)
    print(vec_1 + 1)
    ``` 
    Retourne `[1]  3 4 9 7 8`.


   - Accès à un élément par sa position
    ```r
    vec_1 <- c(2,3,8,6,7)
    print(vec_1[2])
    ``` 
    Retourne `[1]  3`.

   - Accès à des éléments par tranche
    ```r
    vec_1 <- c(2,3,8,6,7)
    print(vec_1[2:4])
    ``` 
    Retourne `[1]  3 8 6`.

   - Filtre sur les éléments
    ```r
    vec_1 <- c(2,3,8,6,7)
    filtre <- c(FALSE, FALSE, TRUE, FALSE, TRUE)
    vec_1_filtre <- vec_1[filtre]
    print(vec_1_filtre)
    ``` 
    Retourne `[1]  8 7`.

- Matrices
  - Création de matrices:
    ```r
    # Première matrice en colonnes
    matrix_1 <- matrix(1:24, ncol=4)
    print(matrix_1)
    ``` 
    Retourne   
    <pre> 
          [,1] [,2] [,3] [,4]
    [1,]    1    7   13   19
    [2,]    2    8   14   20
    [3,]    3    9   15   21
    [4,]    4   10   16   22
    [5,]    5   11   17   23
    [6,]    6   12   18   24.



    ```r
    # Première matrice en lignes
    matrix_2 <- matrix(1:24, ncol=4, byrow = TRUE)
    print(matrix_2)
    ``` 
    Retourne   
    <pre> 
          [,1] [,2] [,3] [,4]
    [1,]    1    2    3    4
    [2,]    5    6    7    8
    [3,]    9   10   11   12
    [4,]   13   14   15   16
    [5,]   17   18   19   20
    [6,]   21   22   23   24

    - Opérations sur les matrices:
    ```r
    # Opérations terme à terme
    cat("*** Somme :\n")
    print(matrix_1 + matrix_2)
    cat("*** Produit :\n")
    print(matrix_1 * matrix_2)
    ``` 
    Retourne   
    <pre> 
    *** Somme :
          [,1] [,2] [,3] [,4]
    [1,]    2    9   16   23
    [2,]    7   14   21   28
    [3,]   12   19   26   33
    [4,]   17   24   31   38
    [5,]   22   29   36   43
    [6,]   27   34   41   48
    *** Produit :
          [,1] [,2] [,3] [,4]
    [1,]    1   14   39   76
    [2,]   10   48   98  160
    [3,]   27   90  165  252
    [4,]   52  140  240  352
    [5,]   85  198  323  460
    [6,]  126  264  414  576




    ```r
    # Transposition
    matrix_2_t <- t(matrix_2)
    print(matrix_2_t)
    ``` 
    Retourne   
    <pre> 
          [,1] [,2] [,3] [,4] [,5] [,6]
    [1,]    1    5    9   13   17   21
    [2,]    2    6   10   14   18   22
    [3,]    3    7   11   15   19   23
    [4,]    4    8   12   16   20   24


    ```r
    # Produit matriciel
    matrix_product <- matrix_1 %*% matrix_2_t
    print(matrix_product)
    ``` 
    Retourne   
    <pre> 
         [,1] [,2] [,3] [,4] [,5] [,6]
    [1,]  130  290  450  610  770  930
    [2,]  140  316  492  668  844 1020
    [3,]  150  342  534  726  918 1110
    [4,]  160  368  576  784  992 1200
    [5,]  170  394  618  842 1066 1290
    [6,]  180  420  660  900 1140 1380


    ```r
    # Appliquer une fonction sur chaque ligne
    apply(matrix_2, 1, sum)
    ``` 
    Retourne   10 26 42 58 74 90 


    ```r
    # Appliquer une fonction sur chaque colonne
    apply(matrix_2, 1, sum)
    ``` 
    Retourne   66 72 78 84