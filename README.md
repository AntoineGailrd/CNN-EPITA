# README

Ce projet est un classifieur d'images de bateaux realise dans le cadre d'une competition Kaggle (competition-2026-fait-main). Le but est de classifier des images de navires en grille de niveaux de gris parmi 10 categories differentes en utilisant TensorFlow et Keras.

Le script telecharge les donnees grace a kagglehub, prepare les lots d'images avec un decoupage d'entrainement et de validation, et applique des poids aux classes pour gerer le desequilibre des donnees.

Le modele est un reseau de neurones convolutionnel (CNN) sequentiel compose de 21 couches. Il integre directement des fonctions d'augmentation de donnees comme des retournements horizontaux, des translations, des zooms et des contrastes. L'architecture utilise quatre blocs de couches convolutionnelles suivies de normalisation par lots et de max pooling, puis une couche de reduction globale par moyenne, une couche dense avec une fonction d'activation swish, un abandon pour eviter le surajustement, et enfin une couche de sortie avec une fonction softmax pour predire les 10 classes.

L'entrainement utilise l'optimiseur Adam avec un taux d'apprentissage initial de 0.001 et la fonction de perte sparse categorical crossentropy. Des fonctions de rappel sont configurees pour sauvegarder le meilleur modele nomme ship_classifier.h5, arreter l'entrainement tot si la precision ne s'ameliore plus, et reduire le taux d'apprentissage en cas de plateau. Le modele atteint une precision d'environ 92.98 pourcent sur l'ensemble de validation.

A la fin, le code charge les images de test, genere les predictions et produit un fichier de soumission au format CSV nomme test.csv contenant les identifiants et les categories predites.

Pour executer ce projet, il faut installer Python avec les bibliotheques TensorFlow, NumPy et KaggleHub.
