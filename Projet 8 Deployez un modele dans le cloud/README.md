# Déployez un modèle dans le Cloud

## Introduction

Dans ce projet, vous incarnez le rôle d'un Data Scientist dans une très jeune start-up de l'AgriTech, nommée  "Fruits!", qui cherche à proposer des solutions innovantes pour la récolte des fruits.
<br>
La volonté de l’entreprise est de préserver la biodiversité des fruits en permettant des traitements spécifiques pour chaque espèce de fruits en développant des robots cueilleurs intelligents.
<br>
Votre start-up souhaite dans un premier temps se faire connaître en mettant à disposition du grand public une application mobile qui permettrait aux utilisateurs de prendre en photo un fruit et d'en obtenir des informations.
<br>
Pour la start-up, cette application permettrait de sensibiliser le grand public à la biodiversité des fruits et de mettre en place une première version du moteur de classification des images de fruits.
<br>
De plus, le développement de l’application mobile permettra de construire une première version de l'architecture Big Data nécessaire.

## Votre mission

Vous utiliserez un [jeu de données](https://www.kaggle.com/moltean/fruits) constitué d'images de fruits et des labels associés pour tester votre environnement Big Data.
<br>
Vous êtes chargé de mettre en place la chaîne de traitement et il n’est pas nécessaire d’entraîner un modèle pour le moment.
L’important est de mettre en place les premières briques de traitement qui serviront lorsqu’il faudra passer à l’échelle en termes de volume de données !

# Contraintes

Lors de votre travail, vous ferez attention aux points suivants :
- Vous devrez tenir compte dans vos développements du fait que le volume de données va augmenter très rapidement après la livraison de ce projet. Vous continuerez donc à développer des scripts en Pyspark et à utiliser le cloud AWS pour profiter d’une architecture Big Data (EMR, S3, IAM). Si vous préférez, vous pourrez transférer les traitements dans un environnement Databricks
- Vous devez faire une démonstration de la mise en place d’une instance EMR opérationnelle, ainsi qu’ expliquer pas à pas le script PySpark que vous aurez rédigé. Celui-ci devra contenir :
    - un traitement de diffusion des poids du modèle Tensorflow sur les clusters (broadcast des “weights” du modèle). Vous pourrez vous appuyer sur l’article “[Distributed model inference using TensorFlow Keras](https://learn.microsoft.com/en-us/azure/databricks/_static/notebooks/deep-learning/keras-metadata.html)”
    - une étape de réduction de dimension de type PCA en PySpark
- Vous respecterez les contraintes du RGPD : dans notre contexte, vous veillerez à paramétrer votre installation afin d’utiliser des serveurs situés sur le territoire européen

Votre retour critique de cette solution sera également précieuse, avant de décider de la généraliser
La mise en œuvre d’une architecture Big Data de type EMR engendrera des coûts. Vous veillerez donc à ne maintenir l’instance EMR opérationnelle que pour les tests et les démos.