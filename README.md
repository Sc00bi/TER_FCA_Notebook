# TER_FCA_Notebook
Répertoire git destiné à regrouper les documents de notre TER de M1 informatique. 

# Docker

## Utilisation de jupyter dans Docker
### Une fois docker installé

Télécharger une image de ubuntu. Pour voir les images, il suffit de se rendre sur docker hub et saisir dans la bar de recherche ce qu'on veut comme image

	- docker image ls (pour voir les images déjà présentes sur notre système)
	- docker pull ubuntu:latest (pour télécharger une image de ubuntu)


Il serait à présent intéressant de pouvoir lancer une instance de l'image :

	- docker ps (pour voir les processus qui tournent sur docker avant de la création)
	- docker run --rm -t -d --name=<psName> -p 8888:8888 <imageId>
	- docker ps (après de la création)
    
On peut maintenant exécuter l'instance 

	- docker exec -it <psName> <commande> [dans notre cas la commande sera bash. Pour la voir, il suffit de faire docker image ls].

### Mise à jour de Ubuntu et installations

	- apt update (pour mettre à jour Ubuntu)
	- apt  install python3-pip (Installer python3)
	- pip install jupyterlab (Installer Jupyterlab)

D'autres commandes à installer

	- apt install wget 
	- apt install unzip
	- apt install default-jre

### Démarrer Jupyterlab dans le conteneur Ubuntu exécuté
Lancer Jupyter lab :

	- jupyter lab --ip='0.0.0.0' --port=8888 --no-browser --allow-root 
	
Pour utiliser java sur Jupyter, installer Ijava. Mais avant cela il faut s'assurer de la disponibilité de wget et de unzip sur ubuntu.
[Se rendre dans le répertoire home et créer des dossiers comme "documents" et "downloads"]
Dans downloads, faire :

	- wget https://github.com/SpencerPark/IJava/releases/download/v1.3.0/ijava-1.3.0.zip
	
Ensuite :

	- unzip ijava-1.3.0.zip
Enfin :

	- python3 install.py

### Sauvegarde et publication de l'image

	- ctrl+c (pour interrompre le server jupyter-lab)
	- exit (pour sortir de l'espace d'exécution de ubuntu)
	- docker commit <psName or psId> <namespace>/<imageName>:<tag> (pour sauvegarder le container avec nos modifications dans une image)
	- docker login -u <username> [si la commande échoue, cela peut signifier qu'une authentification est nécessaire]
	- docker push <namespace>/<imageName>:<tag>
	- docker stop <psName> (pour détruire l'instance précédente) [NE DETRUIRE L'INSTANCE QU'APRES SAUVEGARDE sinon la progression sera perdue]

## Téléchargement de l'image Docker du projet
Commande utilisée afin d'obtenir l'image Docker (dernière version):

	- docker pull seniorrinnegan/ter_fca_notebook:v1

