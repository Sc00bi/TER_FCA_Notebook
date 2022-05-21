# TER_FCA_Notebook
Répertoire git destiné à regrouper les documents de notre TER de M1 informatique. Le sujet de notre TER était : FCA Notebook, Pratiquer l'analyse formelle de concepts dans un notebook. Nous avons donc à disposition les répertoires suivants :

	- *FCA4J correspond* à différentes éxecutions avec FCA4J uniquement. Nous avons étudié le contexte formel des données sur Market et sur Animals11.
	- bases_de_donnees répertorie différentes tables valuées que nous avons pu trouver, diviser selon leur format.
	- chaines_de_traitements 

## Docker

### Utilisation de jupyter dans Docker
#### Une fois docker installé

Télécharger une image de ubuntu. Pour voir les images, il suffit de se rendre sur docker hub et saisir dans la bar de recherche ce qu'on veut comme image

	- docker image ls (pour voir les images déjà présentes sur notre système)
	- docker pull ubuntu:latest (pour télécharger une image de ubuntu)


Il serait à présent intéressant de pouvoir lancer une instance de l'image :

	- docker ps (pour voir les processus qui tournent sur docker avant de la création)
	- docker run --rm -t -d --name=<psName> -p 8888:8888 <imageId>
	- docker ps (après de la création)
    
On peut maintenant exécuter l'instance 

	- docker exec -it <psName> <commande> [dans notre cas la commande sera bash. Pour la voir, il suffit de faire docker image ls].

#### Mise à jour de Ubuntu et installations

	- apt update (pour mettre à jour Ubuntu)
	- apt  install python3-pip (Installer python3)
	- pip install jupyterlab (Installer Jupyterlab)

D'autres commandes à installer

	- apt install wget 
	- apt install unzip
	- apt install default-jre

#### Démarrer Jupyterlab dans le conteneur Ubuntu exécuté
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

#### Sauvegarde et publication de l'image

	- ctrl+c (pour interrompre le server jupyter-lab)
	- exit (pour sortir de l'espace d'exécution de ubuntu)
	- docker commit <psName or psId> <namespace>/<imageName>:<tag> (pour sauvegarder le container avec nos modifications dans une image)
	- docker login -u <username> [si la commande échoue, cela peut signifier qu'une authentification est nécessaire]
	- docker push <namespace>/<imageName>:<tag>
	- docker stop <psName> (pour détruire l'instance précédente) [NE DETRUIRE L'INSTANCE QU'APRES SAUVEGARDE sinon la progression sera perdue]

### Téléchargement de l'image Docker du projet
Commande utilisée afin d'obtenir l'image Docker (dernière version):

	- docker pull seniorrinnegan/ter_fca_notebook:v4

#### Arborescence de docker
Les tests on été réalisés à partir du dossier /home/Documents/TER







## Workflow Weka -> Orange -> Scikit learn

### WEKA (IJava)


```bash
%maven nz.ac.waikato.cms.weka:weka-stable:3.8.0
```


```java
import weka.filters.unsupervised.instance.Resample;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;

import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.core.converters.ArffLoader;
```

#### chargement d'un fichier source


```java
// "chargement des données du fichier iris.csv"
Instances data = DataSource.read("iris.csv");
```

#### Resample


```java
Resample sampler = new Resample();

if(sampler.setInputFormat(data))
    System.out.println("ok !");
else 
    System.err.println("erreur");
```

    ok !


#### enregistrement des données dans un fichier csv


```java
// A ce niveau, les données sont au format arff (observer avec data.toString())

/* code pour sauvegarder les données au format arff */
BufferedWriter writer = new BufferedWriter(new FileWriter("./iris.arff"));
writer.write(data.toString());
writer.flush();
writer.close();

// sauvegarde des données au format csv
CSVSaver saver = new CSVSaver();
saver.setInstances(data);
saver.setFile(new File("iris_resample.csv"));
saver.writeBatch();
```

### ORANGE (IJava -> ipykernel)


```python
import Orange
iris_data = Orange.data.Table("iris_resample.csv")
iris_data
```




    [[5.1, 3.5, 1.4, 0.2, Setosa],
     [4.9, 3.0, 1.4, 0.2, Setosa],
     [4.7, 3.2, 1.3, 0.2, Setosa],
     [4.6, 3.1, 1.5, 0.2, Setosa],
     [5.0, 3.6, 1.4, 0.2, Setosa],
     ...
    ]




```python
# Drop avec panda ...
```


```python
iris_data.save("iris_res_drop.csv")
```

### SKLearn


```python
import numpy as np
import pandas as pd
```


```python
# charger le fichier "iris_res_drop.csv" avec panda
df = pd.read_csv('iris_res_drop.csv', header=0)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal.length</th>
      <th>sepal.width</th>
      <th>petal.length</th>
      <th>petal.width</th>
      <th>variety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>continuous</td>
      <td>continuous</td>
      <td>continuous</td>
      <td>continuous</td>
      <td>Setosa Versicolor Virginica</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sous python, les fonctionnalités de Dataframe sons souvent invoquées avec Orange.

df = df.drop(columns=['petal.width'], axis=1)
```


```python
df = df.drop([0,1])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal.length</th>
      <th>sepal.width</th>
      <th>petal.length</th>
      <th>variety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>Setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>150</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>Virginica</td>
    </tr>
    <tr>
      <th>151</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>Virginica</td>
    </tr>
  </tbody>
</table>
<p>150 rows × 4 columns</p>
</div>




```python
# Séparer les features des targets  
data = df.values
X = data[:,0:3]
y = data[:,3]
```


```python
# algorithme SVM
from sklearn.svm import SVC
svn = SVC()
svn.fit(X, y)
```







```python
# Prédiction
predictions = svn.predict(X)

# Calcul de la précision
from sklearn.metrics import accuracy_score
accuracy_score(y, predictions)
```




    0.9466666666666667




```python
# rapport de classification
from sklearn.metrics import classification_report
print(classification_report(y, predictions))
```

                  precision    recall  f1-score   support
    
          Setosa       1.00      1.00      1.00        50
      Versicolor       0.90      0.94      0.92        50
       Virginica       0.94      0.90      0.92        50
    
        accuracy                           0.95       150
       macro avg       0.95      0.95      0.95       150
    weighted avg       0.95      0.95      0.95       150
    





## Workflow Weka -> FCA4J
Utilisation de Weka pour le calcul des règles d'association à partir de l'algorithme Apriori et génération d'un treillis à l'aide de FCA4J

### importation de quelques fonctionnalités de weka

```Java
%maven nz.ac.waikato.cms.weka:weka-stable:3.8.0
```

```Java
import weka.associations.Apriori;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.CSVLoader
```

### chargement du fichier source "market_ordered_itemset.csv"


```Java
// chargement des données du fichier "market_ordered_itemset.csv"
Instances data = DataSource.read("market_ordered_itemset.csv");
```

### génération de règles d'association


```Java
// construction du model
Apriori model = new Apriori();
model.buildAssociations(data);
System.out.println(model);
```

    
    Apriori
    =======
    
    Minimum support: 0.3 (1 instances)
    Minimum metric <confidence>: 0.9
    Number of cycles performed: 14
    
    Generated sets of large itemsets:
    
    Size of set of large itemsets L(1): 6
    
    Size of set of large itemsets L(2): 13
    
    Size of set of large itemsets L(3): 12
    
    Size of set of large itemsets L(4): 4
    
    Best rules found:
    
     1. P4=coffee 3 ==> P3=tea 3    <conf:(1)> lift:(1.25) lev:(0.12) [0] conv:(0.6)
     2. P6=coke 2 ==> P2=milk 2    <conf:(1)> lift:(1.25) lev:(0.08) [0] conv:(0.4)
     3. P6=coke 2 ==> P3=tea 2    <conf:(1)> lift:(1.25) lev:(0.08) [0] conv:(0.4)
     4. P1=bread P4=coffee 2 ==> P3=tea 2    <conf:(1)> lift:(1.25) lev:(0.08) [0] conv:(0.4)
     5. P2=milk P4=coffee 2 ==> P3=tea 2    <conf:(1)> lift:(1.25) lev:(0.08) [0] conv:(0.4)
     6. P3=tea P6=coke 2 ==> P2=milk 2    <conf:(1)> lift:(1.25) lev:(0.08) [0] conv:(0.4)
     7. P2=milk P6=coke 2 ==> P3=tea 2    <conf:(1)> lift:(1.25) lev:(0.08) [0] conv:(0.4)
     8. P6=coke 2 ==> P2=milk P3=tea 2    <conf:(1)> lift:(1.67) lev:(0.16) [0] conv:(0.8)
     9. P5=eggs 1 ==> P1=bread 1    <conf:(1)> lift:(1.25) lev:(0.04) [0] conv:(0.2)
    10. P5=eggs 1 ==> P3=tea 1    <conf:(1)> lift:(1.25) lev:(0.04) [0] conv:(0.2)
    


A partir des règles précédemment générées, nous avons créé manuellement le fichier "market_binary_named_premises_formal_context.csv" qui nous permet de générer un treillis.

### Changement de kernel pour exécuter le fichier "fca4j-cli-0.4.jar" à l'aide de la commande java (IJava -> ipykernel)


```bash
!mkdir Market
```


```bash
!mkdir Market/Lattice
```


```bash
!java -jar fca4j-cli-0.4.jar LATTICE Market/market_binary_named_premises_formal_context.csv -i CSV -s COMMA -g Market/Lattice/market_binary_named_premises_formal_context.dot
```

    running ADD_EXTENT (BITSET) data: market_binary_named_premises_formal_context.csv ( 10 x 6 )
    duration: 26 ms



```bash
!dot -Tpdf Market/Lattice/market_binary_named_premises_formal_context.dot -o Market/Lattice/market_binary_named_premises_formal_context.pdf
```


```bash
from IPython.display import display_pdf
```


```bash
filename = "Market/Lattice/market_binary_named_premises_formal_context.pdf"
with open(filename,"rb") as f:
    display_pdf(f.read(),raw=True)
```

![workflow_weka_fca4j_result](https://user-images.githubusercontent.com/56521890/169670669-09e76197-a415-4136-8913-ad9b80e852b8.png)



