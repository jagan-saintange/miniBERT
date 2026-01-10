# README for CamemBERT Project

## Objectifs
Ce projet a pour but de **reproduire et évaluer le modèle** présenté dans l'article de 2020 intitulé **CamemBERT: a Tasty French Language Model**. Les étapes clés incluent :

1. Comprendre l'article et ses contributions.
2. **Reproduire les prédictions** sur les tests présentés dans l'article.
3. Utiliser un **modèle pré-entrainé** pour effectuer les évaluations.
4. **Recréer** un modèle light (4 Go) entraîné par nous même et reprendre toute la procédure.
5. **Comparer et analyser** les résultats avec ceux de l'article d'origine.
6. S'amuser à faire plus :) (Si on a le temps)

---

## Méthodologie

### 1. Compréhension de l'Article
Lire et analyser le papier original pour saisir les méthodologies utilisées et les résultats obtenus.

### 2. Modèle Préentraîné
Utiliser le modèle pré-entrainé pour **effectuer les tests** décrits dans l'article et obtenir des résultats comparables.

### 3. Entraîner notre propre modèle
Configurer un environnement pour charger et entraîner un modèle **CamemBERT**. Utiliser des bibliothèques comme **Transformers** de Hugging Face pour faciliter ce processus.

### 4. Comparaison
On souhaite comparer :
- Les performances du modèle reproduit avec le modèle d'origine.
- Les résultats des différents tests réalisés.


### 5. Execution du code 
###Protocole pour lancer les codes camemBERT_Small_2 et réaliser le modèle oscar entrainé
- Télécharger au moins 4Go de données depuis ce lien :  https://oscar-public.huma-num.fr/shuff-orig/fr/
- Dans le bash exécuter la commande : cat fichier1 fichier2 .... fichier1 >> fichierfinal
- Executer en adaptant les path sur chaque code:
  -clean_corpus.py
  -adapt_to_4GB.py
  -remover_blank.py
- Executer ensuite en créant modifiants les path, create_shards.py
- Executer dans cet ordre et en adaptant les paths :
    -train_spm.py
    -make_tokenizer.py
    -create_config.py
    -init_model.py
    -train_camemBERT_MLM-Small_Vanilla.py

###Protocole pour réaliser le fine tuning NLI
-Executer le code fine_tuning_NLI.py en adaptant le path et en choisissant le modèle à entrainer
Les modèles possibles sont : camembert-base, et les modèles réalisé avec le protocole précédent (nom au choix)
  

###Protocole pour lancer les code POS 
importer dans les même répertoire que le script les 3 lignes des bases de donnée:

- fr_****-ud-dev.conllu
- fr_****-ud-train.conllu
- fr_****-ud-test.conllu


voici les liens pour les 4 bases de données : 

- GSD :          https://github.com/UniversalDependencies/UD_French-GSD
- Sequoia :      https://github.com/UniversalDependencies/UD_French-Sequoia
- ParisStories : https://github.com/UniversalDependencies/UD_French-ParisStories
- ParTUT :       https://github.com/UniversalDependencies/UD_French-ParTUT








