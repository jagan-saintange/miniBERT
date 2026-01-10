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


### 5. (Éventuellement) Ajout d'Pipeline de Génération de Texte
Ajouter un pipeline de génération de texte **type LLM** comme GPT. Cela permettra d'explorer des capacités supplémentaires et de créer des applications variées basées sur le modèle.


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







