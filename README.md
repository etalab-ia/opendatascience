# opendatascience
Code et expériences sur le thème 'Open Data Science'. 
Cela concerne les jeux de données de data.gouv.fr 
L'approche est NLP donc se réfère exclusivement au contexte des jeux de données ouverts de la plateforme data.gouv.fr

### On retrouve les différents projets suivants: 
 *Thématisation automatique des jeux de données
    LDA pour représenter les thèmes abordés et les mots clés, dans results LDA, vous pouvez consulter la visualisation au format page web.
    Je vous encourage à le faire si vous êtes curieux
 *Meilleure détection de géolocalisation des jeux de données
    Proposition d'un moyen de remplir un champ localisation à partir
  *Modèle (BERT/S-BERT) pour la recherche de documents: 
      Basé sur SBERT Multilingue voir répo : https://github.com/UKPLab/sentence-transformers
      Permet de faire de la recherche sémantique des jeux de données. 
      Interface Dash application de recherche
      Et format Chatbot Telegram pour une recherche conversationnelle. 

### Module
Les fonctions de traitement, de recherche et de vectorisations etc.. sont disponibles sous la forme d'un package python. 
Dans le module opendatascience_module.py dans ods/search/ Permet de récuperer à partir du fichier CSV des datasets de data.gouv.fr avec tous les datasets disponibles de refaire toutes les étapes permettant de construire le moteur de recherche. 

### Explications
Pour une aperçu plus générale, des rapports ou des présentation vous pouvez trouver des pdfs et des informations dans results. 


