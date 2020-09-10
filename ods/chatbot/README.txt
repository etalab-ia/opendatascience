Le prototype du Chatbot est basé sur celui de Robin Reynaud: https://github.com/Rob192/POC_api_piaf 

Pour la partie implémentation, le chatbot n'est pas dockérisé il faut donc lancer les commandes suivantes: 
Le plus simple est d'utiliser  terminals séparés.
Terminal 1: Lancer ./ngrok http 5005 et recopier le lien dans le fichier credentials.yml avant 'webhook_url: "https://8357ca1c89ec.ngrok.io/webhooks/telegram/webhook" '
Terminal 2: rasa train 
            rasa run actions (lance le serveur interne pour les messages qui demandent du processing) 
Terminal 3: rasa run 

Il faut ensuite se rendre sur https://web.telegram.org/#/im?p=@datagouv_bot 
et commencer la conversation avec le bot ! 

Le bot est assez fluide dans les échanges. Il se présente assez bien. En revanche une fois que l'on a commencé à faire des requetes (des recherches) il suppose que les échanges suivant hors fin de conversation sont encore des requêtes. 
Aussi, même si SBERT est là pour comprendre les structure et intention de phrase, si dans le champ de la requête l'utilisateur précise: 
'je voudrais que tu me trouves..' ou 'Montre moi les jeux de de données qui' etc... cela bruite le contenu important de la phrase. 

