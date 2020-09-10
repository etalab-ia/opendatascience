## INSTALLATION 

pip install -r requirements.txt
python -m spacy download fr_core_news_sm
python -m spacy link fr_core_news_sm fr

## RUN
rasa train
rasa run actions
rasa shell 
