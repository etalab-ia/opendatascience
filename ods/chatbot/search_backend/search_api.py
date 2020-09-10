import ods.search.opendatascience_module as opd

def SBERT(recherche):
    response=opd.request_results(opd.Search(recherche, n=5)[0])
    #response= {'result': result, 'score': score} #sert à rien de renvoyer le score pas pertinent
    #url = "https://piaf.datascience.etalab.studio/models/1/doc-qa"
    #data = {"questions": [f"{question}"], "top_k_reader": 3, "top_k_retriever": 5}
    #response = requests.post(url, json=data).json()['results'][0]['answers'][0]
    # text = response['answer']
    # proba = response['probability']
    # context = response['context']
    return response

if __name__ == '__main__':
    response = SBERT("comment faire une carte d'identité ?")
    print(response)