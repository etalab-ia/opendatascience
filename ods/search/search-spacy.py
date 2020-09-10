import requests

import dash
import dash_dangerously_set_inner_html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import plotly.express as px
import pandas as pd

#######

##### Takes a search query as input and get the vectors from the whole dataset to compare.
import spacy
import pickle
import numpy as np
from scipy import spatial
from sklearn.manifold import TSNE
import sys
import unidecode
#from sklearn.decomposition import PCA
#QUERY  Neighbours Ids_and_Score_bool
directory='../'
argv=sys.argv
nlp = spacy.load("fr_core_news_lg")
pca = pickle.load(open(directory+'models/pca_30.pkl','rb'))
pca_space= np.load(directory+'models/vectors_pca_30.npy', allow_pickle=True)
id_table=list(np.load(directory+'../data/id_table.npy', allow_pickle=True))
data=pd.read_csv('../../data/Catalogue_locs.csv', sep=',',error_bad_lines=False, encoding='latin-1')
tree = spatial.KDTree(pca_space)
from spacy.lang.fr.stop_words import STOP_WORDS
from spacy.lang.fr import French
parser=French()
stopwords = list(STOP_WORDS)

def process_query(search_query):
    query=str(search_query).lower()
    clean_query = unidecode.unidecode(query)
    tokens=parser(clean_query)
    tokens = [ word.lower_ for word in tokens ]
    tokens = [ word for word in tokens if word not in stopwords]
    tokens = " ".join([i for i in tokens])
    return (tokens)

def query2vec(search_query):
    x=nlp(search_query).vector #spacy 300d
    y=pca.transform([x])[0] #pca 30d
    return(y)

def get_id(idx):
    dataset_id=id_table[idx-1]
    return(dataset_id)

def get_idx(ids):
    dataset_idx=id_table.index(ids)
    return(dataset_idx)

def id2vec(ids):
    return(list(pca_space[get_idx(ids)]))

def neighbours(vector, n):
    n_ids=[]
    score=[]
    dist, pos=tree.query(vector, k=n)
    for j in range(len(pos)):
        n_ids.append(get_id(pos[j]))
        score.append(1-dist[j]/50) ##very approximate metric
    return(n_ids, score, vector)
def find_vectors(vector, n_ids):
    vectors=[]
    for ids in n_ids:
        vectors.append(id2vec(ids))
    vectors.append(vector)
    tsne_vec = TSNE(n_components=2).fit_transform(vectors)
    return(tsne_vec)

def Search(search_query, n):
    n_ids, score, vector=neighbours(query2vec(process_query(search_query)), n)
    tsne_vec=find_vectors(vector, n_ids)
    #print(n_ids, score)
    return(n_ids, score, tsne_vec)

def Similarity(ids, n):
    n_ids, score=neighbours(id2vec(ids), n)
    return(n_ids, score)

#######


DATAGOUV='https://static.data.gouv.fr/_themes/gouvfr/img/logo-social.png?_=2.1.4'

search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", id="my-input", placeholder="Recherche", debounce=True)),
        dbc.Col(
            dbc.Button("Go", color="primary", className="ml-2"),
            width="auto",
        ),
    ],
    no_gutters=True,
    className="ml-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)

navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=DATAGOUV, height="40px")),
                    dbc.Col(dbc.NavbarBrand("Rechercher un jeu de donnée", style={'color':'White'}, className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="localhost:8050",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(search_bar, id="navbar-collapse", navbar=True),
    ],
    color="MediumBlue",
)

#####################"
def dataset(ids):
    #res = requests.get(f'https://www.data.gouv.fr/api/1/datasets/{ids}')
    #return res.json()
    idx=get_idx(ids)
    #print(idx)
    df=data.loc[idx]
    return(df)
def embed(ids, score):
    # html = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f"""
    #     <div data-udata-dataset="{id}"></div>
    #     <script data-udata="https://www.data.gouv.fr/" src="https://static.data.gouv.fr/static/oembed.js" async defer></script>
    # """)
    ds = dataset(ids)
    output = html.Tr([
        html.Td(html.A(ds['id'], href=f"https://www.data.gouv.fr/fr/datasets/{ids}"), colSpan=1),
        html.Td(html.A(ds['title'], href=f"https://www.data.gouv.fr/fr/datasets/{ids}"), colSpan=1),
       # html.Td(ds['description'], colSpan=2),
        html.Td(ds['pred locs'], colSpan=1),
        html.Td(score, colSpan=1),
    ])

    return output

############"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    dbc.Row(dbc.Col(
    html.Div( navbar
             ))),
    dbc.Row([dbc.Col(dbc.Spinner(html.Div(id='my-output')),width=7),
    dbc.Col(dcc.Graph(id='graph'), width=5)]),
    html.Div(id='knns', style={'display': 'none'})
])



@app.callback(
    Output(component_id='knns', component_property='children'),
    [Input(component_id='my-input', component_property='value')]
)
def compute(input_value):
    knns=Search(input_value, 10)
    knns=pd.DataFrame(knns).to_json()
    return knns
    
    
    
@app.callback(
    Output(component_id='my-output', component_property='children'),
    [Input(component_id='knns', component_property='children')]
)
def update_output_div(knns):
    knns=(list(pd.read_json(knns).loc[0])[:-1],list(pd.read_json(knns).loc[1])[:-1]  )
    results = [ embed(i, j) for i,j in zip(knns[0], knns[1] ) ]
    # print(f" 👀 {input_value}")
    # print(f" results: {results}")
#    results = html.Div([ embed(r) for r in Search(input_value, 10) ])
#    return 'Output: {}'.format(Search(input_value, 10))

    header = [
        html.Thead(html.Tr([
            html.Th("id", colSpan=1),
            html.Th("titre", colSpan=1),
          #  html.Th("description", colSpan=2),
            html.Th("Localisation", colSpan=1),
            html.Th("Distance", colSpan=1),
        ]))
    ]

    return dbc.Table(header + [html.Tbody(results)], bordered=True, hover=True, responsive=True,  striped=True)

@app.callback(Output(component_id='graph', component_property='figure'), [Input('knns', 'children')])
def update_graph(knns):
    knns=pd.read_json(knns)
    xy=np.array(list(knns.loc[2])).transpose()
    color_list=[0]*(len(xy[0])-1)
    color_list[-1]=2
    figure = go.Figure(data=go.Scatter(
        x=xy[0],
        y=xy[1],
        mode='markers',
        marker=dict(color=color_list)
    ))
    return figure

if __name__ == '__main__':
    app.run_server(port=8048, debug=True)