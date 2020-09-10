######### IMPORTS #############
# -- DASH-- #
import dash
import dash_dangerously_set_inner_html
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

import requests
# -- MISC--#
import numpy as np
import pandas as pd
import warnings

from opendatascience_module import id2vec, neighbours, Search

warnings.filterwarnings('ignore')

# --MODEL--#
import torch
from sentence_transformers import SentenceTransformer, models


# model = SentenceTransformer('distiluse-base-multilingual-cased')
# --LOADS--# (but a separate function should be able to recommpute these from the original csv)
# data=pd.read_csv('../../data/catalogue_locs_dropped.csv',sep=',', encoding='latin-1')
# id_table_new = list(np.load('../../data/test/id_table.npy', allow_pickle=True))
# embeddings = np.load('../../data/test/embeddings.npy', allow_pickle=True)


####### SEARCH FUNCTIONS ##########
# def query2vec(search_query):
#     return(model.encode([search_query]))
#
# def get_id(idx):
#     dataset_id = id_table_new[idx]
#     return (dataset_id)


# def get_idx(ids):
#     dataset_idx = id_table_new.index(ids)
#     return (dataset_idx)


# def id2vec(ids):
#     return (list(embeddings[get_idx(ids, id_table_new=id_table_new)]))


# def neighbours(vector, n, embeddings, id_table_new):
#     n_ids = []
#     score = []
#     distances = scipy.spatial.distance.cdist(vector, embeddings, "cosine")[0]
#     results = zip(range(len(distances)), distances)
#     results = sorted(results, key=lambda x: x[1])
#
#     for idx, distance in results[0:n]:
#         n_ids.append(get_id(idx, id_table_new=id_table_new))
#         score.append(1 - distance)
#     return (n_ids, score, vector)

#
# def find_vectors(vector, n_ids):
#     vectors = []
#     for ids in n_ids:
#         vectors.append(id2vec(ids, embeddings=embeddings, id_table_new=id_table_new))
#     vectors.append(vector[0])
#     tsne_vec = TSNE(n_components=2).fit_transform(vectors)
#     return (tsne_vec)

#
# def Search(search_query, n):
#
#
#     n_ids, score, vector = neighbours(query2vec(search_query, model=model), n)
#     tsne_vec = tsne_vectors(vector, n_ids,
#                             embeddings=embeddings, id_table_new=id_table_new)
#     # print(n_ids, score)
#     return (n_ids, score, tsne_vec)


def Similarity(ids, n):
    n_ids, score = neighbours(id2vec(ids), n)
    return (n_ids, score)


#######


DATAGOUV = 'https://static.data.gouv.fr/_themes/gouvfr/img/logo-social.png?_=2.1.4'

search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", id="my-input", value=' ', placeholder="Recherche", debounce=True)),
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
                    dbc.Col(dbc.NavbarBrand("Rechercher un jeu de donnée", style={'color': 'White'}, className="ml-2")),
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
    res = requests.get(f'https://www.data.gouv.fr/api/1/datasets/{ids}')
    return res.json()
    # idx=get_idx(ids)
    # print(idx)
    # df=data.loc[idx]
    # return(df)


def embed(ids, score):
    # html = dash_dangerously_set_inner_html.DangerouslySetInnerHTML(f"""
    #     <div data-udata-dataset="{id}"></div>
    #     <script data-udata="https://www.data.gouv.fr/" src="https://static.data.gouv.fr/static/oembed.js" async defer></script>
    # """)
    ds = dataset(ids)
    print(ds['id'])
    output = html.Tr([
        html.Td(html.A(ds['id'], href=f"https://www.data.gouv.fr/fr/datasets/{ids}"), colSpan=1),
        html.Td(html.A(ds['title'], href=f"https://www.data.gouv.fr/fr/datasets/{ids}"), colSpan=1),
        # html.Td(ds['description'], colSpan=2),
        # html.Td(ds['pred locs'], colSpan=1),
        html.Td(score, colSpan=1),
    ])

    return output


############"
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    dbc.Row(dbc.Col(
        html.Div(navbar
                 ))),
    dbc.Row([dbc.Col(dbc.Spinner(html.Div(id='my-output')), width=7),
             dbc.Col(dcc.Graph(id='graph'), width=5)]),
    html.Div(id='knns', style={'display': 'none'})
])


@app.callback(
    Output(component_id='knns', component_property='children'),
    [Input(component_id='my-input', component_property='value')]
)
def compute(input_value):
    knns = Search(input_value, 10)
    knns = pd.DataFrame(knns).to_json()
    return knns


@app.callback(
    Output(component_id='my-output', component_property='children'),
    [Input(component_id='knns', component_property='children')]
)
def update_output_div(knns):
    knns = (list(pd.read_json(knns).loc[0])[:-1], list(pd.read_json(knns).loc[1])[:-1])
    results = [embed(i, j) for i, j in zip(knns[0], knns[1])]
    # print(f" 👀 {input_value}")
    # print(f" results: {results}")
    #    results = html.Div([ embed(r) for r in Search(input_value, 10) ])
    #    return 'Output: {}'.format(Search(input_value, 10))

    header = [
        html.Thead(html.Tr([
            html.Th("id", colSpan=1),
            html.Th("titre", colSpan=1),
            #  html.Th("description", colSpan=2),
            #   html.Th("Localisation", colSpan=1),
            html.Th("Distance", colSpan=1),
        ]))
    ]

    return dbc.Table(header + [html.Tbody(results)], bordered=True, hover=True, responsive=True, striped=True)


@app.callback(Output(component_id='graph', component_property='figure'), [Input('knns', 'children')])
def update_graph(knns):
    knns = pd.read_json(knns)
    xy = np.array(list(knns.loc[2])).transpose()
    color_list = ['blue'] * len(xy[0])
    color_list[-1] = 'red'
    figure = go.Figure(data=go.Scatter(
        x=xy[0],
        y=xy[1],
        mode='markers',
        text=[dataset(i)['title'] for i in list(knns.loc[0])[:-1]] + ['origin'],
        marker=dict(color=color_list)
    ))

    # names ds=datasets[knns[0]] ; nmaes=ds['title']
    return figure


if __name__ == '__main__':
    app.run_server(port=8050, debug=True)
