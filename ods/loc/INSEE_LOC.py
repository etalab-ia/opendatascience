from sys import argv

from csv2clean import *
from fuzzywuzzy import fuzz
from tqdm import tqdm
import pandas as pd
import pickle
import spacy

nlp = spacy.load("fr_core_news_lg")
import numpy as np

# file_dir='../../data/Catalogue.csv'
stop_loc = ['Région', 'Métropole', 'Region', 'Metropole', 'Mer', 'mer', 'Département', 'DEPARTEMENT', 'Agglomération',
            'agglomération', 'Communauté', 'communauté']

from joblib import Parallel, delayed

id_table = list(np.load('../../data/id_table.npy', allow_pickle=True))
###########
departements = pd.read_csv('../../data/departement2019.csv')
communes = pd.read_csv('../../data/communes-01012019.csv')
regions = pd.read_csv('../../data/region2019.csv')
communes_prblm = list(np.load('../../data/communes_prblm.npy', allow_pickle=True))

communes_names = communes['nccenr'].to_numpy()
departements_names = departements['nccenr'].to_numpy()
regions_names = regions['nccenr'].to_numpy()

not_comm_names = list(departements_names) + list(regions_names)


###########


###########
def get_id(idx):
    dataset_id = id_table[idx]  # TODO: The zero we receive the first time is then -1, we get the last elem of the list, not sure thats what we want
    return (dataset_id)


def get_idx(ids):
    dataset_idx = id_table.index(ids) + 1
    return (dataset_idx)


###############

def process_data(file_dir):
    all_locs_arr = stoplocs(extract_locs(clean(file_dir, 'none')))
    return (all_locs_arr)


def extract_locs(doc):
    all_locs = [None] * len(doc)
    for i in tqdm(range(len(doc))):
        this_doc = []
        for token in nlp(doc[i]):
            if token.ent_type_ == 'LOC':
                this_doc.append(token.text)
        all_locs[i] = this_doc
    return (all_locs)


def stoplocs(all_locs):
    all_locs_ns = [None] * len(all_locs)
    for i in range(len(all_locs)):
        mini_list = [x for x in all_locs[i] if x not in stop_loc]
        all_locs_ns[i] = ' '.join(mini_list)
    return (np.array(all_locs_ns))


###########

def is_in(locs, category):
    values = []
    for i in category:
        values.append(fuzz.token_set_ratio(locs, i))
        maxi = max(values)
    max_item = [i for i, j in enumerate(values) if j == maxi]
    # print(max_item)
    if values[max_item[0]] == 100:
        found = True
        if len(max_item) > 1:
            values2 = []
            for w in max_item:
                values2.append(fuzz.ratio(locs, category[w]))
            #  print(values2)
            max_item_item = values2.index(max(values2))
            max_item = [max_item[max_item_item]]
    else:
        found = False
    return max_item[0], found


def text_to_loc(text):
    if text == '':
        return pd.DataFrame({'ncc': ['None']})
    if (text.find(' france')!=-1) or (text.find(' France')!=-1):
            return pd.DataFrame({'ncc': ['France']})
    max_item, found_c = is_in(text, communes_names)
    location = communes.loc[[max_item]]
    if communes_names[max_item] in communes_prblm:
        found_c = False
    if not found_c:
        max_item, found_d = is_in(text, not_comm_names)
        try:
            location = departements.loc[[max_item]]
        except:
            location = regions.loc[[max_item - len(departements_names)]]
        return location
    return location


def add_id(dataframe, idx):
    dataframe['id'] = get_id(idx)  # TODO: we are sending here a 0 the first time
    return (dataframe)


def save_to(df_clean, new_location):
    df_clean.to_pickle(new_location + '/locs.pkl')
    df_clean_np = df_clean.to_numpy()
    np.save(new_location + '/locs.npy', df_clean_np, allow_pickle=True, fix_imports=True)


def run(text_str:str, idx:int):
    """
    I crete a function that will receive and treat the elements from the iteration. Appending to a dataframe
    is an expensive operation, so we will return a dict at each treatment. The output of the Parallel function
    will be a list of dicts. This also has the advantage of not making a special index-zero case, as you have done
    in line 140,141
    """
    location_df = text_to_loc(text_str)
    add_id(location_df, idx)
    return location_df.to_dict("r")[0]  # we transform it to dict using array mode ('r') cause i dont need the indices

# def main(file_dir):
#     all_locs_arr = np.load('../../data/locs_insee.npy', allow_pickle=True)  # process_data(file_dir)
#     locs = text_to_loc(all_locs_arr[0])  # TODO we are sending here the 0th element, which is ['title', 'organization']
#     add_id(locs, 0)
#     for l in tqdm(range(1, len(all_locs_arr))):
#         locs.append(add_id(text_to_loc(all_locs_arr[l]), l))
#
#     save_to(locs, 'FOLDER')
#     return (locs)

## 1. First, we try to split the for loop and the function that actually does something with the iterated values
def main(file_dir):
    all_locs_arr = np.load('../../data/locs_insee_str.npy', allow_pickle=True)  # process_data(file_dir)
    job_output = []
    for idx, locations in tqdm(enumerate(all_locs_arr[:10])):
        current_loc = run(locations, idx)
        job_output.append(current_loc)

    locs_df = pd.DataFrame(job_output)
    save_to(locs_df, 'FOLDER')
    return locs_df

## 2. Now, we try and parallelize it
def main_parallel(file_dir, n_jobs=10):
    """
    We do the same as before, but now parallelized. Make sure n_jobs > 1
    """
    all_locs_arr = np.load('../../data/locs_insee_str.npy', allow_pickle=True)  # process_data(file_dir)
    job_output = Parallel(n_jobs=n_jobs)(delayed(run)(locations, idx)
                                         for idx, locations in tqdm(enumerate(all_locs_arr)))

    locs_df = pd.DataFrame(job_output)
    save_to(locs_df, 'FOLDER')
    return locs_df



if __name__ == "__main__":
    #main(argv[1])
    main_parallel("some_file_dir", 10)