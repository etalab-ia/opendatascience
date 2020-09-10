import opendatascience_module as opd
import pandas as pd
import sys
import numpy as np
argv=sys.argv
def load_queries(path='../../data/querys.csv'):
    df=pd.read_csv('../../data/querys.csv', sep=',',error_bad_lines=False, encoding='latin-1')
    ids=df['expected']
    queries=np.array(df['query'], dtype=str)
    return queries, ids

embeddings= opd.get_large_files('../../data/test')[0]

def first_found_pos(queries, ids):
    positions=[101]*len(queries)
    for i in range(len(queries)):
        results=opd.Search(queries[i],100)[0]
        k=0
        for r in results:
            k=k+1
            if r==ids[i]:
                positions[i]=k
    return positions

def classify_results(queries, positions):
    acceptable=[]
    rejected=[]
    bof=[]
    for q in range(len(queries)):
        if positions[q]==100:
            rejected.append((queries[q], positions[q]))
        else:
            if positions[q]<=5:
                 acceptable.append((queries[q], positions[q]))
            else:
                bof.append((queries[q], positions[q]))
    return acceptable, bof, rejected

def bench_score(queries, ids):
    positions=first_found_pos(queries, ids)
    score=0
    for i in positions:
        score+=101-i
    score=score/len(queries)
    print(score)
    print(positions)
    return(score)
    
if __name__ == "__main__":
    
    bench_args=load_queries(argv[0])
    bench_score(bench_args[0], bench_args[1])