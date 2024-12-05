import json
import numpy as np
import os


def build_wiki_relation(market_name, connection_file, tic_wiki_file,
                        sel_path_file):
    # readin tickers
    tickers = np.genfromtxt(tic_wiki_file, dtype=str, delimiter=',',
                            skip_header=False)
    print('#tickers selected:', tickers.shape)
    
    wikiid_ticind_dic = {}
    for ind, tw in enumerate(tickers):
        if tw[-1] != 'unknown':
            # Store both the index and ticker symbol to ensure correct ordering
            wikiid_ticind_dic[tw[-1]] = {'index': ind, 'ticker': tw[0]}
    print('#tickers aligned:', len(wikiid_ticind_dic))

    # readin selected paths/connections
    sel_paths = np.genfromtxt(sel_path_file, dtype=str, delimiter=' ',
                              skip_header=False)
    print('#paths selected:', len(sel_paths))
    sel_paths = set(sel_paths[:, 0])

    # readin connections
    with open(connection_file, 'r') as fin:
        connections = json.load(fin)
    print('#connection items:', len(connections))

    # get occured paths
    occur_paths = set()
    for sou_item, conns in connections.items():
        for tar_item, paths in conns.items():
            for p in paths:
                path_key = '_'.join(p)
                if path_key in sel_paths:
                    occur_paths.add(path_key)

    # generate
    occur_paths = sorted(list(occur_paths))
    valid_path_index = {path: ind for ind, path in enumerate(occur_paths)}
    print('#valid paths:', len(valid_path_index))
    for path, ind in valid_path_index.items():
        print(path, ind)
    # one_hot_path_embedding = np.identity(len(valid_path_index) + 1, dtype=int)
    wiki_relation_embedding = np.zeros(
        [tickers.shape[0], tickers.shape[0], len(valid_path_index) + 1],
        dtype=int
    )
    conn_count = 0
    for sou_item, conns in connections.items():
        for tar_item, paths in conns.items():
            for p in paths:
                path_key = '_'.join(p)
                print(sou_item, tar_item, path_key)
                if path_key in valid_path_index.keys():
                    sou_idx = wikiid_ticind_dic[sou_item]['index']
                    tar_idx = wikiid_ticind_dic[tar_item]['index']
                    path_idx = valid_path_index[path_key]
                    print(sou_idx, tar_idx, path_idx)
                    wiki_relation_embedding[sou_idx][tar_idx][path_idx] = 1
                    conn_count += 1
    print('connections count:', conn_count, 'ratio:', conn_count / float(tickers.shape[0] * tickers.shape[0]))

    # handle self relation
    for i in range(tickers.shape[0]):
        wiki_relation_embedding[i][i][-1] = 1
    print(wiki_relation_embedding.shape)
    np.save(market_name + '_wiki_relation', wiki_relation_embedding)

    # Create flattened embedding where value is 1 if any relation exists between companies
    flattened_wiki_relation_embedding = (wiki_relation_embedding.sum(axis=2) > 0).astype(int)
    print("Flattened relation embedding shape:", flattened_wiki_relation_embedding.shape)
    return flattened_wiki_relation_embedding


# single thread version
if __name__ == '__main__':
    path = '../data/relation/wikidata/'
    graph = build_wiki_relation('StockNet',
                        os.path.join(path, 'StockNet_connections.json'),
                        os.path.join(path, 'StockNet_wiki.csv'),
                        os.path.join(path, 'selected_wiki_connections.csv'))
    np.save('../../man-sf-emnlp/graph.npy', graph)
    print(graph)
    # print('----------')
    # build_wiki_relation('NYSE',
    #                     os.path.join(path, 'NYSE_connections.json'),
    #                     os.path.join(path, 'NYSE_wiki.csv'),
    #                     os.path.join(path, 'selected_wiki_connections.csv'))
