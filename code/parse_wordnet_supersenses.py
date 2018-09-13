
from re import findall, compile
from pandas import DataFrame
from os import path, walk
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer


def get_supersenses_dict():
    orthform_re = compile('<orthForm>(\w*)</orthForm>')
    words_supersenses = defaultdict(lambda: {})
    for _, _, files in walk(path_to_supersenses):
        for file in files:
            with open(path.join(path_to_supersenses, file), 'r') as f:
                supersense = file.split('.xml')[0]
                text = f.read()
                for found_word in findall(orthform_re, text):
                    words_supersenses[found_word][supersense] = 1
    return words_supersenses

def get_one_hot_supersenses(supersenses_dict):
    one_hots = []
    word_key = 'word'
    dict_vectorizer = DictVectorizer(sparse=False)
    one_hot_array = dict_vectorizer.fit_transform(supersenses_dict.values())
    for ind, key in enumerate(supersenses_dict):
        one_hots.append([key] + list(one_hot_array[ind]))
    supersenses_df = DataFrame(one_hots, columns=[word_key]+dict_vectorizer.get_feature_names())
    return supersenses_df

if __name__ == "__main__":
    path_to_supersenses = path.join('data', 'supersenses', 'GN_V120', 'GN_V120_XML')
    supersenses_df = get_one_hot_supersenses(get_supersenses_dict())
    supersenses_df.to_csv(path.join('data', 'output', 'supersenses.csv'))
