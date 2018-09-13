from pandas import DataFrame
from operator import itemgetter
from gensim.models import KeyedVectors
from os import path
from numpy import ndenumerate, isnan, array, dot, load


def get_voxel_coordinates(weights):
    voxel_coordinates = {} 
    for (i, j, k, _), value in ndenumerate(weights):
        if not isnan(weights[i, j]).any():
            voxel_coordinates[(i, j, k)] = array(weights[i, j, k],dtype="float64")
    return voxel_coordinates


def make_brain_map(voxel_coordinates, embeddings_de, top_n=10):
    brain_map = {}
    for coords, voxel in voxel_coordinates.items():
        activations = {}
        for word in emb_dict:
            activations[word] = dot(voxel, emb_dict[word])
        activations = sorted(activations.items(), key=itemgetter(1), reverse=True)
        brain_map['{}, {}, {}'.format(*coords)] = activations[:top_n]
    return brain_map 


def load_data(filename, lemmas=True, filepath=path.join('data')):
    with open(path.join(filepath, filename), 'r') as f:
        data = f.read().split('\n')
    if lemmas:
        word_position_in_conll = 2
    else:
        word_position_in_conll = 1
    return [sentence.split('\t')[word_position_in_conll] for sentence in data if '\t' in sentence and sentence.split('\t')[word_position_in_conll].isalpha()]


def load_weights(filename, filepath=path.join('data', 'weights')):
    return load(path.join(filepath, filename))


def load_embedding(filename, filepath=path.join('..', '..', 'word-embeddings', 'monolang', 'MODELS', 'FastText')):
    return KeyedVectors.load_word2vec_format(path.join(filepath, filename))


def get_word_vectors(data, embeddings):
    emb_dict = {}
    for word in data:
        try:
            emb_dict[word] = embeddings[word]
        except KeyError:
            pass
    return emb_dict


if __name__ == "__main__": 
    data = load_data('udpipe.conll') 
    embeddings = load_embedding('wiki.de.vec')
    word_vectors = get_word_vectors(data, embeddings)
    weights = load_weights('weights_sub09.npy')
    DataFrame(make_brain_map(get_voxel_coordinates(weights), emb_dict)).to_csv(path.join('data', 'output', 'brain_map.csv'))
