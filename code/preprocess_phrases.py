from ufal.udpipe import Model, Pipeline, ProcessingError
from os import path, walk
from nltk.tokenize import RegexpTokenizer
from argparse import ArgumentParser


def make_conll_with_udpipe(text, language='german'):
    if language == 'german':
        model_path = path.join('..', '..', 'udpipe', 'german-ud-2.0-170801.udpipe')
    model = Model.load(model_path)
    pipeline = Pipeline(model, 'tokenize', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
    return pipeline.process(text)


def save(data, filename, output_dir='data', extension='conll'):
    with open(path.join(output_dir, '{}.{}'.format(filename, extension)), 'w') as f:
        f.write(data)

        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--morphology', default='udpipe', help='Specify morphological analyzer (options: "udpipe", default: "udpipe")')
    parsed_args = parser.parse_args()
    morph_analyzers = {
            'udpipe': make_conll_with_udpipe,
    }
    if parsed_args.morphology not in morph_analyzers:
        exit('Morphological analyzer is not supported!')
    phrases_full= []
    aligned_segments_path = path.join('data', 'aligned_segments')
    phrases_file_indicator = '_phrases.txt'
    alpha_tokenizer = RegexpTokenizer('\w+')
    for _, _, files in walk(aligned_segments_path):
        for file in files:
            if phrases_file_indicator in file:
                with open(path.join(aligned_segments_path, file), 'r') as f:
                    phrases_full += [phrase.rsplit('\t', 1)[1] for phrase in f.read().split('\n')[:-1]]
    save(morph_analyzers[parsed_args.morphology]('. '.join(phrases_full)), parsed_args.morphology)
