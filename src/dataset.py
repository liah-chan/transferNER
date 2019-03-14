import sklearn.preprocessing
import utils
import collections
import codecs
import utils_nlp
import re
import time
import token
import os
import sys
import pickle


def log_variable_to_file(var, file_name):
    tmp_log_dir = os.path.join('.', 'tmp')
    if not os.path.isdir(tmp_log_dir):
        os.makedirs(tmp_log_dir)

    if isinstance(var, list):
        pass
    if isinstance(var, dict) or isinstance(var, collections.defaultdict):
        token_count_log_file = os.path.join(tmp_log_dir, file_name + '.txt')
        with codecs.open(token_count_log_file, 'w', 'latin-1', errors='replace') as f:
            for k, v in var.items():
                f.writelines('{} {}\n'.format(k, v))


class Dataset(object):
    """A class for handling data sets."""

    def __init__(self, name='', verbose=False, debug=False):
        self.name = name
        self.verbose = verbose
        self.debug = debug

    def tokenize(self, word):
        if len(word) >= 25:
            word = re.sub(r'a-z', 'a', word)
        word = re.sub(r'\d', '0', word)

        return word

    def _parse_dataset(self, dataset_filepath, use_adapter=False, include_pos=False, tagging_format='bioes'):
        token_count = collections.defaultdict(lambda: 0)
        label_count = collections.defaultdict(lambda: 0)
        character_count = collections.defaultdict(lambda: 0)
        if use_adapter:
            label_adapter_count = collections.defaultdict(lambda: 0)
        if include_pos:
            label_pos_count = collections.defaultdict(lambda: 0)
        line_count = -1
        tokens = []
        labels = []
        characters = []
        token_lengths = []
        new_token_sequence = []
        new_label_sequence = []
        if use_adapter:
            labels_adapter = []
            new_label_sequence_adapter = []
        if include_pos:
            labels_pos = []
            new_label_sequence_pos = []
        if dataset_filepath:
            f = codecs.open(dataset_filepath, encoding='latin-1')
            for line in f:
                line_count += 1
                line = line.strip().split(' ')
                if len(line) == 0 or len(line[0]) == 0 or '-DOCSTART-' in line[0]:
                    if len(new_token_sequence) > 0:
                        labels.append(new_label_sequence)
                        tokens.append(new_token_sequence)
                        if use_adapter:
                            labels_adapter.append(new_label_sequence_adapter)
                            new_label_sequence_adapter = []
                        if include_pos:
                            labels_pos.append(new_label_sequence_pos)
                            new_label_sequence_pos = []

                        new_token_sequence = []
                        new_label_sequence = []
                    continue
                token = self.tokenize(str(line[0]))
                label = str(line[-1])

                if use_adapter:
                    # the index because of *_compat_with_brat_bioes file
                    if tagging_format == 'bio':
                        label_adapter = str(line[-2])
                        if include_pos:
                            label_pos = str(line[-4])
                    else:  # conll file in this exp uses bioes
                        label_adapter = str(line[-3])
                        if include_pos:
                            label_pos = str(line[-5])
                token_count[token] += 1
                label_count[label] += 1

                new_token_sequence.append(token)
                new_label_sequence.append(label)
                if use_adapter:
                    new_label_sequence_adapter.append(label_adapter)
                    label_adapter_count[label_adapter] += 1
                if include_pos:
                    new_label_sequence_pos.append(label_pos)
                    label_pos_count[label_pos] += 1

                for character in token:
                    character_count[character] += 1

                if self.debug and line_count > 200: break  # for debugging purposes

            if len(new_token_sequence) > 0:
                labels.append(new_label_sequence)
                tokens.append(new_token_sequence)
                if use_adapter:
                    labels_adapter.append(new_label_sequence_adapter)
                if include_pos:
                    labels_pos.append(new_label_sequence_pos)
        if use_adapter:
            if include_pos:
                return labels_pos, labels_adapter, labels, tokens, \
                       token_count, label_pos_count, label_adapter_count, label_count, character_count
            else:
                return labels_adapter, labels, tokens, token_count, label_adapter_count, label_count, character_count

        else:
            return labels, tokens, token_count, label_count, character_count

    def load_dataset(self, dataset_filepaths, parameters):
        '''
        dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
        '''
        start_time = time.time()
        print('Load dataset... ', end='', flush=True)
        all_pretrained_tokens = []
        if parameters['token_pretrained_embedding_filepath'] != '':
            all_pretrained_tokens = utils_nlp.load_tokens_from_pretrained_token_embeddings(parameters)
        if self.verbose: print("len(all_pretrained_tokens): {0}".format(len(all_pretrained_tokens)))

        # Load pretraining dataset to ensure that index to label is compatible to the pretrained model,
        #   and that token embeddings that are learned in the pretrained model are loaded properly.
        all_tokens_in_pretraining_dataset = []
        if parameters['use_pretrained_model']:
            pretraining_dataset = pickle.load(
                open(os.path.join(parameters['pretrained_model_folder'], 'dataset.pickle'), 'rb'))
            all_tokens_in_pretraining_dataset = pretraining_dataset.index_to_token.values()

        remap_to_unk_count_threshold = 1
        self.UNK_TOKEN_INDEX = 0
        self.PADDING_CHARACTER_INDEX = 0
        self.tokens_mapped_to_unk = []
        self.UNK = 'UNK'
        self.unique_labels = []
        labels = {}
        tokens = {}
        if parameters['use_adapter']:
            labels_adapter = {}
            label_adapter_count = {}
            self.unique_labels_adapter = []
            if parameters['include_pos']:
                labels_pos = {}
                label_pos_count = {}
                self.unique_labels_pos = []
        characters = {}
        token_lengths = {}
        label_count = {}
        token_count = {}
        character_count = {}

        for dataset_type in ['train', 'valid', 'test', 'deploy']:
            if parameters['use_adapter']:
                if parameters['include_pos']:
                    labels_pos[dataset_type], labels_adapter[dataset_type], labels[dataset_type], \
                    tokens[dataset_type], token_count[dataset_type], label_pos_count[dataset_type], \
                    label_adapter_count[dataset_type], label_count[dataset_type], character_count[dataset_type] \
                        = self._parse_dataset(dataset_filepaths.get(dataset_type, None),
                                              use_adapter=True, include_pos=True,
                                              tagging_format=parameters['tagging_format'])
                else:
                    labels_adapter[dataset_type], labels[dataset_type], tokens[dataset_type], \
                    token_count[dataset_type], label_adapter_count[dataset_type], label_count[dataset_type], \
                    character_count[dataset_type] \
                        = self._parse_dataset(dataset_filepaths.get(dataset_type, None), use_adapter=True,
                                              tagging_format=parameters['tagging_format'])

            else:
                labels[dataset_type], tokens[dataset_type], token_count[dataset_type], label_count[dataset_type], \
                character_count[dataset_type] \
                    = self._parse_dataset(dataset_filepaths.get(dataset_type, None),
                                          tagging_format=parameters['tagging_format'])
            if self.verbose:
                print("len(token_count[{1}]): {0}".format(len(token_count[dataset_type]), dataset_type))
        token_count['all'] = {}
        for token in list(token_count['train'].keys()) + list(token_count['valid'].keys()) + list(
                token_count['test'].keys()) + list(token_count['deploy'].keys()):
            token_count['all'][token] = token_count['train'][token] + token_count['valid'][token] + token_count['test'][
                token] + token_count['deploy'][token]

        if self.verbose: print("len(token_count[all]): {0}".format(len(token_count['all'])))

        for dataset_type in dataset_filepaths.keys():
            if self.verbose: print("len(token_count[{1}]): {0}".format(len(token_count[dataset_type]), dataset_type))

        character_count['all'] = {}
        for character in list(character_count['train'].keys()) + list(character_count['valid'].keys()) + list(
                character_count['test'].keys()) + list(character_count['deploy'].keys()):
            character_count['all'][character] = character_count['train'][character] + character_count['valid'][
                character] + character_count['test'][character] + character_count['deploy'][character]

        label_count['all'] = {}
        for character in list(label_count['train'].keys()) + list(label_count['valid'].keys()) + list(
                label_count['test'].keys()) + list(label_count['deploy'].keys()):
            label_count['all'][character] = label_count['train'][character] + label_count['valid'][character] + \
                                            label_count['test'][character] + label_count['deploy'][character]
        if parameters['use_adapter']:
            label_adapter_count['all'] = {}
            for label in list(label_adapter_count['train'].keys()) + list(label_adapter_count['valid'].keys()) + list(
                    label_adapter_count['test'].keys()) + list(label_adapter_count['deploy'].keys()):
                label_adapter_count['all'][label] = label_adapter_count['train'][label] + label_adapter_count['valid'][
                    label] + \
                                                    label_adapter_count['test'][label] + label_adapter_count['deploy'][
                                                        label]
            label_adapter_count['all'] = utils.order_dictionary(label_adapter_count['all'], 'key', reverse=False)

            if parameters['include_pos']:
                label_pos_count['all'] = {}
                for label in list(label_pos_count['train'].keys()) + list(label_pos_count['valid'].keys()) + list(
                        label_pos_count['test'].keys()) + list(label_pos_count['deploy'].keys()):
                    label_pos_count['all'][label] = label_pos_count['train'][label] + label_pos_count['valid'][label] + \
                                                    label_pos_count['test'][label] + label_pos_count['deploy'][label]
                label_pos_count['all'] = utils.order_dictionary(label_pos_count['all'], 'key', reverse=False)

        token_count['all'] = utils.order_dictionary(token_count['all'], 'value_key', reverse=True)
        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse=False)

        character_count['all'] = utils.order_dictionary(character_count['all'], 'value', reverse=True)
        if self.verbose: print('character_count[\'all\']: {0}'.format(character_count['all']))

        token_to_index = {}
        token_to_index[self.UNK] = self.UNK_TOKEN_INDEX
        iteration_number = 0
        number_of_unknown_tokens = 0
        if self.verbose: print(
            "parameters['remap_unknown_tokens_to_unk']: {0}".format(parameters['remap_unknown_tokens_to_unk']))
        if self.verbose: print("len(token_count['train'].keys()): {0}".format(len(token_count['train'].keys())))
        for token, count in token_count['all'].items():
            if iteration_number == self.UNK_TOKEN_INDEX: iteration_number += 1

            if parameters['remap_unknown_tokens_to_unk'] == 1 and \
                    (token_count['train'][token] == 0 or \
                     parameters['load_only_pretrained_token_embeddings']) and \
                    not utils_nlp.is_token_in_pretrained_embeddings(token, all_pretrained_tokens, parameters) and \
                    token not in all_tokens_in_pretraining_dataset:
                token_to_index[token] = self.UNK_TOKEN_INDEX
                number_of_unknown_tokens += 1
                self.tokens_mapped_to_unk.append(token)
            else:
                token_to_index[token] = iteration_number
                iteration_number += 1
        if self.verbose: print("number_of_unknown_tokens: {0}".format(number_of_unknown_tokens))

        infrequent_token_indices = []
        for token, count in token_count['train'].items():
            if 0 < count <= remap_to_unk_count_threshold:
                infrequent_token_indices.append(token_to_index[token])
        if self.verbose: print("len(token_count['train']): {0}".format(len(token_count['train'])))
        if self.verbose: print("len(infrequent_token_indices): {0}".format(len(infrequent_token_indices)))

        # Ensure that both B- and I- versions exist for each label
        labels_without_bio = set()
        for label in label_count['all'].keys():
            new_label = utils_nlp.remove_bio_from_label_name(label)
            labels_without_bio.add(new_label)
        for label in labels_without_bio:
            if label == 'O':
                continue
            if parameters['tagging_format'] == 'bioes':
                prefixes = ['B-', 'I-', 'E-', 'S-']
            else:
                prefixes = ['B-', 'I-']
            for prefix in prefixes:
                l = prefix + label
                if l not in label_count['all']:
                    label_count['all'][l] = 0
        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse=False)

        if parameters['use_pretrained_model'] and not parameters['add_class']:
            self.unique_labels = sorted(list(pretraining_dataset.label_to_index.keys()))
            # Make sure labels are compatible with the pretraining dataset.
            for label in label_count['all']:
                if label not in pretraining_dataset.label_to_index:
                    raise AssertionError("The label {0} does not exist in the pretraining dataset. ".format(label) +
                                         "Please ensure that only the following labels exist in the dataset: {0}".format(
                                             ', '.join(self.unique_labels)))
            label_to_index = pretraining_dataset.label_to_index.copy()

        elif parameters['use_pretrained_model'] and parameters['add_class']:
            # make sure that the added labels are mapped to the end of the dectionary
            print('Adding new label-index pair to label_to_index dictionary')
            old_label_to_index = pretraining_dataset.label_to_index.copy()
            for label, count in label_count['all'].items():
                if label not in old_label_to_index.keys():
                    old_label_to_index[label] = len(old_label_to_index.keys())
            label_to_index = old_label_to_index.copy()

            self.unique_labels = list(label_to_index.keys())
        else:
            label_to_index = {}
            iteration_number = 0
            for label, count in label_count['all'].items():
                label_to_index[label] = iteration_number
                iteration_number += 1
                self.unique_labels.append(label)
        if parameters['use_adapter']:
            label_adapter_to_index = {}
            self.unique_labels_adapter = list(label_adapter_count['all'].keys())
            for n, label in enumerate(self.unique_labels_adapter):
                label_adapter_to_index[label] = n
            if parameters['include_pos']:
                label_pos_to_index = {}
                self.unique_labels_pos = list(label_pos_count['all'].keys())
                for n, pos in enumerate(self.unique_labels_pos):
                    label_pos_to_index[pos] = n

        if self.verbose: print('self.unique_labels: {0}'.format(self.unique_labels))

        character_to_index = {}
        iteration_number = 0
        for character, count in character_count['all'].items():
            if iteration_number == self.PADDING_CHARACTER_INDEX: iteration_number += 1
            character_to_index[character] = iteration_number
            iteration_number += 1

        if self.verbose: print('token_count[\'train\'][0:10]: {0}'.format(list(token_count['train'].items())[0:10]))
        token_to_index = utils.order_dictionary(token_to_index, 'value', reverse=False)
        index_to_token = utils.reverse_dictionary(token_to_index)
        if parameters['remap_unknown_tokens_to_unk'] == 1: index_to_token[self.UNK_TOKEN_INDEX] = self.UNK

        if self.verbose: print('label_count[\'train\']: {0}'.format(label_count['train']))
        label_to_index = utils.order_dictionary(label_to_index, 'value', reverse=False)
        if self.verbose: print('label_to_index: {0}'.format(label_to_index))
        index_to_label = utils.reverse_dictionary(label_to_index)
        if self.verbose: print('index_to_label: {0}'.format(index_to_label))

        if parameters['use_adapter']:
            label_adapter_to_index = utils.order_dictionary(label_adapter_to_index, 'value', reverse=False)
            index_to_label_adapter = utils.reverse_dictionary(label_adapter_to_index)
            if parameters['include_pos']:
                label_pos_to_index = utils.order_dictionary(label_pos_to_index, 'value', reverse=False)
                index_to_label_pos = utils.reverse_dictionary(label_pos_to_index)

        character_to_index = utils.order_dictionary(character_to_index, 'value', reverse=False)
        index_to_character = utils.reverse_dictionary(character_to_index)
        if self.verbose: print('character_to_index: {0}'.format(character_to_index))
        if self.verbose: print('index_to_character: {0}'.format(index_to_character))

        if self.verbose: print('labels[\'train\'][0:10]: {0}'.format(labels['train'][0:10]))
        if self.verbose: print('tokens[\'train\'][0:10]: {0}'.format(tokens['train'][0:10]))

        if self.verbose:
            # Print sequences of length 1 in train set
            for token_sequence, label_sequence in zip(tokens['train'], labels['train']):
                if len(label_sequence) == 1 and label_sequence[0] != 'O':
                    print("{0}\t{1}".format(token_sequence[0], label_sequence[0]))

        # Map tokens and labels to their indices
        token_indices = {}
        label_indices = {}
        if parameters['use_adapter']:
            label_indices_adapter = {}
            if parameters['include_pos']:
                label_indices_pos = {}

        character_indices = {}
        character_indices_padded = {}
        for dataset_type in dataset_filepaths.keys():
            token_indices[dataset_type] = []
            characters[dataset_type] = []
            character_indices[dataset_type] = []
            token_lengths[dataset_type] = []
            character_indices_padded[dataset_type] = []

            for token_sequence in tokens[dataset_type]:
                token_indices[dataset_type].append([token_to_index[token] for token in token_sequence])
                characters[dataset_type].append([list(token) for token in token_sequence])
                character_indices[dataset_type].append(
                    [[character_to_index[character] for character in token] for token in token_sequence])
                token_lengths[dataset_type].append([len(token) for token in token_sequence])

                longest_token_length_in_sequence = max(token_lengths[dataset_type][-1])
                character_indices_padded[dataset_type].append(
                    [utils.pad_list(temp_token_indices, longest_token_length_in_sequence, self.PADDING_CHARACTER_INDEX)
                     for temp_token_indices in character_indices[dataset_type][-1]])
            label_indices[dataset_type] = []
            for label_sequence in labels[dataset_type]:
                label_indices[dataset_type].append([label_to_index[label] for label in label_sequence])
            if parameters['use_adapter']:
                label_indices_adapter[dataset_type] = []
                for label_sequence_adapter in labels_adapter[dataset_type]:
                    label_indices_adapter[dataset_type].append(
                        [label_adapter_to_index[label] for label in label_sequence_adapter])
                if parameters['include_pos']:
                    label_indices_pos[dataset_type] = []
                    for label_sequence_pos in labels_pos[dataset_type]:
                        label_indices_pos[dataset_type].append(
                            [label_pos_to_index[label] for label in label_sequence_pos])

        if self.verbose: print('token_lengths[\'train\'][0][0:10]: {0}'.format(token_lengths['train'][0][0:10]))
        if self.verbose: print('characters[\'train\'][0][0:10]: {0}'.format(characters['train'][0][0:10]))
        if self.verbose: print('token_indices[\'train\'][0:10]: {0}'.format(token_indices['train'][0:10]))
        if self.verbose: print('label_indices[\'train\'][0:10]: {0}'.format(label_indices['train'][0:10]))
        if self.verbose: print('character_indices[\'train\'][0][0:10]: {0}'.format(character_indices['train'][0][0:10]))
        if self.verbose: print(
            'character_indices_padded[\'train\'][0][0:10]: {0}'.format(character_indices_padded['train'][0][0:10]))

        label_binarizer = sklearn.preprocessing.LabelBinarizer()
        label_binarizer.fit(range(max(index_to_label.keys()) + 1))
        label_vector_indices = {}

        for dataset_type in dataset_filepaths.keys():
            label_vector_indices[dataset_type] = []
            for label_indices_sequence in label_indices[dataset_type]:
                label_vector_indices[dataset_type].append(label_binarizer.transform(label_indices_sequence))

        if parameters['use_adapter']:
            label_binarizer_adapter = sklearn.preprocessing.LabelBinarizer()
            label_binarizer_adapter.fit(range(max(index_to_label_adapter.keys()) + 1))
            label_adapter_vector_indices = {}
            for dataset_type in dataset_filepaths.keys():
                label_adapter_vector_indices[dataset_type] = []
                for label_indices_sequence in label_indices_adapter[dataset_type]:
                    label_adapter_vector_indices[dataset_type].append(
                        label_binarizer_adapter.transform(label_indices_sequence))
            if parameters['include_pos']:
                label_binarizer_pos = sklearn.preprocessing.LabelBinarizer()
                label_binarizer_pos.fit(range(max(index_to_label_pos.keys()) + 1))
                label_pos_vector_indices = {}
                for dataset_type in dataset_filepaths.keys():
                    label_pos_vector_indices[dataset_type] = []
                    for label_indices_sequence in label_indices_pos[dataset_type]:
                        label_pos_vector_indices[dataset_type].append(
                            label_binarizer_pos.transform(label_indices_sequence))
        if self.verbose: print('label_vector_indices[\'train\'][0:2]: {0}'.format(label_vector_indices['train'][0:2]))

        if self.verbose: print('len(label_vector_indices[\'train\']): {0}'.format(len(label_vector_indices['train'])))
        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        self.token_indices = token_indices
        self.label_indices = label_indices
        self.character_indices_padded = character_indices_padded
        self.index_to_character = index_to_character
        self.character_to_index = character_to_index
        self.character_indices = character_indices
        self.token_lengths = token_lengths
        self.characters = characters
        self.tokens = tokens
        self.labels = labels
        self.label_vector_indices = label_vector_indices
        self.index_to_label = index_to_label
        self.label_to_index = label_to_index
        if parameters['use_adapter']:
            self.index_to_label_adapter = index_to_label_adapter
            self.label_adapter_to_index = label_adapter_to_index
            self.label_indices_adapter = label_indices_adapter
            self.label_adapter_vector_indices = label_adapter_vector_indices
            if parameters['include_pos']:
                self.index_to_label_pos = index_to_label_pos
                self.label_pos_to_index = label_pos_to_index
                self.label_indices_pos = label_indices_pos
                self.label_pos_vector_indices = label_pos_vector_indices
        if self.verbose: print("len(self.token_to_index): {0}".format(len(self.token_to_index)))
        if self.verbose: print("len(self.index_to_token): {0}".format(len(self.index_to_token)))

        if parameters['add_class'] and parameters['tagging_format'] == 'bioes' and len(self.index_to_label) > 100:
            self.number_of_classes = max(self.index_to_label.keys()) + 1 - 8
        elif parameters['add_class'] and parameters['tagging_format'] == 'bioes':
            print('here')
            self.number_of_classes = max(self.index_to_label.keys()) + 1 - 4
        elif parameters['add_class'] and parameters['tagging_format'] == 'bio':
            print('here2')
            self.number_of_classes = max(self.index_to_label.keys()) + 1 - 2
        else:
            self.number_of_classes = max(self.index_to_label.keys()) + 1  # 1 is for O label
        print('max(self.index_to_label.keys()) : {:d}'.format(max(self.index_to_label.keys())))
        print(self.index_to_label.keys())
        print(self.number_of_classes)

        self.vocabulary_size = max(self.index_to_token.keys()) + 1
        self.alphabet_size = max(self.index_to_character.keys()) + 1
        if self.verbose: print("self.number_of_classes: {0}".format(self.number_of_classes))
        if self.verbose: print("self.alphabet_size: {0}".format(self.alphabet_size))
        if self.verbose: print("self.vocabulary_size: {0}".format(self.vocabulary_size))

        self.unique_labels_of_interest = list(self.unique_labels)
        self.unique_labels_of_interest.remove('O')

        self.unique_label_indices_of_interest = []
        for lab in self.unique_labels_of_interest:
            self.unique_label_indices_of_interest.append(label_to_index[lab])

        self.infrequent_token_indices = infrequent_token_indices

        if self.verbose: print('self.unique_labels_of_interest: {0}'.format(self.unique_labels_of_interest))
        if self.verbose: print(
            'self.unique_label_indices_of_interest: {0}'.format(self.unique_label_indices_of_interest))

        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))
