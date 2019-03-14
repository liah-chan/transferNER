'''
To run:
CUDA_VISIBLE_DEVICES="" python3.5 main.py &
CUDA_VISIBLE_DEVICES=1 python3.5 main.py &
CUDA_VISIBLE_DEVICES=2 python3.5 main.py &
CUDA_VISIBLE_DEVICES=3 python3.5 main.py &
'''
from __future__ import print_function
import tensorflow as tf
import os
import utils
import numpy as np
import matplotlib
import copy
import distutils.util
import pickle
import glob
import brat_to_conll
import conll_to_brat
import codecs
import utils_nlp

matplotlib.use('Agg')
import utils_plots
import dataset as ds
import time
import random
import evaluate
import configparser
import train
from pprint import pprint
from entity_lstm import EntityLSTM
from tensorflow.contrib.tensorboard.plugins import projector
import argparse
from argparse import RawTextHelpFormatter
import sys
import sklearn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings

warnings.filterwarnings('ignore')


def load_parameters(parameters_filepath, arguments={}, verbose=True):
    '''
    Load parameters from the ini file if specified, take into account any command line argument, and ensure that each parameter is cast to the correct type.
    Command line arguments take precedence over parameters specified in the parameter file.
    '''

    parameters = {}
    # If a parameter file is specified, load it
    if len(parameters_filepath) > 0:
        conf_parameters = configparser.ConfigParser()
        conf_parameters.read(parameters_filepath)
        nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
        for k, v in nested_parameters.items():
            parameters.update(v)
    # Ensure that any arguments the specified in the command line overwrite parameters specified in the parameter file
    for k, v in arguments.items():
        if arguments[k] != arguments['argument_default_value']:
            parameters[k] = v
    for k, v in parameters.items():
        # If the value is a list delimited with a comma, choose one element at random.
        if ',' in v:
            v = random.choice(v.split(','))
            parameters[k] = v
        # Ensure that each parameter is cast to the correct type
        if k in ['num_of_model_to_keep',
                 'character_embedding_dimension', 'character_lstm_hidden_state_dimension',
                 'token_embedding_dimension', 'num_of_model_to_keep',
                 'token_lstm_hidden_state_dimension', 'patience', 'maximum_number_of_epochs',
                 'maximum_training_time', 'number_of_cpu_threads', 'number_of_gpus',
                 'additional_epochs_with_crf']:
            parameters[k] = int(v)
        elif k in ['dropout_rate', 'learning_rate', 'gradient_clipping_value', 'adapter_drop_rate']:
            parameters[k] = float(v)
        elif k in ['dataset_name']:
            parameters[k] = str(v)
        elif k in ['add_class', 'hard_freeze',
                   'remap_unknown_tokens_to_unk', 'use_character_lstm', 'use_crf', 'train_model',
                   'use_pretrained_model', 'debug', 'verbose',
                   'reload_character_embeddings', 'reload_character_lstm',
                   'reload_token_embeddings', 'reload_token_lstm', 'reload_feedforward', 'reload_crf',
                   'check_for_lowercase', 'check_for_digits_replaced_with_zeros', 'freeze_token_embeddings',
                   'load_only_pretrained_token_embeddings', 'refine_with_crf', 'use_adapter', 'include_pos']:
            parameters[k] = distutils.util.strtobool(v)
    if verbose: pprint(parameters)
    return parameters, conf_parameters


def get_valid_dataset_filepaths(parameters):
    print('parameters:', parameters)
    dataset_filepaths = {}
    dataset_brat_folders = {}
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_text_folder'],
                                                       '{0}.txt'.format(dataset_type))
        dataset_brat_folders[dataset_type] = os.path.join(parameters['dataset_text_folder'], dataset_type)
        dataset_compatible_with_brat_filepath = os.path.join(parameters['dataset_text_folder'],
                                                             '{0}_compatible_with_brat.txt'.format(dataset_type))

        # Conll file exists
        if os.path.isfile(dataset_filepaths[dataset_type]) and os.path.getsize(dataset_filepaths[dataset_type]) > 0:
            # Brat text files exist
            if os.path.exists(dataset_brat_folders[dataset_type]) and len(
                    glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:

                # Check compatibility between conll and brat files
                print('executed here: conll exists, brat exists')
                brat_to_conll.check_brat_annotation_and_text_compatibility(dataset_brat_folders[dataset_type])
                if os.path.exists(dataset_compatible_with_brat_filepath):
                    dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath
                conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepaths[dataset_type],
                                                                              dataset_brat_folders[dataset_type])

            # Brat text files do not exist
            else:
                print('executed here: conll exists, brat not exists')

                # Populate brat text and annotation files based on conll file
                conll_to_brat.conll_to_brat(dataset_filepaths[dataset_type], dataset_compatible_with_brat_filepath,
                                            dataset_brat_folders[dataset_type], dataset_brat_folders[dataset_type])
                dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath

        # Conll file does not exist
        else:
            # Brat text files exist
            print('executed here: conll not exists, brat not exists')

            if os.path.exists(dataset_brat_folders[dataset_type]) and len(
                    glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:
                dataset_filepath_for_tokenizer = os.path.join(parameters['dataset_text_folder'],
                                                              '{0}_{1}.txt'.format(dataset_type,
                                                                                   parameters['tokenizer']))
                if os.path.exists(dataset_filepath_for_tokenizer):
                    conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepath_for_tokenizer,
                                                                                  dataset_brat_folders[dataset_type])
                else:
                    # Populate conll file based on brat files
                    brat_to_conll.brat_to_conll(dataset_brat_folders[dataset_type], dataset_filepath_for_tokenizer,
                                                parameters['tokenizer'], parameters['spacylanguage'])
                dataset_filepaths[dataset_type] = dataset_filepath_for_tokenizer

            # Brat text files do not exist
            else:
                del dataset_filepaths[dataset_type]
                del dataset_brat_folders[dataset_type]
                continue

        if parameters['tagging_format'] == 'bioes':
            # Generate conll file with BIOES format
            bioes_filepath = os.path.join(parameters['dataset_text_folder'], '{0}_bioes.txt'.format(
                utils.get_basename_without_extension(dataset_filepaths[dataset_type])))
            utils_nlp.convert_conll_from_bio_to_bioes(dataset_filepaths[dataset_type], bioes_filepath)
            dataset_filepaths[dataset_type] = bioes_filepath

    print('dataset_filepaths:')
    print(dataset_filepaths)
    print('dataset_brat_folders:')
    print(dataset_brat_folders)
    return dataset_filepaths, dataset_brat_folders


def check_parameter_compatiblity(parameters, dataset_filepaths):
    # Check mode of operation
    if parameters['train_model']:
        if 'train' not in dataset_filepaths or 'valid' not in dataset_filepaths:
            raise IOError(
                "If train_model is set to True, both train and valid set must exist in the specified dataset folder: {0}".format(
                    parameters['dataset_text_folder']))
    elif parameters['use_pretrained_model']:
        if 'train' in dataset_filepaths and 'valid' in dataset_filepaths:
            print(
                "WARNING: train and valid set exist in the specified dataset folder, but train_model is set to FALSE: {0}".format(
                    parameters['dataset_text_folder']))
        if 'test' not in dataset_filepaths and 'deploy' not in dataset_filepaths:
            raise IOError(
                "For prediction mode, either test set and deploy set must exist in the specified dataset folder: {0}".format(
                    parameters['dataset_text_folder']))
    else:
        raise ValueError('At least one of train_model and use_pretrained_model must be set to True.')

    if parameters['use_pretrained_model']:
        if all([not parameters[s] for s in
                ['reload_character_embeddings', 'reload_character_lstm', 'reload_token_embeddings', 'reload_token_lstm',
                 'reload_feedforward', 'reload_crf']]):
            raise ValueError(
                'If use_pretrained_model is set to True, at least one of reload_character_embeddings, reload_character_lstm, reload_token_embeddings, reload_token_lstm, reload_feedforward, reload_crf must be set to True.')

    if parameters['gradient_clipping_value'] < 0:
        parameters['gradient_clipping_value'] = abs(parameters['gradient_clipping_value'])


def parse_arguments(arguments=None):
    ''' Parse the arguments

    arguments:
        arguments the arguments, optionally given as argument
    '''
    parser = argparse.ArgumentParser(description='''NeuroNER CLI''', formatter_class=RawTextHelpFormatter)
    parser.add_argument('--parameters_filepath', required=False,
                        default='/home/liah/ner/ner_incremental/src/src/parameters.ini', help='The parameters file')

    argument_default_value = 'argument_default_dummy_value_please_ignore_d41d8cd98f00b204e9800998ecf8427e'
    parser.add_argument('--character_embedding_dimension', required=False, default=argument_default_value, help='')
    parser.add_argument('--character_lstm_hidden_state_dimension', required=False, default=argument_default_value,
                        help='')
    parser.add_argument('--check_for_digits_replaced_with_zeros', required=False, default=argument_default_value,
                        help='')
    parser.add_argument('--check_for_lowercase', required=False, default=argument_default_value, help='')
    parser.add_argument('--dataset_text_folder', required=False, default=argument_default_value, help='')
    parser.add_argument('--debug', required=False, default=argument_default_value, help='')
    parser.add_argument('--dropout_rate', required=False, default=argument_default_value, help='')
    parser.add_argument('--experiment_name', required=False, default=argument_default_value, help='')
    parser.add_argument('--freeze_token_embeddings', required=False, default=argument_default_value, help='')
    parser.add_argument('--gradient_clipping_value', required=False, default=argument_default_value, help='')
    parser.add_argument('--learning_rate', required=False, default=argument_default_value, help='')
    parser.add_argument('--load_only_pretrained_token_embeddings', required=False, default=argument_default_value,
                        help='')
    parser.add_argument('--main_evaluation_mode', required=False, default=argument_default_value, help='')
    parser.add_argument('--maximum_number_of_epochs', required=False, default=argument_default_value, help='')
    parser.add_argument('--number_of_cpu_threads', required=False, default=argument_default_value, help='')
    parser.add_argument('--number_of_gpus', required=False, default=argument_default_value, help='')
    parser.add_argument('--optimizer', required=False, default=argument_default_value, help='')
    parser.add_argument('--output_folder', required=False, default=argument_default_value, help='')
    parser.add_argument('--patience', required=False, default=argument_default_value, help='')
    parser.add_argument('--plot_format', required=False, default=argument_default_value, help='')
    parser.add_argument('--pretrained_model_folder', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_character_embeddings', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_character_lstm', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_crf', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_feedforward', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_token_embeddings', required=False, default=argument_default_value, help='')
    parser.add_argument('--reload_token_lstm', required=False, default=argument_default_value, help='')
    parser.add_argument('--remap_unknown_tokens_to_unk', required=False, default=argument_default_value, help='')
    parser.add_argument('--spacylanguage', required=False, default=argument_default_value, help='')
    parser.add_argument('--tagging_format', required=False, default=argument_default_value, help='')
    parser.add_argument('--token_embedding_dimension', required=False, default=argument_default_value, help='')
    parser.add_argument('--token_lstm_hidden_state_dimension', required=False, default=argument_default_value, help='')
    parser.add_argument('--token_pretrained_embedding_filepath', required=False, default=argument_default_value,
                        help='')
    parser.add_argument('--tokenizer', required=False, default=argument_default_value, help='')
    parser.add_argument('--train_model', required=False, default=argument_default_value, help='')
    parser.add_argument('--use_character_lstm', required=False, default=argument_default_value, help='')
    parser.add_argument('--use_crf', required=False, default=argument_default_value, help='')
    parser.add_argument('--use_pretrained_model', required=False, default=argument_default_value, help='')
    parser.add_argument('--verbose', required=False, default=argument_default_value, help='')
    parser.add_argument('--add_class', required=False, default=argument_default_value, help='')
    parser.add_argument('--hard_freeze', required=False, default=argument_default_value, help='')
    parser.add_argument('--num_of_model_to_keep', required=False, default=argument_default_value, help='')

    try:
        arguments = parser.parse_args(args=arguments)
    except:
        parser.print_help()
        sys.exit(0)

    arguments = vars(
        arguments)  # http://stackoverflow.com/questions/16878315/what-is-the-right-way-to-treat-python-argparse-namespace-as-a-dictionary
    arguments['argument_default_value'] = argument_default_value
    return arguments


def main(argv=sys.argv):
    ''' NeuroNER main method

    Args:
        parameters_filepath the path to the parameters file
        output_folder the path to the output folder
    '''
    arguments = parse_arguments(argv[1:])
    parameters, conf_parameters = load_parameters(arguments['parameters_filepath'], arguments=arguments)
    dataset_filepaths, dataset_brat_folders = get_valid_dataset_filepaths(parameters)
    check_parameter_compatiblity(parameters, dataset_filepaths)

    # Load dataset
    dataset = ds.Dataset(verbose=parameters['verbose'], debug=parameters['debug'])
    dataset.load_dataset(dataset_filepaths, parameters)

    # Create graph and session
    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                intra_op_parallelism_threads=parameters['number_of_cpu_threads'],
                inter_op_parallelism_threads=parameters['number_of_cpu_threads'],
                device_count={'CPU': 1, 'GPU': parameters['number_of_gpus']},
                allow_soft_placement=True,
                # automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
                log_device_placement=False
            )

            sess = tf.Session(config=session_conf)

            with sess.as_default():

                start_time = time.time()
                experiment_timestamp = utils.get_current_time_in_miliseconds()
                results = {}
                results['epoch'] = {}
                results['execution_details'] = {}
                results['execution_details']['train_start'] = start_time
                results['execution_details']['time_stamp'] = experiment_timestamp
                results['execution_details']['early_stop'] = False
                results['execution_details']['keyboard_interrupt'] = False
                results['execution_details']['num_epochs'] = 0
                results['model_options'] = copy.copy(parameters)

                dataset_name = utils.get_basename_without_extension(parameters['dataset_text_folder'])
                model_name = dataset_name
                utils.create_folder_if_not_exists(parameters['output_folder'])
                stats_graph_folder = os.path.join(parameters['output_folder'],
                                                  model_name)  # Folder where to save graphs
                final_weights_folder = os.path.join(parameters['output_folder'], 'weights')
                utils.create_folder_if_not_exists(stats_graph_folder)
                utils.create_folder_if_not_exists(final_weights_folder)
                model_folder = os.path.join(stats_graph_folder, 'model')
                utils.create_folder_if_not_exists(model_folder)
                # saving the parameter setting to the output model dir. For later resuming training
                with open(os.path.join(model_folder, 'parameters.ini'), 'w') as parameters_file:
                    conf_parameters.write(parameters_file)
                tensorboard_log_folder = os.path.join(stats_graph_folder, 'tensorboard_logs')
                utils.create_folder_if_not_exists(tensorboard_log_folder)
                tensorboard_log_folders = {}
                for dataset_type in dataset_filepaths.keys():
                    tensorboard_log_folders[dataset_type] = os.path.join(stats_graph_folder, 'tensorboard_logs',
                                                                         dataset_type)
                    utils.create_folder_if_not_exists(tensorboard_log_folders[dataset_type])
                pickle.dump(dataset, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))

                # Instantiate the model
                # graph initialization should be before FileWriter, otherwise the graph will not appear in TensorBoard
                model = EntityLSTM(dataset, parameters)

                # Instantiate the writers for TensorBoard
                writers = {}
                for dataset_type in dataset_filepaths.keys():
                    writers[dataset_type] = tf.summary.FileWriter(tensorboard_log_folders[dataset_type],
                                                                  graph=sess.graph)
                # embedding_writer has to write in model_folder, otherwise TensorBoard won't be able to view embeddings
                embedding_writer = tf.summary.FileWriter(model_folder)

                embeddings_projector_config = projector.ProjectorConfig()
                tensorboard_token_embeddings = embeddings_projector_config.embeddings.add()
                tensorboard_token_embeddings.tensor_name = model.token_embedding_weights.name
                token_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_tokens.tsv')
                tensorboard_token_embeddings.metadata_path = os.path.relpath(token_list_file_path, '..')

                tensorboard_character_embeddings = embeddings_projector_config.embeddings.add()
                tensorboard_character_embeddings.tensor_name = model.character_embedding_weights.name
                character_list_file_path = os.path.join(model_folder, 'tensorboard_metadata_characters.tsv')
                tensorboard_character_embeddings.metadata_path = os.path.relpath(character_list_file_path, '..')

                projector.visualize_embeddings(embedding_writer, embeddings_projector_config)

                # Write metadata for TensorBoard embeddings
                token_list_file = codecs.open(token_list_file_path, 'w', 'latin-1')
                for token_index in range(dataset.vocabulary_size):
                    token_list_file.write('{0}\n'.format(dataset.index_to_token[token_index]))
                token_list_file.close()

                character_list_file = codecs.open(character_list_file_path, 'w', 'latin-1')
                for character_index in range(dataset.alphabet_size):
                    if character_index == dataset.PADDING_CHARACTER_INDEX:
                        character_list_file.write('PADDING\n')
                    else:
                        character_list_file.write('{0}\n'.format(dataset.index_to_character[character_index]))
                character_list_file.close()

                # Initialize the model
                sess.run(tf.global_variables_initializer())
                if not parameters['use_pretrained_model']:
                    model.load_pretrained_token_embeddings(sess, dataset, parameters)

                # Start training + evaluation loop. Each iteration corresponds to 1 epoch.
                patience_counter = 0
                f1_score_best = 0
                f1_scores = {
                    'train-F1': [],
                    'valid-F1': [],
                    'test-F1': []
                }
                f1_scores_conll = {
                    'train-F1': [],
                    'valid-F1': [],
                    'test-F1': []
                }
                transition_params_trained = np.random.rand(len(dataset.unique_labels) + 2,
                                                           len(dataset.unique_labels) + 2)
                model_saver = tf.train.Saver(max_to_keep=parameters[
                    'num_of_model_to_keep'])
                epoch_number = -1
                try:
                    while True:
                        step = 0
                        epoch_number += 1
                        print('\nStarting epoch {0}'.format(epoch_number))

                        epoch_start_time = time.time()

                        # use pre-trained model and epoch_number = 0
                        if parameters['use_pretrained_model'] and epoch_number == 0:

                            if parameters['use_adapter']:
                                parameters['use_adapter'] = False
                                transition_params_trained = train.restore_pretrained_model(parameters, dataset,
                                                                                           sess, model, model_saver)
                                print('Getting the 3-label predictions from the step1 model.')
                                all_pred_labels, y_pred_for_adapter, y_true_for_adapter, \
                                output_filepaths = train.predict_labels(sess, model,
                                                                        transition_params_trained,
                                                                        parameters, dataset,
                                                                        epoch_number,
                                                                        stats_graph_folder,
                                                                        dataset_filepaths,
                                                                        for_adapter=True)
                                # use the label2idx mapping (for adapter) in the dataset to transform all_pred_labels
                                all_pred_indices = {}
                                for dataset_type in dataset_filepaths.keys():
                                    all_pred_indices[dataset_type] = []
                                    for i in range(len(all_pred_labels[dataset_type])):
                                        indices = [dataset.label_adapter_to_index[label] for label in
                                                   all_pred_labels[dataset_type][i]]
                                        all_pred_indices[dataset_type].append(indices)

                                # and use binarizer to transform to ndarray
                                label_binarizer_adapter = sklearn.preprocessing.LabelBinarizer()
                                label_binarizer_adapter.fit(range(max(dataset.index_to_label_adapter.keys()) + 1))
                                predicted_label_adapter_vector_indices = {}
                                for dataset_type in dataset_filepaths.keys():
                                    predicted_label_adapter_vector_indices[dataset_type] = []
                                    for label_indices_sequence in all_pred_indices[dataset_type]:
                                        predicted_label_adapter_vector_indices[dataset_type].append(
                                            label_binarizer_adapter.transform(label_indices_sequence))
                                parameters['use_adapter'] = True

                            if parameters['train_model'] and parameters['add_class']:
                                transition_params_trained, model, glo_step = \
                                    train.restore_model_parameters_from_pretrained_model(parameters, dataset, sess,
                                                                                         model, model_saver)
                                init_new_vars_op = tf.initialize_variables([glo_step])
                                sess.run(init_new_vars_op)
                            else:
                                transition_params_trained = \
                                    train.restore_pretrained_model(parameters, dataset, sess, model, model_saver)

                            for dataset_type in dataset_filepaths.keys():
                                writers[dataset_type] = tf.summary.FileWriter(tensorboard_log_folders[dataset_type],
                                                                              graph=sess.graph)
                                # embedding_writer has to write in model_folder, otherwise TensorBoard won't be able to view embeddings
                                embedding_writer = tf.summary.FileWriter(model_folder)

                        # epoch_number != 0, no matter use or not use pre-trained model
                        elif epoch_number != 0:
                            # Train model: loop over all sequences of training set with shuffling
                            sequence_numbers = list(range(len(dataset.token_indices['train'])))
                            random.shuffle(sequence_numbers)
                            for sequence_number in sequence_numbers:
                                transition_params_trained, W_before_crf = train.train_step(sess,
                                                                                           dataset, sequence_number,
                                                                                           model,
                                                                                           transition_params_trained,
                                                                                           parameters)
                                step += 1
                        epoch_elapsed_training_time = time.time() - epoch_start_time
                        print('Training completed in {0:.2f} seconds'.format(epoch_elapsed_training_time), flush=False)

                        if parameters['use_adapter']:  # model evaluation, using adapter
                            # pass the pred_for_adapter as label_indices vector
                            original_label_adapter_vector_indices = dataset.label_adapter_vector_indices
                            dataset.label_adapter_vector_indices = predicted_label_adapter_vector_indices
                            y_pred, y_true, output_filepaths = train.predict_labels(sess, model,
                                                                                    transition_params_trained,
                                                                                    parameters, dataset, epoch_number,
                                                                                    stats_graph_folder,
                                                                                    dataset_filepaths)

                            evaluate.evaluate_model(results, dataset, y_pred, y_true, stats_graph_folder, epoch_number,
                                                    epoch_start_time, output_filepaths, parameters)
                            dataset.label_adapter_vector_indices = original_label_adapter_vector_indices

                        else:  # model evaluation,  not using adapter
                            y_pred, y_true, output_filepaths = train.predict_labels(sess, model,
                                                                                    transition_params_trained,
                                                                                    parameters, dataset, epoch_number,
                                                                                    stats_graph_folder,
                                                                                    dataset_filepaths)

                            # Evaluate model: save and plot results
                            evaluate.evaluate_model(results, dataset, y_pred, y_true, stats_graph_folder, epoch_number,
                                                    epoch_start_time, output_filepaths, parameters)

                        summary = sess.run(model.summary_op, feed_dict=None)
                        writers['train'].add_summary(summary, epoch_number)
                        writers['train'].flush()
                        utils.copytree(writers['train'].get_logdir(), model_folder)

                        # Early stopping
                        train_f1_score = results['epoch'][epoch_number][0]['train']['f1_score']['weighted']
                        valid_f1_score = results['epoch'][epoch_number][0]['valid']['f1_score']['weighted']
                        test_f1_score = results['epoch'][epoch_number][0]['test']['f1_score']['weighted']
                        f1_scores['train-F1'].append(train_f1_score)
                        f1_scores['valid-F1'].append(valid_f1_score)
                        f1_scores['test-F1'].append(test_f1_score)

                        train_f1_score_conll = results['epoch'][epoch_number][0]['train']['f1_conll']['micro']
                        valid_f1_score_conll = results['epoch'][epoch_number][0]['valid']['f1_conll']['micro']
                        test_f1_score_conll = results['epoch'][epoch_number][0]['test']['f1_conll']['micro']
                        f1_scores_conll['train-F1'].append(train_f1_score_conll)
                        f1_scores_conll['valid-F1'].append(valid_f1_score_conll)
                        f1_scores_conll['test-F1'].append(test_f1_score_conll)

                        if valid_f1_score > f1_score_best:
                            patience_counter = 0
                            f1_score_best = valid_f1_score
                            # Save the best model
                            model_saver.save(sess, os.path.join(model_folder, 'best_model.ckpt'))
                            print('updated model to current epoch : epoch {:d}'.format(epoch_number))
                            print('the model is saved in: {:s}'.format(model_folder))
                        else:
                            patience_counter += 1
                        print("In epoch {:d}, the valid F1 is : {:f}".format(epoch_number, valid_f1_score))
                        print("The last {0} epochs have not shown improvements on the validation set.".format(
                            patience_counter))

                        if patience_counter >= parameters['patience']:
                            print('Early Stop!')
                            results['execution_details']['early_stop'] = True
                            # save last model
                            model_saver.save(sess, os.path.join(model_folder, 'last_model.ckpt'))
                            print('the last model is saved in: {:s}'.format(model_folder))

                            break

                        if epoch_number >= parameters['maximum_number_of_epochs'] and not parameters['refine_with_crf']:
                            break
                    if not parameters['use_pretrained_model']:
                        plot_name = 'F1-summary-step1.svg'
                    else:
                        plot_name = 'F1-summary-step2.svg'

                    print('Sklearn result:')
                    for k, l in f1_scores.items():
                        print(k, l)

                    print('Conll result:')
                    for k, l in f1_scores_conll.items():
                        print(k, l)
                    utils_plots.plot_f1(f1_scores, os.path.join(stats_graph_folder, '..', plot_name),
                                        'F1 score summary')

                    # TODO: in step 1, for task a, add the best deploy data to step 2 train set, and call script
                    print('(sklearn micro) test F1:')
                    micro_f1 = ','.join(
                        [str(results['epoch'][ep][0]['test']['f1_score']['micro']) for ep in range(epoch_number + 1)])
                    print(micro_f1)
                    print('(sklearn macro) test F1:')
                    macro_f1 = ','.join(
                        [str(results['epoch'][ep][0]['test']['f1_score']['macro']) for ep in range(epoch_number + 1)])
                    print(macro_f1)


                except KeyboardInterrupt:
                    results['execution_details']['keyboard_interrupt'] = True
                    print('Training interrupted')

                print('Finishing the experiment')
                end_time = time.time()
                results['execution_details']['train_duration'] = end_time - start_time
                results['execution_details']['train_end'] = end_time
                evaluate.save_results(results, stats_graph_folder)
                for dataset_type in dataset_filepaths.keys():
                    writers[dataset_type].close()

    sess.close()  # release the session's resources


if __name__ == "__main__":
    main()
