import os
import random
import time
import tensorflow as tf
import numpy as np
import sklearn.metrics
from evaluate import remap_labels
from pprint import pprint
import pickle
import utils_tf
import main
import codecs
import utils_nlp
import utils
import evaluate
import entity_lstm


# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

def train_step(sess, dataset, sequence_number, model, transition_params_trained, parameters):
    # Perform one iteration
    token_indices_sequence = dataset.token_indices['train'][sequence_number]
    for i, token_index in enumerate(token_indices_sequence):
        if token_index in dataset.infrequent_token_indices and np.random.uniform() < 0.5:
            token_indices_sequence[i] = dataset.token_to_index[dataset.UNK]

    if parameters['use_adapter']:
        feed_dict = {
            model.input_token_indices: token_indices_sequence,
            model.input_label_indices_vector: dataset.label_vector_indices['train'][sequence_number],
            model.input_token_character_indices: dataset.character_indices_padded['train'][sequence_number],
            model.input_token_lengths: dataset.token_lengths['train'][sequence_number],
            model.input_label_indices_flat: dataset.label_indices['train'][sequence_number],
            model.dropout_keep_prob: 1 - parameters['dropout_rate'],
            model.adapter_keep_prob: 1 - parameters['adapter_drop_rate'],
            model.input_label_adapter_indices_vector: dataset.label_adapter_vector_indices['train'][
                sequence_number],
            model.input_label_adapter_indices_flat: dataset.label_indices_adapter['train'][sequence_number]
        }
        if parameters['include_pos']:
            feed_dict[model.input_label_pos_indices_vector] = dataset.label_pos_vector_indices['train'][sequence_number]
            feed_dict[model.input_label_pos_indices_flat] = dataset.label_indices_pos['train'][sequence_number]
    else:
        feed_dict = {
            model.input_token_indices: token_indices_sequence,
            model.input_label_indices_vector: dataset.label_vector_indices['train'][sequence_number],
            model.input_token_character_indices: dataset.character_indices_padded['train'][sequence_number],
            model.input_token_lengths: dataset.token_lengths['train'][sequence_number],
            model.input_label_indices_flat: dataset.label_indices['train'][sequence_number],
            model.dropout_keep_prob: 1 - parameters['dropout_rate']
        }
    _, _, loss, accuracy, transition_params_trained, W_before_crf = sess.run(
        [model.train_op, model.global_step, model.loss,
         model.accuracy, model.transition_parameters, model.W_before_crf],
        feed_dict)

    return transition_params_trained, W_before_crf


def prediction_step(sess, dataset, dataset_type, model, transition_params_trained, stats_graph_folder,
                    epoch_number, parameters, dataset_filepaths, for_adapter=False):
    if dataset_type == 'deploy':
        print('Predict labels for the {0} set'.format(dataset_type))
    else:
        print('Evaluate model on the {0} set'.format(dataset_type))
    if for_adapter == True:
        all_predictions_per_sentence = []
        all_y_true_per_sentence = []
        all_prediction_labels_per_sentence = []

    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder, '{1:03d}_{0}.txt'.format(dataset_type, epoch_number))
    output_file = codecs.open(output_filepath, 'w', 'latin-1', errors='replace')
    original_conll_file = codecs.open(dataset_filepaths[dataset_type], 'r', 'latin-1', errors='replace')
    for i in range(len(dataset.token_indices[dataset_type])):
        if parameters['use_adapter']:
            feed_dict = {
                model.input_token_indices: dataset.token_indices[dataset_type][i],
                model.input_token_character_indices: dataset.character_indices_padded[dataset_type][i],
                model.input_token_lengths: dataset.token_lengths[dataset_type][i],
                model.input_label_indices_vector: dataset.label_vector_indices[dataset_type][i],
                model.input_label_adapter_indices_vector: dataset.label_adapter_vector_indices[dataset_type][i],
                model.dropout_keep_prob: 1.,
                model.adapter_keep_prob: 1.

            }
            if parameters['include_pos']:
                feed_dict[model.input_label_pos_indices_vector] = dataset.label_pos_vector_indices[dataset_type][i]
        elif for_adapter == True:
            # use for pred/eval step, not to provide the gold labels in dataset
            feed_dict = {
                model.input_token_indices: dataset.token_indices[dataset_type][i],
                model.input_token_character_indices: dataset.character_indices_padded[dataset_type][i],
                model.input_token_lengths: dataset.token_lengths[dataset_type][i],
                model.dropout_keep_prob: 1.

            }
        else:
            feed_dict = {
                model.input_token_indices: dataset.token_indices[dataset_type][i],
                model.input_token_character_indices: dataset.character_indices_padded[dataset_type][i],
                model.input_token_lengths: dataset.token_lengths[dataset_type][i],
                model.input_label_indices_vector: dataset.label_vector_indices[dataset_type][i],
                model.dropout_keep_prob: 1.
            }
        unary_scores, predictions = sess.run([model.unary_scores, model.predictions], feed_dict)
        if parameters['use_crf']:
            predictions, _ = tf.contrib.crf.viterbi_decode(unary_scores, transition_params_trained)
            predictions = predictions[1:-1]
        else:
            predictions = predictions.tolist()

        assert (len(predictions) == len(dataset.tokens[dataset_type][i]))
        output_string = ''
        prediction_labels = [dataset.index_to_label[prediction] for prediction in predictions]
        gold_labels = dataset.labels[dataset_type][i]
        if parameters['tagging_format'] == 'bioes':
            prediction_labels = utils_nlp.bioes_to_bio(prediction_labels)
            gold_labels = utils_nlp.bioes_to_bio(gold_labels)
        try:
            assert len(prediction_labels) == len(gold_labels)
        except AssertionError:
            print(dataset.tokens[dataset_type][i])
            print(gold_labels)
            print(prediction_labels)

        for z, (prediction, token, gold_label) in enumerate(
                zip(prediction_labels, dataset.tokens[dataset_type][i], gold_labels)):
            while True:
                line = original_conll_file.readline()
                split_line = line.strip().split(' ')
                if '-DOCSTART-' in split_line[0] or len(split_line) == 0 or len(split_line[0]) == 0:
                    continue
                else:
                    token_original = split_line[0]
                    if parameters['tagging_format'] == 'bioes':
                        split_line.pop()
                    gold_label_original = split_line[-1]
                    try:
                        assert (token == dataset.tokenize(token_original) and gold_label == gold_label_original)
                    except AssertionError:
                        print(' '.join([dataset.tokens[dataset_type][i][x] + '/' + gold_labels[x] for x in
                                        range(len(gold_labels))]))
                        print('token: {:s} - gold_label: {:s} - gold_label_original: {:s}'.format(
                            dataset.tokens[dataset_type][i][z], gold_label, gold_label_original))

                    break
            split_line.append(prediction)
            output_string += ' '.join(split_line) + '\n'
        newstr = output_string + '\n'
        output_file.write(newstr)

        if for_adapter == True:
            all_predictions_per_sentence.append(predictions)
            all_y_true_per_sentence.append(dataset.label_indices[dataset_type][i])
            all_prediction_labels_per_sentence.append(prediction_labels)

        all_predictions.extend(predictions)
        all_y_true.extend(dataset.label_indices[dataset_type][i])

    output_file.close()
    original_conll_file.close()

    if dataset_type != 'deploy':
        if parameters['main_evaluation_mode'] == 'conll':
            conll_evaluation_script = os.path.join('.', 'conlleval')
            conll_output_filepath = '{0}_conll_evaluation.txt'.format(output_filepath)
            shell_command = '/usr/bin/perl {0} < {1} > {2}'.format(conll_evaluation_script, output_filepath,
                                                                   conll_output_filepath)
            os.system(shell_command)
            with open(conll_output_filepath, 'r') as f:
                classification_report = f.read()
                print(classification_report)
        else:
            new_y_pred, new_y_true, new_label_indices, new_label_names, _, _ = remap_labels(all_predictions, all_y_true,
                                                                                            dataset, parameters[
                                                                                                'main_evaluation_mode'])

            print(sklearn.metrics.classification_report(new_y_true, new_y_pred, digits=4, labels=new_label_indices,
                                                        target_names=new_label_names))

    if for_adapter == True:
        return all_prediction_labels_per_sentence, all_predictions, all_y_true, output_filepath
    else:
        return all_predictions, all_y_true, output_filepath


def predict_labels(sess, model, transition_params_trained, parameters, dataset,
                   epoch_number, stats_graph_folder, dataset_filepaths, for_adapter=False):
    y_pred = {}
    y_true = {}
    output_filepaths = {}
    y_labels_per_sentence = {}
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        if dataset_type not in dataset_filepaths.keys():
            continue
        prediction_output = prediction_step(sess, dataset, dataset_type, model, transition_params_trained,
                                            stats_graph_folder, epoch_number, parameters, dataset_filepaths,
                                            for_adapter)
        if for_adapter == True:
            y_labels_per_sentence[dataset_type], y_pred[dataset_type], \
            y_true[dataset_type], output_filepaths[dataset_type] = prediction_output
        else:
            y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output
    if for_adapter == True:
        return y_labels_per_sentence, y_pred, y_true, output_filepaths
    else:
        return y_pred, y_true, output_filepaths


def restore_model_parameters_from_pretrained_model(parameters, dataset, sess, model, model_saver):
    pretraining_dataset = pickle.load(open(os.path.join(parameters['pretrained_model_folder'], 'dataset.pickle'), 'rb'))
    pretrained_model_checkpoint_filepath = os.path.join(parameters['pretrained_model_folder'], 'best_model.ckpt')
    print(len(pretraining_dataset.index_to_label))
    print(len(dataset.index_to_label))
    if not parameters['add_class']:
        assert len(pretraining_dataset.index_to_label) == len(dataset.index_to_label)
    elif parameters['tagging_format'] == 'bioes' and parameters['add_class']:
        assert len(pretraining_dataset.index_to_label) + 4 == len(dataset.index_to_label)
    else:
        assert len(pretraining_dataset.index_to_label) + 2 == len(dataset.index_to_label)
    pretraining_parameters = \
        main.load_parameters(parameters_filepath=os.path.join(parameters['pretrained_model_folder'],
                                                              'parameters.ini'), verbose=False)[0]
    for name in ['use_character_lstm', 'character_embedding_dimension', 'character_lstm_hidden_state_dimension',
                 'token_embedding_dimension', 'token_lstm_hidden_state_dimension', 'use_crf']:
        if parameters[name] != pretraining_parameters[name]:
            print("Parameters of the pretrained model:")
            pprint(pretraining_parameters)
            raise AssertionError("The parameter {0} ({1}) is different from the pretrained model ({2}).".format(name,
                                                                                                                parameters[
                                                                                                                    name],
                                                                                                                pretraining_parameters[
                                                                                                                    name]))

    # If the token and character mappings are exactly the same
    if pretraining_dataset.index_to_token == dataset.index_to_token and \
            pretraining_dataset.index_to_character == dataset.index_to_character:

        model_saver = tf.train.import_meta_graph(pretrained_model_checkpoint_filepath)
        model_saver.restore(sess, pretrained_model_checkpoint_filepath_weights)
        last_layer = tf.get_collection('crf/transitions')[0]
        last_layer_shape = last_layer.get_shape().as_list()

        if parameters['tagging_format'] == 'bioes':
            number_of_classes_new = 17
        else:
            number_of_classes_new = 9

        weights_new = tf.Variable(tf.truncated_normal([last_layer_shape[1], number_of_classes_new], stddev=0.05))
        biases_new = tf.Variable(tf.constant(0.05, shape=[number_of_classes_new]))
        output_new = tf.matmul(last_layer, weights_new) + biases_new
        pred = tf.nn.softmax(output_new)

    else:
        # Resize the token and character embedding weights to match them with the pretrained model
        # (required in order to restore the pretrained model)
        utils_tf.resize_tensor_variable(sess, model.character_embedding_weights,
                                        [pretraining_dataset.alphabet_size,
                                         parameters['character_embedding_dimension']])
        utils_tf.resize_tensor_variable(sess, model.token_embedding_weights,
                                        [pretraining_dataset.vocabulary_size,
                                         parameters['token_embedding_dimension']])

        if parameters['tagging_format'] == 'bioes':
            n_new_neuron = 4  # four times the number of new class
        else:
            n_new_neuron = 2
        print("number of new neuron: {:d}".format(n_new_neuron))
        number_of_classes_new = dataset.number_of_classes + n_new_neuron
        print("number of classes (new): {:d}".format(number_of_classes_new))
        model_saver.restore(sess, pretrained_model_checkpoint_filepath)  # [15 x 15]
        graph = tf.get_default_graph()

        model.input_label_indices_vector = tf.placeholder(tf.float32,
                                                          [None, number_of_classes_new],
                                                          name="input_label_indices_vector")
        model.input_label_indices_flat = tf.placeholder(tf.int32, [None], name="input_label_indices_flat")

        if parameters['use_adapter']:
            model.input_label_adapter_indices_vector = tf.placeholder(tf.float32,
                                                                      [None, len(dataset.index_to_label_adapter)],
                                                                      name="input_label_adapter_indices_vector")
            model.input_label_adapter_indices_flat = tf.placeholder(tf.int32, [None],
                                                                    name="input_label_adapter_indices_flat")
            model.adapter_keep_prob = tf.placeholder(tf.float32, name="adapter_keep_prob")

            if parameters['include_pos']:
                model.input_label_pos_indices_vector = tf.placeholder(tf.float32,
                                                                      [None,
                                                                       len(dataset.index_to_label_pos)],
                                                                      name="input_label_pos_indices_vector")
                model.input_label_pos_indices_flat = tf.placeholder(tf.int32, [None],
                                                                    name="input_label_pos_indices_flat")

        old_outputs_w_gradient = graph.get_tensor_by_name('feedforward_after_lstm/output_after_tanh:0')
        old_outputs = old_outputs_w_gradient
        old_last_layer_W = graph.get_tensor_by_name('feedforward_before_crf/W:0')
        old_last_layer_b = graph.get_tensor_by_name('feedforward_before_crf/bias:0')
        print("old last layer W shape")
        print(old_last_layer_W.get_shape())  # .as_list())
        print("old last layer b shape")
        print(old_last_layer_b.get_shape())  # .as_list())

        last_layer_W_mean = np.mean(old_last_layer_W.eval())
        last_layer_W_stddev = np.std(old_last_layer_W.eval())
        last_layer_b_mean = np.mean(old_last_layer_b.eval())
        last_layer_b_stddev = np.std(old_last_layer_b.eval())

        old_W_width, old_W_height = old_last_layer_W.get_shape().as_list()[0], old_last_layer_W.get_shape().as_list()[1]

        last_layer_b_new_col = tf.truncated_normal([n_new_neuron],
                                                   mean=last_layer_b_mean,
                                                   stddev=last_layer_b_stddev)
        last_layer_b_new = tf.concat([old_last_layer_b, last_layer_b_new_col], 0)

        adapter_new_dim = parameters['token_lstm_hidden_state_dimension']
        if parameters['use_adapter']:
            last_layer_W_new_row_adapter = tf.truncated_normal([adapter_new_dim,
                                                                dataset.number_of_classes])
            last_layer_W_new_rows = tf.concat([old_last_layer_W, last_layer_W_new_row_adapter], 0)
            last_layer_W_new_col = tf.truncated_normal([old_W_width + adapter_new_dim,
                                                        n_new_neuron])
            last_layer_W_new = tf.concat([last_layer_W_new_rows, last_layer_W_new_col], 1)

        else:
            last_layer_W_new_col = tf.truncated_normal([old_W_width, n_new_neuron],
                                                       mean=last_layer_W_mean,
                                                       stddev=last_layer_W_stddev)
            last_layer_W_new = tf.concat([old_last_layer_W, last_layer_W_new_col], 1)

        print("new last layer W shape")
        print(last_layer_W_new.get_shape())  # .as_list())
        print("new last layer b shape")
        print(last_layer_b_new.get_shape())  # .as_list())

        if parameters['hard_freeze']:
            last_layer_W_orig = last_layer_W_new[:, 0:dataset.number_of_classes]
            last_layer_b_orig = last_layer_b_new[0:dataset.number_of_classes]
            print("original last layer W (for resetting at each step) shape:")
            print(last_layer_W_orig.get_shape())  # .as_list())
            print("original last layer b (for resetting at each step) shape:")
            print(last_layer_b_orig.get_shape())  # .as_list())
        old_transition_parameters = graph.get_tensor_by_name('crf/transitions:0')
        transition_parameters_mean = np.mean(old_transition_parameters.eval())
        transition_parameters_stddev = np.std(old_transition_parameters.eval())
        trainsition_new_col = tf.truncated_normal([old_transition_parameters.get_shape().as_list()[0], n_new_neuron],
                                                  mean=transition_parameters_mean,
                                                  stddev=transition_parameters_stddev)
        trainsition_new_col = tf.concat([old_transition_parameters, trainsition_new_col], axis=1)
        trainsition_new_row = tf.truncated_normal([n_new_neuron, trainsition_new_col.get_shape().as_list()[1]],
                                                  mean=transition_parameters_mean,
                                                  stddev=transition_parameters_stddev)
        new_transition_parameters = tf.concat([trainsition_new_col, trainsition_new_row], axis=0)
        if parameters['use_adapter']:
            with tf.variable_scope("concat_before_adapter") as vs:
                token_embedded = graph.get_tensor_by_name('concatenate_token_and_character_vectors/token_lstm_input:0')
                if parameters['include_pos']:
                    embed_label_concat = tf.concat([
                        token_embedded,
                        model.input_label_adapter_indices_vector,
                        model.input_label_pos_indices_vector],
                        axis=-1,
                        name='embed_label_concat')
                else:
                    embed_label_concat = tf.concat([token_embedded,
                                                    model.input_label_adapter_indices_vector],
                                                   axis=-1,
                                                   name='embed_label_concat')
                embed_label_concat_expanded = tf.expand_dims(embed_label_concat,
                                                             axis=0,
                                                             name='embed_label_concat_expanded')
            with tf.variable_scope("adapter") as vs:
                initializer = tf.contrib.layers.xavier_initializer()

                adapter_lstm_output = entity_lstm.bidirectional_LSTM(embed_label_concat_expanded,
                                                                     parameters['token_lstm_hidden_state_dimension'],
                                                                     initializer=initializer,
                                                                     output_sequence=True,
                                                                     sum_fw_bw=True)

                adapter_lstm_output_squeezed = tf.squeeze(adapter_lstm_output, axis=0,
                                                          name='adapter_lstm_output_squeezed')
                old_outputs_before_drop = tf.concat([old_outputs, adapter_lstm_output_squeezed], axis=-1)

                old_outputs = tf.nn.dropout(old_outputs_before_drop,
                                            model.adapter_keep_prob,
                                            name='old_outputs_drop')
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='adapter'):
            sess.run(tf.variables_initializer([var]))

        with tf.variable_scope("feedforward_before_crf") as vs:
            model.b_before_crf = tf.get_variable(
                "b_new",
                initializer=last_layer_b_new)
            model.W_before_crf = tf.get_variable(
                "W_new",
                initializer=last_layer_W_new)

            print("W after renewal:")
            print(graph.get_tensor_by_name('feedforward_before_crf/W_new:0').get_shape().as_list())
            print("b after renewal:")
            print(graph.get_tensor_by_name('feedforward_before_crf/b_new:0').get_shape().as_list())

            new_scores = tf.nn.xw_plus_b(old_outputs, model.W_before_crf, model.b_before_crf, name="scores_new")
            model.unary_scores = new_scores
            print("new scores shape")
            print(model.unary_scores.get_shape().as_list())
            model.predictions = tf.argmax(model.unary_scores, 1, name="predictions_new")
            print("new prediction shape")
            print(model.predictions.get_shape().as_list())

        if parameters['use_crf']:
            with tf.variable_scope("crf") as vs:

                print("new number of class: {:d}".format(number_of_classes_new))
                small_score = -1000.0
                large_score = 0.0
                sequence_length = tf.shape(model.unary_scores)[0]
                unary_scores_with_start_and_end = tf.concat(
                    [model.unary_scores, tf.tile(tf.constant(small_score, shape=[1, 2]), [sequence_length, 1])], 1)
                start_unary_scores = [[small_score] * number_of_classes_new + [large_score, small_score]]
                end_unary_scores = [[small_score] * number_of_classes_new + [small_score, large_score]]
                model.unary_scores = tf.concat([start_unary_scores, unary_scores_with_start_and_end, end_unary_scores],
                                               0)
                start_index = number_of_classes_new
                end_index = number_of_classes_new + 1
                input_label_indices_flat_with_start_and_end = tf.concat([tf.constant(start_index, shape=[1]),
                                                                         model.input_label_indices_flat,
                                                                         tf.constant(end_index, shape=[1])],
                                                                        0)
                # Apply CRF layer
                sequence_length = tf.shape(model.unary_scores)[0]
                sequence_lengths = tf.expand_dims(sequence_length, axis=0, name='sequence_lengths_new')
                new_unary_scores_expanded = tf.expand_dims(model.unary_scores, axis=0, name='unary_scores_expanded_new')
                new_input_label_indices_flat_batch = tf.expand_dims(input_label_indices_flat_with_start_and_end,
                                                                    axis=0,
                                                                    name='input_label_indices_flat_batch_new')

                print('new unary_scores_expanded: {0}'.format(new_unary_scores_expanded))
                print('new input_label_indices_flat_batch: {0}'.format(new_input_label_indices_flat_batch))
                print("new sequence_lengths: {0}".format(sequence_lengths))

                log_likelihood, model.transition_parameters = tf.contrib.crf.crf_log_likelihood(
                    new_unary_scores_expanded,
                    new_input_label_indices_flat_batch,
                    sequence_lengths,
                    transition_params=new_transition_parameters)
                model.loss = tf.reduce_mean(-log_likelihood, name='cross_entropy_mean_loss_new')
                model.accuracy = tf.constant(1)

        else:  # not using crf
            with tf.variable_scope("crf") as vs:
                print("new number of class: {:d}".format(number_of_classes_new))
                model.transition_parameters = tf.get_variable(
                    "transitions_new",
                    initializer=new_transition_parameters
                )
            with tf.variable_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=model.unary_scores,
                                                                 labels=model.input_label_indices_vector,
                                                                 name='softmax_new')
                model.loss = tf.reduce_mean(losses, name='cross_entropy_mean_loss_new')
            with tf.variable_scope("accuracy"):
                correct_predictions = tf.equal(model.predictions,
                                               tf.argmax(model.input_label_indices_vector, 1))
                model.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy_new')

        if not parameters['use_crf']:
            sess.run(tf.variables_initializer([model.transition_parameters]))
        sess.run(tf.variables_initializer([model.b_before_crf]))
        sess.run(tf.variables_initializer([model.W_before_crf]))

        model.optimizer = tf.train.MomentumOptimizer(parameters['learning_rate'], 0.8)
        if parameters['hard_freeze']:
            glo_step = model.define_training_procedure(parameters,
                                                       dataset=dataset,
                                                       last_layer_W_orig=last_layer_W_orig,
                                                       last_layer_b_orig=last_layer_b_orig)
        else:
            glo_step = model.define_training_procedure(parameters)

        model.summary_op = tf.summary.merge_all()

        # Get pretrained embeddings
        character_embedding_weights, token_embedding_weights = sess.run(
            [model.character_embedding_weights, model.token_embedding_weights])

        # Restore the sizes of token and character embedding weights
        utils_tf.resize_tensor_variable(sess, model.character_embedding_weights,
                                        [dataset.alphabet_size, parameters['character_embedding_dimension']])
        utils_tf.resize_tensor_variable(sess, model.token_embedding_weights,
                                        [dataset.vocabulary_size, parameters['token_embedding_dimension']])

        # Re-initialize the token and character embedding weights
        sess.run(tf.variables_initializer([model.character_embedding_weights, model.token_embedding_weights]))

        # Load embedding weights from pretrained token embeddings first
        model.load_pretrained_token_embeddings(sess, dataset, parameters)

        # Load embedding weights from pretrained model
        model.load_embeddings_from_pretrained_model(sess, dataset, pretraining_dataset, token_embedding_weights,
                                                    embedding_type='token')
        model.load_embeddings_from_pretrained_model(sess, dataset, pretraining_dataset, character_embedding_weights,
                                                    embedding_type='character')

        del pretraining_dataset
        del character_embedding_weights
        del token_embedding_weights

    # Get transition parameters
    transition_params_trained = sess.run(model.transition_parameters)

    if not parameters['reload_character_embeddings']:
        sess.run(tf.variables_initializer([model.character_embedding_weights]))
    if not parameters['reload_character_lstm']:
        sess.run(tf.variables_initializer(model.character_lstm_variables))
    if not parameters['reload_token_embeddings']:
        sess.run(tf.variables_initializer([model.token_embedding_weights]))
    if not parameters['reload_token_lstm']:
        sess.run(tf.variables_initializer(model.token_lstm_variables))
    if not parameters['reload_feedforward']:
        sess.run(tf.variables_initializer(model.feedforward_variables))
    if not parameters['reload_crf']:
        sess.run(tf.variables_initializer(model.crf_variables))

    return transition_params_trained, model, glo_step

def restore_pretrained_model(parameters, dataset, sess, model, model_saver):
    pretraining_dataset = pickle.load(open(os.path.join(parameters['pretrained_model_folder'], 'dataset.pickle'), 'rb'))
    pretrained_model_checkpoint_filepath = os.path.join(parameters['pretrained_model_folder'], 'best_model.ckpt')

    pretraining_parameters = \
        main.load_parameters(parameters_filepath=os.path.join(parameters['pretrained_model_folder'], 'parameters.ini'),
                             verbose=False)[0]
    for name in ['use_character_lstm', 'character_embedding_dimension', 'character_lstm_hidden_state_dimension',
                 'token_embedding_dimension', 'token_lstm_hidden_state_dimension', 'use_crf']:
        if parameters[name] != pretraining_parameters[name]:
            print("Parameters of the pretrained model:")
            pprint(pretraining_parameters)
            raise AssertionError(
                "The parameter {0} ({1}) is different from the pretrained model ({2}).".format(name, parameters[name],
                                                                                               pretraining_parameters[
                                                                                                   name]))

    if pretraining_dataset.index_to_token == dataset.index_to_token and pretraining_dataset.index_to_character == dataset.index_to_character:

        # Restore the pretrained model
        model_saver.restore(sess,
                            pretrained_model_checkpoint_filepath)
    else:

        utils_tf.resize_tensor_variable(sess, model.character_embedding_weights, [pretraining_dataset.alphabet_size,
                                                                                  parameters[
                                                                                      'character_embedding_dimension']])
        utils_tf.resize_tensor_variable(sess, model.token_embedding_weights,
                                        [pretraining_dataset.vocabulary_size, parameters['token_embedding_dimension']])

        # Restore the pretrained model
        model_saver.restore(sess,
                            pretrained_model_checkpoint_filepath)
        # Get pretrained embeddings
        character_embedding_weights, token_embedding_weights = sess.run(
            [model.character_embedding_weights, model.token_embedding_weights])

        # Restore the sizes of token and character embedding weights
        utils_tf.resize_tensor_variable(sess, model.character_embedding_weights,
                                        [dataset.alphabet_size, parameters['character_embedding_dimension']])
        utils_tf.resize_tensor_variable(sess, model.token_embedding_weights,
                                        [dataset.vocabulary_size, parameters['token_embedding_dimension']])

        # Re-initialize the token and character embedding weights
        sess.run(tf.variables_initializer([model.character_embedding_weights, model.token_embedding_weights]))

        # Load embedding weights from pretrained token embeddings first
        model.load_pretrained_token_embeddings(sess, dataset, parameters)

        # Load embedding weights from pretrained model
        model.load_embeddings_from_pretrained_model(sess, dataset, pretraining_dataset, token_embedding_weights,
                                                    embedding_type='token')
        model.load_embeddings_from_pretrained_model(sess, dataset, pretraining_dataset, character_embedding_weights,
                                                    embedding_type='character')

        del pretraining_dataset
        del character_embedding_weights
        del token_embedding_weights

    # Get transition parameters
    transition_params_trained = sess.run(model.transition_parameters)

    if not parameters['reload_character_embeddings']:
        sess.run(tf.variables_initializer([model.character_embedding_weights]))
    if not parameters['reload_character_lstm']:
        sess.run(tf.variables_initializer(model.character_lstm_variables))
    if not parameters['reload_token_embeddings']:
        sess.run(tf.variables_initializer([model.token_embedding_weights]))
    if not parameters['reload_token_lstm']:
        sess.run(tf.variables_initializer(model.token_lstm_variables))
    if not parameters['reload_feedforward']:
        sess.run(tf.variables_initializer(model.feedforward_variables))
    if not parameters['reload_crf']:
        sess.run(tf.variables_initializer(model.crf_variables))

    return transition_params_trained
