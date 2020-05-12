'''
Convolutional spiking neural network training, testing, and evaluation script. Evaluation can be done outside of this script; however, it is most straightforward to call this
script with mode=train, then mode=test on HPC systems, where in the test mode, the network evaluation is written to disk.
'''

import warnings

warnings.filterwarnings('ignore')
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import brian_no_units
import networkx as nx
import pickle as p
import pandas as pd
import pickle
import numpy as np
import brian as b
import argparse
import timeit
import math
import os
from numba import jit

from sklearn.cluster import KMeans
from struct import unpack
from brian import *

from util import *

# np.set_printoptions(threshold=np.nan, linewidth=200)

# only show log messages of level ERROR or higher
b.log_level_error()

# set these appropriate to your directory structure
# top_level_path = '/Users/emb24/PycharmProjects/stdpNew' #sos.path.join('..', '..')

top_level_path = os.path.join('..', '..')
MNIST_data_path = os.path.join(top_level_path, 'data')
model_name = 'csnn_pc'
results_path = os.path.join(top_level_path, 'results', model_name)

performance_dir = os.path.join(top_level_path, 'performance', model_name)
activity_dir = os.path.join(top_level_path, 'activity', model_name)
weights_dir = os.path.join(top_level_path, 'weights', model_name)
random_dir = os.path.join(top_level_path, 'random', model_name)

'''for d in [performance_dir, activity_dir, weights_dir, random_dir, MNIST_data_path, results_path]:
    if not os.path.isdir(d):
        os.makedirs(d)
'''


def set_weights_most_fired(current_spike_countAudio, current_spike_countVideo):
    '''
    For each convolutional patch, set the weights to those of the neuron which
    fired the most in the last iteration.
    '''

    for conn_name in input_connections:
        if 'X' in conn_name:
            for feature in xrange(conv_featuresVideo):
                # count up the spikes for the neurons in this convolution patch
                column_sums = np.sum(current_spike_countVideo[feature: feature + 1, :], axis=0)

                # find the excitatory neuron which spiked the most
                most_spiked = np.argmax(column_sums)

                # create a "dense" version of the most spiked excitatory neuron's weight
                most_spiked_dense = input_connections[conn_name][:, feature * n_eVideo + most_spiked].todense()

                # set all other neurons' (in the same convolution patch) weights the same as the most-spiked neuron in the patch
                for n in xrange(n_eVideo):
                    if n != most_spiked:
                        other_dense = input_connections[conn_name][:, feature * n_eVideo + n].todense()
                        other_dense[convolution_locationsVideo[n]] = most_spiked_dense[
                            convolution_locationsVideo[most_spiked]]
                        input_connections[conn_name][:, feature * n_eVideo + n] = other_dense
        else:
            for feature in xrange(conv_featuresAudio):
                # count up the spikes for the neurons in this convolution patch
                column_sums = np.sum(current_spike_countAudio[feature: feature + 1, :], axis=0)

                # find the excitatory neuron which spiked the most
                most_spiked = np.argmax(column_sums)

                # create a "dense" version of the most spiked excitatory neuron's weight
                most_spiked_dense = input_connections[conn_name][:, feature * n_eAudio + most_spiked].todense()

                # set all other neurons' (in the same convolution patch) weights the same as the most-spiked neuron in the patch
                for n in xrange(n_eAudio):
                    if n != most_spiked:
                        other_dense = input_connections[conn_name][:, feature * n_eAudio + n].todense()
                        other_dense[convolution_locationsAudio[n]] = most_spiked_dense[
                            convolution_locationsAudio[most_spiked]]
                        input_connections[conn_name][:, feature * n_eAudio + n] = other_dense


def normalize_weights():
    '''
    Squash the input -> excitatory weights to sum to a prespecified number.
    '''
    for conn_name in input_connections:
        if 'X' in conn_name:
            connection = input_connections[conn_name][:].todense()
            for feature in xrange(conv_featuresVideo):
                feature_connection = connection[:, feature * n_eVideo: (feature + 1) * n_eVideo]
                column_sums = np.sum(np.asarray(feature_connection), axis=0)
                column_factors = weight['ee_input'] / column_sums

                for n in xrange(n_eVideo):
                    dense_weights = input_connections[conn_name][:, feature * n_eVideo + n].todense()
                    dense_weights[convolution_locationsVideo[n]] *= column_factors[n]
                    input_connections[conn_name][:, feature * n_eVideo + n] = dense_weights
        else:
            connection = input_connections[conn_name][:].todense()
            for feature in xrange(conv_featuresAudio):
                feature_connection = connection[:, feature * n_eAudio: (feature + 1) * n_eAudio]
                column_sums = np.sum(np.asarray(feature_connection), axis=0)
                column_factors = weight['ee_input'] / column_sums

                for n in xrange(n_eAudio):
                    dense_weights = input_connections[conn_name][:, feature * n_eAudio + n].todense()
                    dense_weights[convolution_locationsAudio[n]] *= column_factors[n]
                    input_connections[conn_name][:, feature * n_eAudio + n] = dense_weights

    for conn_name in connections:

        if 'AeTe' in conn_name and lattice_structure != 'none' and lattice_structure != 'none':

            connection = connections[conn_name][:].todense()
            for featureA, feature in zip(range(conv_featuresAudio), range(conv_featuresVideo)):
                # for feature in xrange(conv_featuresAudio):
                feature_connection = connection[featureA * n_eAudio: (feature + 1) * n_eVideo, :]
                column_sums = np.sum(feature_connection)
                column_factors = weight['ee_recurr'] / column_sums

                for idx in xrange(feature * n_eAudio, (feature + 1) * n_eAudio):
                    connections[conn_name][idx, :] *= column_factors

        elif 'VeTe' in conn_name and lattice_structure != 'none' and lattice_structure != 'none':

            connection = connections[conn_name][:].todense()
            for featureA, feature in zip(range(conv_featuresVideo), range(conv_featuresVideo)):
                # for feature in xrange(conv_featuresAudio):
                feature_connection = connection[featureA * n_eVideo: (feature + 1) * n_eVideo, :]
                column_sums = np.sum(feature_connection)
                column_factors = weight['ee_recurr'] / column_sums

                for idx in xrange(feature * n_eVideo, (feature + 1) * n_eVideo):
                    connections[conn_name][idx, :] *= column_factors


        elif 'VeVe' in conn_name and lattice_structure != 'none' and lattice_structure != 'none':
            connection = connections[conn_name][:].todense()
            for feature in xrange(conv_featuresVideo):
                feature_connection = connection[feature * n_eVideo: (feature + 1) * n_eVideo, :]
                column_sums = np.sum(feature_connection)
                column_factors = weight['ee_recurr'] / column_sums

                for idx in xrange(feature * n_eVideo, (feature + 1) * n_eVideo):
                    connections[conn_name][idx, :] *= column_factors
        elif 'AeAe' in conn_name:
            connection = connections[conn_name][:].todense()
            for feature in xrange(conv_featuresAudio):
                feature_connection = connection[feature * n_eAudio: (feature + 1) * n_eAudio, :]
                column_sums = np.sum(feature_connection)
                column_factors = weight['ee_recurr'] / column_sums

                for idx in xrange(feature * n_eAudio, (feature + 1) * n_eAudio):
                    connections[conn_name][idx, :] *= column_factors

        elif 'AeVe' in conn_name:

            connection = connections[conn_name][:].todense()
            for featureA, feature in zip(range(conv_featuresAudio), range(conv_featuresVideo)):
                # for feature in xrange(conv_featuresAudio):
                feature_connection = connection[featureA * n_eAudio: (feature + 1) * n_eVideo, :]
                column_sums = np.sum(feature_connection)
                column_factors = weight['ee_recurr'] / column_sums

                for idx in xrange(feature * n_eAudio, (feature + 1) * n_eAudio):
                    connections[conn_name][idx, :] *= column_factors

        elif 'VeAe' in conn_name:

            connection = connections[conn_name][:].todense()
            for featureA, feature in zip(range(conv_featuresVideo), range(conv_featuresAudio)):
                # for feature in xrange(conv_featuresVideo):
                feature_connection = connection[featureA * n_eVideo: (feature + 1) * n_eAudio, :]
                column_sums = np.sum(feature_connection)
                column_factors = weight['ee_recurr'] / column_sums

                for idx in xrange(feature * n_eVideo, (feature + 1) * n_eVideo):
                    connections[conn_name][idx, :] *= column_factors


def plot_input(rates):
    '''
    Plot the current input example during the training procedure.
    '''
    fig = b.figure(fig_num, figsize=(5, 5))
    im = b.imshow(rates.reshape((100, 100)), interpolation='nearest', vmin=0, vmax=64, cmap='binary')
    b.colorbar(im)
    b.title('Current input example')
    fig.canvas.draw()

    return im, fig


def plot_inputAudio(rates):
    '''
    Plot the current input example during the training procedure.
    '''
    fig = b.figure(fig_num, figsize=(5, 5))
    im = b.imshow(rates.reshape((40, 388)), interpolation='nearest', vmin=0, vmax=64, cmap='binary')
    b.colorbar(im)
    b.title('Current input example')
    fig.canvas.draw()

    return im, fig


def update_input(rates, im, fig):
    '''
    Update the input image to use for input plotting.
    '''
    im.set_array(rates.reshape((100, 100)))
    fig.canvas.draw()
    return im


def update_inputAudio(rates, im, fig):
    '''
    Update the input image to use for input plotting.
    '''
    im.set_array(rates.reshape((40, 388)))
    fig.canvas.draw()
    return im


def get_2d_input_weightsVideo():
    '''
    Get the weights from the input to excitatory layer and reshape it to be two
    dimensional and square.
    '''
    # specify the desired shape of the reshaped input -> excitatory weights
    rearranged_weights = np.zeros((conv_featuresVideo * conv_sizeVideo, conv_sizeVideo * n_eVideo))

    # get the input -> excitatory synaptic weights
    connection = input_connections['XeVe'][:]

    if sort_euclidean:
        # for each excitatory neuron in this convolution feature
        euclid_dists = np.zeros((n_eVideo, conv_featuresVideo))
        temps = np.zeros((n_eVideo, conv_featuresVideo, n_inputVideo))
        for n in xrange(n_eVideo):
            # for each convolution feature
            for feature in xrange(conv_featuresVideo):
                temp = connection[:,
                       feature * n_eVideo + (n // n_e_sqrtVideo) * n_e_sqrtVideo + (n % n_e_sqrtVideo)].todense()
                if feature == 0:
                    if n == 0:
                        euclid_dists[n, feature] = 0.0
                    else:
                        euclid_dists[n, feature] = np.linalg.norm(
                            temps[0, 0, convolution_locationsVideo[n]] - temp[convolution_locationsVideo[n]])
                else:
                    euclid_dists[n, feature] = np.linalg.norm(
                        temps[n, 0, convolution_locationsVideo[n]] - temp[convolution_locationsVideo[n]])

                temps[n, feature, :] = temp.ravel()

            for idx, feature in enumerate(np.argsort(euclid_dists[n])):
                temp = temps[n, feature]
                rearranged_weights[idx * conv_sizeVideo: (idx + 1) * conv_sizeVideo,
                n * conv_sizeVideo: (n + 1) * conv_sizeVideo] = \
                    temp[convolution_locationsVideo[n]].reshape((conv_sizeVideo, conv_sizeVideo))

    else:
        for n in xrange(n_eVideo):
            for feature in xrange(conv_featuresVideo):
                temp = connection[:,
                       feature * n_eVideo + (n // n_e_sqrtVideo) * n_e_sqrtVideo + (n % n_e_sqrtVideo)].todense()
                print('print')
                # print(temp)
                rearranged_weights[feature * conv_sizeVideo: (feature + 1) * conv_sizeVideo,
                n * conv_sizeVideo: (n + 1) * conv_sizeVideo] = \
                    temp[convolution_locationsVideo[n]].reshape((conv_sizeVideo, conv_sizeVideo))

    # return the rearranged weights to display to the user
    if n_e == 1:
        ceil_sqrt = int(math.ceil(math.sqrt(conv_featuresVideo)))
        square_weights = np.zeros((100 * ceil_sqrt, 100 * ceil_sqrt))
        for n in xrange(conv_featuresVideo):
            square_weights[(n // ceil_sqrt) * 100: ((n // ceil_sqrt) + 1) * 100,
            (n % ceil_sqrt) * 100: ((n % ceil_sqrt) + 1) * 100] = rearranged_weights[n * 100: (n + 1) * 100, :]

        return square_weights.T
    else:
        return rearranged_weights.T


def get_2d_input_weightsAudio():
    '''
    Get the weights from the input to excitatory layer and reshape it to be two
    dimensional and square.
    '''
    # specify the desired shape of the reshaped input -> excitatory weights
    rearranged_weights = np.zeros((conv_featuresAudio * conv_sizeAudio, conv_sizeAudio * n_eAudio))

    # get the input -> excitatory synaptic weights
    connection = input_connections['YeAe'][:]

    if sort_euclidean:
        # for each excitatory neuron in this convolution feature
        euclid_dists = np.zeros((n_eAudio, conv_featuresAudio))
        temps = np.zeros((n_eAudio, conv_featuresAudio, n_inputAudio))
        for n in xrange(n_eAudio):
            # for each convolution feature
            for feature in xrange(conv_featuresAudio):
                temp = connection[:,
                       feature * n_eAudio + (n // n_e_sqrtAudio) * n_e_sqrtAudio + (n % n_e_sqrtAudio)].todense()
                if feature == 0:
                    if n == 0:
                        euclid_dists[n, feature] = 0.0
                    else:
                        euclid_dists[n, feature] = np.linalg.norm(
                            temps[0, 0, convolution_locations[n]] - temp[convolution_locationsAudio[n]])
                else:
                    euclid_dists[n, feature] = np.linalg.norm(
                        temps[n, 0, convolution_locations[n]] - temp[convolution_locationsAudio[n]])

                temps[n, feature, :] = temp.ravel()

            for idx, feature in enumerate(np.argsort(euclid_dists[n])):
                temp = temps[n, feature]
                rearranged_weights[idx * conv_sizeAudio: (idx + 1) * conv_sizeAudio,
                n * conv_sizeAudio: (n + 1) * conv_sizeAudio] = \
                    temp[convolution_locations[n]].reshape((conv_sizeAudio, conv_sizeAudio))

    else:
        for n in xrange(n_eAudio):
            for feature in xrange(conv_featuresAudio):
                temp = connection[:,
                       feature * n_eAudio + (n // n_e_sqrtAudio) * n_e_sqrtAudio + (n % n_e_sqrtAudio)].todense()
                print('print')
                # print(temp)
                rearranged_weights[feature * conv_sizeAudio: (feature + 1) * conv_sizeAudio,
                n * conv_sizeAudio: (n + 1) * conv_sizeAudio] = \
                    temp[convolution_locationsAudio[n]].reshape((conv_sizeAudio, conv_sizeAudio))

    # return the rearranged weights to display to the user
    if n_e == 1:
        ceil_sqrt = int(math.ceil(math.sqrt(conv_featuresAudio)))
        square_weights = np.zeros((100 * ceil_sqrt, 100 * ceil_sqrt))
        for n in xrange(conv_featuresAudio):
            square_weights[(n // ceil_sqrt) * 100: ((n // ceil_sqrt) + 1) * 100,
            (n % ceil_sqrt) * 100: ((n % ceil_sqrt) + 1) * 100] = rearranged_weights[n * 100: (n + 1) * 100, :]

        return square_weights.T
    else:
        return rearranged_weights.T


def get_input_weights(weight_matrix):
    '''
    Get the weights from the input to excitatory layer and reshape it to be two
    dimensional and square.
    '''
    weights = []

    # for each convolution feature
    for feature in xrange(conv_features):
        # for each excitatory neuron in this convolution feature
        for n in xrange(n_e):
            temp = weight_matrix[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)]
            weights.append(np.ravel(temp[convolution_locations[n]]))

    # return the rearranged weights to display to the user
    return weights


def plot_2d_input_weightsAudio():
    '''
    Plot the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weightsAudio()

    if n_e != 1:
        fig = plt.figure(fig_num, figsize=(10, 10))  # fig = plt.figure(fig_num, figsize=(18, 9))
    else:
        fig = plt.figure(fig_num, figsize=(6, 6))

    im = plt.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))  # hot_r'))

    if n_e != 1:
        plt.colorbar(im, fraction=0.016)
    else:
        plt.colorbar(im, fraction=0.06)

    plt.title('Convolution weights updates')

    if n_e != 1:
        '''print(n_e)
        print('l')
        print(xrange(conv_size, conv_size * (n_e + 1), conv_size))
        print( xrange(1, n_e + 1))'''
        plt.xticks(xrange(conv_sizeAudio, conv_sizeAudio * (conv_featuresAudio + 1), conv_sizeAudio),
                   xrange(1, conv_featuresAudio + 1))
        plt.yticks(xrange(conv_sizeAudio, conv_sizeAudio * (n_e + 1), conv_sizeAudio), xrange(1, n_eAudio + 1))

        # (conv_features * conv_size, conv_size * n_e)
        plt.xlabel('Convolution feature Audio')
        plt.ylabel('Location in input')  # (from top left to bottom right')

    fig.canvas.draw()
    return im, fig


def plot_2d_input_weightsVideo():
    '''
    Plot the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weightsVideo()

    if n_e != 1:
        fig = plt.figure(fig_num, figsize=(10, 10))  # fig = plt.figure(fig_num, figsize=(18, 9))
    else:
        fig = plt.figure(fig_num, figsize=(6, 6))

    im = plt.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))  # hot_r'))

    if n_e != 1:
        plt.colorbar(im, fraction=0.016)
    else:
        plt.colorbar(im, fraction=0.06)

    plt.title('Convolution weights updates')

    if n_e != 1:
        '''print(n_e)
        print('l')
        print(xrange(conv_size, conv_size * (n_e + 1), conv_size))
        print( xrange(1, n_e + 1))'''
        plt.xticks(xrange(conv_sizeVideo, conv_sizeVideo * (conv_featuresVideo + 1), conv_sizeVideo),
                   xrange(1, conv_featuresVideo + 1))
        plt.yticks(xrange(conv_sizeVideo, conv_sizeVideo * (n_eVideo + 1), conv_sizeVideo), xrange(1, n_eVideo + 1))

        # (conv_features * conv_size, conv_size * n_e)
        plt.xlabel('Convolution feature Video')
        plt.ylabel('Location in input')  # (from top left to bottom right')

    fig.canvas.draw()
    return im, fig


def update_2d_input_weights(im, fig):
    '''
    Update the plot of the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weightsAudio()
    im.set_array(weights)
    fig.canvas.draw()

    return im


def get_patch_weights():
    '''
    Get the weights from the input to excitatory layer and reshape them.
    '''
    rearranged_weights = np.zeros((conv_features * n_e, conv_features * n_e))
    connection = connections['AeAe'][:]

    for feature in xrange(conv_features):
        for other_feature in xrange(conv_features):
            if feature != other_feature:
                for this_n in xrange(n_e):
                    for other_n in xrange(n_e):
                        if is_lattice_connection(n_e_sqrt, this_n, other_n, lattice_structure):
                            rearranged_weights[feature * n_e + this_n, other_feature * n_e + other_n] = connection[
                                feature * n_e + this_n, other_feature * n_e + other_n]

    return rearranged_weights


def plot_patch_weights():
    '''
    Plot the weights between convolution patches to view during training.
    '''
    weights = get_patch_weights()
    fig = b.figure(fig_num, figsize=(5, 5))
    im = b.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r'))
    for idx in xrange(n_e, n_e * conv_features, n_e):
        b.axvline(idx, ls='--', lw=1)
        b.axhline(idx, ls='--', lw=1)
    b.colorbar(im)
    b.title('Between-patch connectivity')

    fig.canvas.draw()
    return im, fig


def update_patch_weights(im, fig):
    '''
    Update the plot of the weights between convolution patches to view during training.
    '''
    weights = get_patch_weights()
    im.set_array(weights)
    fig.canvas.draw()

    return im


def plot_neuron_votes(assignments, spike_rates):
    '''
    Plot the votes of the neurons per label.
    '''
    all_summed_rates = [0] * 6
    num_assignments = [0] * 6
    print(assignments)

    for i in xrange(6):
        num_assignments[i] = len(np.where(assignments == i)[0])
        print('i' + str(i))

        print('assi')
        print(num_assignments[i])
        if num_assignments[i] > 0:
            # print(all_summed_rates[i])
            all_summed_rates[i] = np.sum(spike_rates[:, assignments == i]) / num_assignments[i]

    fig = plt.figure(fig_num, figsize=(6, 4))
    rects = plt.bar(xrange(6), [0.1] * 6, align='center')

    plt.ylim([0, 1])
    plt.xticks(xrange(6))
    plt.title('Percentage votes per label')
    plt.show()
    fig.canvas.draw()
    return rects, fig


def update_neuron_votes(rects, fig, spike_rates):
    '''
    Update the plot of the votes of the neurons by label.
    '''
    all_summed_rates = [0] * 6
    num_assignments = [0] * 6

    for i in xrange(6):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            all_summed_rates[i] = np.sum(spike_rates[:, assignments == i]) / num_assignments[i]

    total_votes = np.sum(all_summed_rates)

    if total_votes != 0:
        for rect, h in zip(rects, all_summed_rates):
            rect.set_height(h / float(total_votes))

    fig.canvas.draw()

    return rects


def get_current_performance(performances, current_example_num):
    '''
    Evaluate the performance of the network on the past 'update_interval' training
    examples.
    '''
    global input_numbers

    current_evaluation = int(current_example_num / update_interval)
    start_num = current_example_num - update_interval
    end_num = current_example_num

    for performance in performances.keys():
        difference = output_numbers[performance][start_num: end_num, 0] - input_numbers[start_num: end_num]
        correct = len(np.where(difference == 0)[0])
        performances[performance][current_evaluation] = correct / float(update_interval) * 100

    return performances


def plot_performance(fig_num, performances, num_evaluations):
    '''
    Set up the performance plot for the beginning of the simulation.
    '''
    time_steps = range(0, num_evaluations)

    fig = plt.figure(fig_num, figsize=(12, 4))
    fig_num += 1

    for performance in performances:
        plt.plot(time_steps, performances[performance], label=performance)

    lines = plt.gca().lines

    plt.ylim(ymax=100)
    plt.xticks(xrange(0, num_evaluations + 6, 6), xrange(0, ((num_evaluations + 6) * update_interval), 6))
    plt.legend()
    plt.title('Classification performance per update interval')

    fig.canvas.draw()

    return lines, fig_num, fig


def update_performance_plot(lines, performances, current_example_num, fig):
    '''
    Update the plot of the performance based on results thus far.
    '''
    performances = get_current_performance(performances, current_example_num)

    for line, performance in zip(lines, performances):
        line.set_ydata(performances[performance])

    fig.canvas.draw()

    return lines, performances


def predict_label(assignments, input_numbers, spike_rates):
    '''
    Given the label assignments of the excitatory layer and their spike rates over
    the past 'update_interval', get the ranking of each of the categories of input.
    '''
    most_spiked_summed_rates = [0] * 6
    num_assignments = [0] * 6

    most_spiked_array = np.array(np.zeros((conv_features, n_e)), dtype=bool)

    for feature in xrange(conv_features):
        # count up the spikes for the neurons in this convolution patch
        column_sums = np.sum(spike_rates[feature: feature + 1, :], axis=0)

        # find the excitatory neuron which spiked the most
        most_spiked_array[feature, np.argmax(column_sums)] = True

    # for each label
    for i in xrange(6):
        # get the number of label assignments of this type
        num_assignments[i] = len(np.where(assignments[most_spiked_array] == i)[0])

        if len(spike_rates[np.where(assignments[most_spiked_array] == i)]) > 0:
            # sum the spike rates of all excitatory neurons with this label, which fired the most in its patch
            most_spiked_summed_rates[i] = np.sum(
                spike_rates[np.where(np.logical_and(assignments == i, most_spiked_array))]) / float(
                np.sum(spike_rates[most_spiked_array]))

    all_summed_rates = [0] * 6
    num_assignments = [0] * 6

    for i in xrange(6):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
            all_summed_rates[i] = np.sum(spike_rates[assignments == i]) / num_assignments[i]

    top_percent_summed_rates = [0] * 6
    num_assignments = [0] * 6

    top_percent_array = np.array(np.zeros((conv_features, n_e)), dtype=bool)
    top_percent_array[np.where(spike_rates > np.percentile(spike_rates, 100 - top_percent))] = True

    # for each label
    for i in xrange(6):
        # get the number of label assignments of this type
        num_assignments[i] = len(np.where(assignments[top_percent_array] == i)[0])

        if len(np.where(assignments[top_percent_array] == i)) > 0:
            # sum the spike rates of all excitatory neurons with this label, which fired the most in its patch
            top_percent_summed_rates[i] = len(
                spike_rates[np.where(np.logical_and(assignments == i, top_percent_array))])

    # print(all_summed_rates)
    return (np.argsort(summed_rates)[::-1] for summed_rates in
            (all_summed_rates, most_spiked_summed_rates, top_percent_summed_rates))


def assign_labels(result_monitor, input_numbersv):
    '''
    Based on the results from the previous 'update_interval', assign labels to the
    excitatory neurons.
    '''
    assignments = np.ones((conv_featuresAudio, n_eAudio))
    input_nums = np.asarray(input_numbersv)
    maximum_rate = np.zeros(conv_featuresAudio * n_eAudio)

    for j in xrange(6):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis=0) / num_assignments

            for i in xrange(conv_featuresAudio * n_eAudio):
                if rate[i // n_eAudio, i % n_eAudio] > maximum_rate[i]:
                    maximum_rate[i] = rate[i // n_eAudio, i % n_eAudio]
                    assignments[i // n_eAudio, i % n_eAudio] = j

    return assignments


def build_network():
    global fig_num


    # Audio layer
    neuron_groups['ea'] = b.NeuronGroup(n_e_totalAudio, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e,
                                        reset=scr_e,
                                        compile=True, freeze=True)
    neuron_groups['ia'] = b.NeuronGroup(n_e_totalAudio, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i,
                                        reset=v_reset_i, compile=True, freeze=True)


    # Video layer
    neuron_groups['ev'] = b.NeuronGroup(n_e_totalVideo, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e,
                                        reset=scr_e,
                                        compile=True, freeze=True)

    neuron_groups['iv'] = b.NeuronGroup(n_e_totalVideo, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i,
                                        reset=v_reset_i, compile=True, freeze=True)

    # convergence layer
    neuron_groups['e'] = b.NeuronGroup(n_e_total, neuron_eqs_e, threshold=v_thresh_eConv, refractory=refrac_e,
                                       reset=scr_e,
                                       compile=True, freeze=True)

    neuron_groups['i'] = b.NeuronGroup(n_e_total, neuron_eqs_i, threshold=v_thresh_iConv, refractory=refrac_i,
                                       reset=v_reset_i, compile=True, freeze=True)

    for name in population_names:
        print '...Creating neuron group:', name

        # get a subgroup of size 'n_e' from all exc

        # Subgroup VIDEO
        if name == 'V':
            neuron_groups[name + 'e'] = neuron_groups['ev'].subgroup(conv_featuresVideo * n_eVideo)
            # get a subgroup of size 'n_i' from the inhibitory layer
            neuron_groups[name + 'i'] = neuron_groups['iv'].subgroup(conv_featuresVideo * n_eVideo)

        elif name == 'A':
            # Subgroup Audio

            neuron_groups[name + 'e'] = neuron_groups['ea'].subgroup(conv_featuresAudio * n_eAudio)
            # get a subgroup of size 'n_i' from the inhibitory layer
            neuron_groups[name + 'i'] = neuron_groups['ia'].subgroup(conv_featuresAudio * n_eAudio)

        else:
            # Total Convergence layer

            neuron_groups[name + 'e'] = neuron_groups['e'].subgroup(conv_features * n_e)
            # get a subgroup of size 'n_i' from the inhibitory layer
            neuron_groups[name + 'i'] = neuron_groups['i'].subgroup(conv_features * n_e)

        # start the membrane potentials of these groups 40mV below their resting potentials
        neuron_groups[name + 'e'].v = v_rest_e - 40. * b.mV
        neuron_groups[name + 'i'].v = v_rest_i - 40. * b.mV

        print '...Creating recurrent connections'

    # Add connection from and to convergence layer

    for conn_type in recurrent_conn_names:

        if conn_type == 'ee':
            # create connection name from audio and video to the concergence layer

            conn_name = 'VeTe'

            if test_mode:
                weight_matrix = np.load(conn_name + 'N_' + ending + '.npy')
            else:

                connections[conn_name] = b.Connection(neuron_groups['Ve'], neuron_groups['Te'],
                                                      structure='sparse', state='g' + conn_type[0])

            conn_name = 'AeTe'

            if test_mode:
                weight_matrix = np.load(conn_name + 'N_' + ending + '.npy')
            else:

                connections[conn_name] = b.Connection(neuron_groups['Ae'], neuron_groups['Te'],
                                                      structure='sparse', state='g' + conn_type[0])
            if connectivity == 'none':
                pass

            conn_name = 'TeTe'
            if test_mode:
                weight_matrix = np.load(conn_name + 'N_' + ending + '.npy')
            else:
                connections[conn_name] = b.Connection(neuron_groups['Te'], neuron_groups['Te'], structure='sparse',
                                                      state='g' + conn_type[0])
            if connectivity == 'none':
                pass

    colorcode = ['r', 'c', 'y']  # np.array([[1, 1, 0],[1, 0, 0],[0, 1, 0]])
    c = 0
    for name in population_names:

        if test_mode:
            # load up adaptive threshold parameters
            # neuron_groups['e'].theta = np.load(os.path.join(weights_dir, 'theta_A' + '_' + ending +'.npy'))
            if 'V' in name:
                neuron_groups['ev'].theta = np.load('theta_V' + 'N_' + ending + '.npy')
            elif 'A' in name:
                neuron_groups['ea'].theta = np.load('theta_A' + 'N_' + ending + '.npy')

            else:
                neuron_groups['e'].theta = np.load('theta_T' + 'N_' + ending + '.npy')


        else:
            # set the adaptive additive threshold parameter at 20mV
            if 'V' in name:

                neuron_groups['ev'].theta = np.ones((n_e_totalVideo)) * 20.0 * b.mV
            elif 'A' in name:
                neuron_groups['ea'].theta = np.ones((n_e_totalAudio)) * 20.0 * b.mV
            else:
                neuron_groups['e'].theta = np.ones((n_e_total)) * 20.0 * b.mV

        for conn_type in recurrent_conn_names:
            if conn_type == 'ei':
                # create connection name (composed of population and connection types)

                conn_name = name + conn_type[0] + name + conn_type[1]
                # If this is the ocnvergence layers connenct both audio and video
                '''if 'T' in name:
                    connections[conn_name] = b.Connection(neuron_groups['Ae'], neuron_groups['Vi'],
                                                          structure='sparse', state='g' + conn_type[0])
                else:'''

                # create a connection from the first group in conn_name with the second group
                connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]],
                                                      structure='sparse', state='g' + conn_type[0])
                # instantiate the created connection

                if 'V' in name:

                    for feature in xrange(conv_featuresVideo):
                        for n in xrange(n_eVideo):
                            connections[conn_name][feature * n_eVideo + n, feature * n_eVideo + n] = 10.4
                elif 'A' in name:
                    for feature in xrange(conv_featuresAudio):
                        for n in xrange(n_eAudio):
                            connections[conn_name][feature * n_eAudio + n, feature * n_eAudio + n] = 10.4

                else: # for the convergence layer
                    for feature in xrange(conv_features):
                        for n in xrange(n_e):
                            connections[conn_name][feature * n_e + n, feature * n_e + n] = 10.4







            elif conn_type == 'ie':
                # create connection name (composed of population and connection types)
                conn_name = name + conn_type[0] + name + conn_type[1]
                '''if 'T' in name:
                    connections[conn_name] = b.Connection(neuron_groups['Ai'], neuron_groups['Ve'],
                                                          structure='sparse', state='g' + conn_type[0])
                else:'''
                # create a connection from the first group in conn_name with the second group
                connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]],
                                                      structure='sparse', state='g' + conn_type[0])

                if 'V' in name:
                    # instantiate the created connection
                    for feature in xrange(conv_featuresVideo):
                        for other_feature in xrange(conv_featuresVideo):
                            if feature != other_feature:
                                for n in xrange(n_eVideo):
                                    connections[conn_name][feature * n_eVideo + n, other_feature * n_eVideo + n] = 17.4

                    if random_inhibition_prob != 0.0:
                        for feature in xrange(conv_featuresVideo):
                            for other_feature in xrange(conv_featuresVideo):
                                for n_this in xrange(n_eVideo):
                                    for n_other in xrange(n_eVideo):
                                        if n_this != n_other:
                                            if b.random() < random_inhibition_prob:
                                                connections[conn_name][
                                                    feature * n_eVideo + n_this, other_feature * n_eVideo + n_other] = 17.4

                elif 'A' in name:
                    # instantiate the created connection
                    for feature in xrange(conv_featuresAudio):
                        for other_feature in xrange(conv_featuresAudio):
                            if feature != other_feature:
                                for n in xrange(n_eAudio):
                                    connections[conn_name][feature * n_eAudio + n, other_feature * n_eAudio + n] = 17.4

                    if random_inhibition_prob != 0.0:
                        for feature in xrange(conv_featuresAudio):
                            for other_feature in xrange(conv_featuresAudio):
                                for n_this in xrange(n_eAudio):
                                    for n_other in xrange(n_eAudio):
                                        if n_this != n_other:
                                            if b.random() < random_inhibition_prob:
                                                connections[conn_name][
                                                    feature * n_eAudio + n_this, other_feature * n_eAudio + n_other] = 17.4

                else:

                    for feature in xrange(conv_features):
                        for other_feature in xrange(conv_features):
                            if feature != other_feature:
                                for n in xrange(n_e):
                                    connections[conn_name][feature * n_e + n, other_feature * n_e + n] = 17.4



            elif conn_type == 'ee':
                # create connection name (composed of population and connection types)
                conn_name = name + conn_type[0] + name + conn_type[1]
                if 'T' in name:

                    if test_mode:
                        weight_matrix = np.load(conn_name + 'N_' + ending + '.npy')
                    else:

                        connections['AeTe'] = b.Connection(neuron_groups['Ae'], neuron_groups['Te'],
                                                           structure='sparse', state='g' + conn_type[0])

                        connections['VeTe'] = b.Connection(neuron_groups['Ve'], neuron_groups['Te'],
                                                           structure='sparse', state='g' + conn_type[0])
                else:
                    # get weights from file if we are in test mode
                    if test_mode:
                        weight_matrix = np.load(conn_name + 'N_' + ending + '.npy')
                    # create a connection from the first group in conn_name with the second group
                    connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]],
                                                          structure='sparse', state='g' + conn_type[0])


        # if STDP from excitatory -> excitatory is on and this connection is excitatory -> excitatory
        if ee_STDP_on and 'ee' in recurrent_conn_names:
            # stdp_methods[name + 'e' + name + 'e'] = b.STDP(connections[name + 'e' + name + 'e'], eqs=eqs_stdp_ee,
            #                                             pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0.,
            #                                               wmax=wmax_ee)
            stdp_methods['AeTe'] = b.STDP(connections['AeTe'], eqs=eqs_stdp_ee,
                                          pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0.,
                                          wmax=wmax_ee)
            stdp_methods['VeTe'] = b.STDP(connections['VeTe'], eqs=eqs_stdp_ee,
                                          pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0.,
                                          wmax=wmax_ee)

            '''stdp_methods['AeVe'] = b.STDP(connections['AeVe'], eqs=eqs_stdp_ee, pre=eqs_stdp_pre_ee,
                                          post=eqs_stdp_post_ee, wmin=0., wmax=wmax_ee)

            stdp_methods['VeAe'] = b.STDP(connections['VeAe'], eqs=eqs_stdp_ee, pre=eqs_stdp_pre_ee,
                                             post=eqs_stdp_post_ee, wmin=0., wmax=wmax_ee)'''

        print '...Creating monitors for:', name

        # spike rate monitors for excitatory and inhibitory neuron populations
        rate_monitors[name + 'e'] = b.PopulationRateMonitor(neuron_groups[name + 'e'],
                                                            bin=(single_example_time + resting_time) / b.second)
        rate_monitors[name + 'i'] = b.PopulationRateMonitor(neuron_groups[name + 'i'],
                                                            bin=(single_example_time + resting_time) / b.second)
        spike_counters[name + 'e'] = b.SpikeCounter(neuron_groups[name + 'e'])

        # record neuron population spikes if specified
        if record_spikes:
            spike_monitors[name + 'e'] = b.SpikeMonitor(neuron_groups[name + 'e'])
            spike_monitors[name + 'i'] = b.SpikeMonitor(neuron_groups[name + 'i'])

        if record_spikes and do_plot:
            b.figure(fig_num, figsize=(8, 2))

            # fig_num += 1

            b.ion()
            b.subplot(211)
            print(name)
            b.raster_plot(spike_monitors[name + 'e'], refresh=100 * b.ms, showlast=1000 * b.ms,
                          title='Excitatory spikes per neuron ', color=colorcode[c])
            b.subplot(212)
            b.raster_plot(spike_monitors[name + 'i'], refresh=100 * b.ms, showlast=1000 * b.ms,
                          title='Inhibitory spikes per neuron ', color=colorcode[c])
            b.tight_layout()
            c += 1

    lattice_locations = {}

    # setting up parameters for weight normalization between patches
    num_lattice_connections = sum([len(value) for value in lattice_locations.values()])
    weight['ee_recurr'] = (num_lattice_connections / conv_features) * 0.15

    # creating Poission spike train from input image (784 vector, 100x100 image)
    for name in input_population_names:

        if 'X' in name:

            input_groups[name + 'e'] = b.PoissonGroup(n_inputVideo, 0)
            rate_monitors[name + 'e'] = b.PopulationRateMonitor(input_groups[name + 'e'],
                                                                bin=(single_example_time + resting_time) / b.second)
        else:
            input_groups[name + 'e'] = b.PoissonGroup(n_inputAudio, 0)
            rate_monitors[name + 'e'] = b.PopulationRateMonitor(input_groups[name + 'e'],
                                                                bin=(single_example_time + resting_time) / b.second)

    # creating connections from input Poisson spike train to convolution patch populations
    for name in input_connection_names:
        print '\n...Creating connections between', name[0], 'and', name[1]

        # for each of the input connection types (in this case, excitatory -> excitatory)
        for conn_type in input_conn_names:
            # saved connection name
            conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]

            # get weight matrix depending on training or test phase
            if test_mode:
                # weight_matrix = np.load(os.path.join(weights_dir, conn_name + '_' + ending + '.npy'))
                weight_matrix = np.load(conn_name + 'N_' + ending + '.npy')
            # weight_matrix[weight_matrix < 0.20] = 0

            # create connections from the windows of the input group to the neuron population
            if 'X' in name:

                input_connections[conn_name] = b.Connection(input_groups['Xe'], neuron_groups[name[1] + conn_type[1]],
                                                            structure='sparse', state='g' + conn_type[0], delay=True,
                                                            max_delay=delay[conn_type][1])
            else:
                input_connections[conn_name] = b.Connection(input_groups['Ye'], neuron_groups[name[1] + conn_type[1]],
                                                            structure='sparse', state='g' + conn_type[0], delay=True,
                                                            max_delay=delay[conn_type][1])

            if test_mode:
                if 'X' in name:

                    for feature in xrange(conv_featuresVideo):
                        for n in xrange(n_eVideo):
                            for idx in xrange(conv_sizeVideo ** 2):
                                input_connections[conn_name][
                                    convolution_locationsVideo[n][idx], feature * n_eVideo + n] = \
                                    weight_matrix[convolution_locationsVideo[n][idx], feature * n_eVideo + n]
                else:

                    for feature in xrange(conv_featuresAudio):
                        for n in xrange(n_eAudio):
                            for idx in xrange(conv_sizeAudio ** 2):
                                input_connections[conn_name][
                                    convolution_locationsAudio[n][idx], feature * n_eAudio + n] = \
                                    weight_matrix[convolution_locationsAudio[n][idx], feature * n_eAudio + n]


            else:

                if 'X' in name:
                    for feature in xrange(conv_featuresVideo):
                        for n in xrange(n_eVideo):
                            for idx in xrange(conv_sizeVideo ** 2):
                                input_connections[conn_name][
                                    convolution_locationsVideo[n][idx], feature * n_eVideo + n] = (
                                                                                                          b.random() + 0.01) * 0.3
                else:
                    for feature in xrange(conv_featuresAudio):
                        for n in xrange(n_eAudio):
                            for idx in xrange(conv_sizeAudio ** 2):
                                input_connections[conn_name][
                                    convolution_locationsAudio[n][idx], feature * n_eAudio + n] = (
                                                                                                          b.random() + 0.01) * 0.3

            if test_mode:
                # normalize_weights()

                if do_plot:
                    plot_2d_input_weightsAudio()
                    plot_2d_input_weightsVideo()

                    fig_num += 1

        # if excitatory -> excitatory STDP is specified, add it here (input to excitatory populations)
        if ee_STDP_on:
            print '...Creating STDP for connection', name

            # STDP connection name
            conn_name = name[0] + conn_type[0] + name[1] + conn_type[1]
            # create the STDP object
            stdp_methods[conn_name] = b.STDP(input_connections[conn_name], eqs=eqs_stdp_ee, pre=eqs_stdp_pre_ee,
                                             post=eqs_stdp_post_ee, wmin=0., wmax=wmax_ee)

    print '\n'


@jit
def simulation_loop():
    j = 0
    while j < num_examplesVideo:

        # fetched rates depend on training / test phase, and whether we use the
        # testing dataset for the test phase
        if test_mode:
            ratesVideo = (dataVideo[j % data_sizeVideo, :, :] / 8.0) * input_intensityVideo
        else:
            # ensure weights don't grow without bound
            normalize_weights()
            # get the firing rates of the next input example
            ratesVideo = (dataVideo[j % data_sizeVideo, :, :] / 8.0) * input_intensityVideo

            # plot the input at this step
        # input_image_monitorVideo = update_input(ratesVideo, input_image_monitorVideo, input_imageVideo)
        if do_plot:
            input_image_monitorVideo = update_input(ratesVideo, input_image_monitorVideo, input_imageVideo)
            input_image_monitorAudio = update_input(ratesAudio, input_image_monitorAudio, input_imageAudio)

        # sets the input firing rates
        input_groups['Xe'].rate = ratesVideo.reshape(n_inputVideo)

        # get next frame

        single_example_timev = 0.05 * b.second
        b.run(single_example_timev)

        # get new neuron label assignments every 'update_interval'
        '''if y % update_interval == 0 and y > 0:
            assignments = assign_labels(result_monitorVideo[:], input_numbersVideo[y - update_interval: y])'''

        # get count of spikes over the past iteration
        current_spike_countVideo = np.copy(spike_counters['Ve'].count[:]).reshape(
            (conv_featuresVideo, n_eVideo)) - previous_spike_countVideo
        previous_spike_countVideo = np.copy(spike_counters['Ve'].count[:]).reshape((conv_featuresVideo, n_eVideo))

        if test_mode:
            Framecounter = testingFrameCounter[y]
        else:
            Framecounter = trainingFrameCounter[y]

            # after all frames of a video get audio spikes
        current_spike_countAudio = np.copy(spike_counters['Ae'].count[:]).reshape(
            (conv_featuresAudio, n_eAudio))

        if CounterFrames == int(Framecounter):
            # run the network for a single example time
            # print('video')

            # en of one video

            print('frm:' + str(CounterFrames))
            if test_mode:
                ratesAudio = (dataAudio['x'][y % data_sizeAudio, :, :] / 8.0) * input_intensityAudio
            else:
                # ensure weights don't grow without bound
                normalize_weights()
                # get the firing rates of the next input example
                ratesAudio = (dataAudio['x'][y % data_sizeAudio, :, :] / 8.0) * input_intensityAudio

            # ratesAudio = abs(ratesAudio)
            input_groups['Ye'].rate = ratesAudio.reshape(n_inputAudio)

            test = spike_monitors['Ae']

            if y % update_interval == 0 and y > 0:
                assignments = assign_labels(result_monitorAudio[:], input_numbersAudio[y - update_interval: y])

            single_example_timea = 0.15 * b.second
            b.run(single_example_timea)

            current_spike_countAudio = np.copy(spike_counters['Ae'].count[:]).reshape(
                (conv_featuresAudio, n_eAudio)) - previous_spike_countAudio
            previous_spike_countAudio = np.copy(spike_counters['Ae'].count[:]).reshape((conv_featuresAudio, n_eAudio))

            # neuron_groups['Te'].rate = spike_monitors['Ae'].source[:]

            # set weights to those of the most-fired neuron
            if not test_mode and weight_sharing == 'weight_sharing':
                if CounterFrames == int(Framecounter):
                    set_weights_most_fired(current_spike_countAudio, current_spike_countVideo)
                else:

                    set_weights_most_fired(current_spike_countAudio, current_spike_countVideo)

            current_spike_countConv = np.copy(spike_counters['Te'].count[:]).reshape(
                (conv_features, n_e)) - previous_spike_countConv
            previous_spike_countConv = np.copy(spike_counters['Te'].count[:]).reshape((conv_features, n_e))

            current_spike_countConv = current_spike_countAudio
            # update weights every 'weight_update_interval'
            if y % weight_update_interval == 0 and not test_mode and do_plot:
                update_2d_input_weights(input_weight_monitorAudio, fig_weightsAudio)
                if connectivity != 'none':
                    update_patch_weights(patch_weight_monitor, fig2_weights)

            if not test_mode and do_plot:
                update_neuron_votes(neuron_rectsAudio, fig_neuron_votesAudio, result_monitorAudio[:])

            # if the neurons in the network didn't spike more than four times
            t = np.sum(current_spike_countAudio)
            # current_spike_countAudio = previous_spike_countConv
            print (t)
            if np.sum(current_spike_countConv) < 5 and num_retries < 6:
                # increase the intensity of input
                input_intensity += 2
                num_retries += 1

                # set all network firing rates to zero
                for name in input_population_names:
                    input_groups[name + 'e'].rate = 0

                # let the network relax back to equilibrium
                b.run(resting_time)

            # otherwise, record results and continue simulation
            else:
                num_retries = 0
                # record the current number of spikes
                result_monitorVideo[j % update_interval, :] = current_spike_countVideo

                # decide whether to evaluate on test or training set
                if test_mode:
                    input_numbersVideo[j] = labelsVideo[j % data_sizeVideo]
                else:
                    input_numbersVideo[j] = labelsVideo['y'][j % data_sizeVideo]

                # if CounterFrames == int(Framecounter):

                # reset frame counter
                #    CounterFrames = 0

                result_monitorAudio[y % update_interval, :] = current_spike_countAudio

                result_monitor[y % update_interval, :] = current_spike_countAudio

                # decide whether to evaluate on test or training set
                if test_mode:
                    input_numbersAudio[y] = dataAudio['y'][y % data_sizeAudio]
                else:
                    input_numbersAudio[y] = dataAudio['y'][y % data_sizeAudio]

                # get the output classifications of the network
                output_numbers['all'][y, :], output_numbers['most_spiked'][y, :], output_numbers['top_percent'][y, :] = \
                    predict_label(assignments, input_numbersAudio[y - update_interval - (y % update_interval): y - \
                                                                                                               (
                                                                                                                       y % update_interval)],
                                  result_monitorAudio[y % update_interval, :])

                # print progress
                if y % print_progress_interval == 0 and y > 0:
                    print 'runs done:', y, 'of', int(
                        num_examplesAudio), '(time taken for past', print_progress_interval, 'runs:', str(
                        timeit.default_timer() - start_time) + ')'
                    start_time = timeit.default_timer()

                # plot performance if appropriate
                print(y)
                if y % update_interval == 0 and y > 0:
                    if not test_mode and do_plot:
                        # updating the performance plot
                        perf_plot, performances = update_performance_plot(performance_monitor, performances, y,
                                                                          fig_performance)
                    else:
                        performances = get_current_performance(performances, y)

                    # pickling performance recording and iteration number
                    # p.dump((j, performances), open(os.path.join(performance_dir, ending + '.p'), 'wb'))

                    for performance in performances:
                        print '\nClassification performance (' + performance + ')', performances[performance][
                                                                                    1:int(y / float(
                                                                                        update_interval)) + 1], \
                            '\nAverage performance:', sum(
                            performances[performance][1:int(y / float(update_interval)) + 1]) / \
                                                      float(len(performances[performance][
                                                                1:int(y / float(update_interval)) + 1])), '\n'

                        # set input firing rates back to zero
                for name in input_population_names:
                    input_groups[name + 'e'].rate = 0

                # run the network for 'resting_time' to relax back to rest potentials
                b.run(resting_time)
                # bookkeeping
                input_intensity = start_input_intensity + 20
                input_intensityAudio = start_input_intensity + 20
                input_intensityVideo = start_input_intensity + 20

            CounterFrames = 0
            y += 1
            # set weights to those of the most-fired neuron
            if not test_mode and weight_sharing == 'weight_sharing':
                set_weights_most_fired(current_spike_countAudio, current_spike_countVideo)

        j += 1
        CounterFrames += 1


def run_simulation():
    '''
    Logic for running the simulation itself.
    '''
    global fig_num, input_intensity, input_intensityVideo, input_intensityAudio, previous_spike_countConv, previous_spike_countVideo, previous_spike_countAudio, ratesVideo, ratesAudio, rates, assignments, clusters, cluster_assignments, \
        kmeans, kmeans_assignments, current_spike_countAudio, simple_clusters, simple_cluster_assignments, index_matrix

    # plot input weights
    # print(test_mode)
    if not test_mode and do_plot:
        input_weight_monitorAudio, fig_weightsAudio = plot_2d_input_weightsAudio()

        input_weight_monitorVideo, fig_weightsVideo = plot_2d_input_weightsVideo()
        fig_num += 1
        if connectivity != 'none':
            patch_weight_monitor, fig2_weights = plot_patch_weights()
            fig_num += 1
        neuron_rectsVideo, fig_neuron_votesVideo = plot_neuron_votes(assignments, result_monitorVideo[:])
        neuron_rectsAudio, fig_neuron_votesAudio = plot_neuron_votes(assignments, result_monitorAudio[:])
        fig_num += 1

    # plot input intensities
    if do_plot:
        input_image_monitorVideo, input_imageVideo = plot_input(ratesVideo)
        input_image_monitorAudio, input_imageAudio = plot_inputAudio(ratesAudio)
        fig_num += 1

    # set up performance recording and plotting
    num_evaluations = int(num_examplesAudio / update_interval) + 1
    performances = {voting_scheme: np.zeros(num_evaluations) for voting_scheme in ['all', 'most_spiked', 'top_percent']}
    if not test_mode and do_plot:
        performance_monitor, fig_num, fig_performance = plot_performance(fig_num, performances, num_evaluations)
    else:
        performances = get_current_performance(performances, 0)

    # set firing rates to zero initially
    for name in input_population_names:
        input_groups[name + 'e'].rate = 0

    # initialize network
    j = 0
    num_retries = 0
    b.run(0)

    # start recording time
    start_time = timeit.default_timer()

    # FOR each video/Audio segment go through all video image frames then link  to the audio
    y = 0
    # while y < num_examplesAudio:

    CounterFrames = 0
    # Get rates for audio sequences

    # print(trainingFrameCounter)

    # simulation_loop()
    while j < num_examplesVideo:

        # fetched rates depend on training / test phase, and whether we use the
        # testing dataset for the test phase
        if test_mode:
            ratesVideo = (dataVideo['x'][j % data_sizeVideo, :, :] / 8.0) * input_intensityVideo
        else:
            # ensure weights don't grow without bound
            normalize_weights()
            # get the firing rates of the next input example
            ratesVideo = (dataVideo['x'][j % data_sizeVideo, :, :] / 8.0) * input_intensityVideo

            # plot the input at this step
        # input_image_monitorVideo = update_input(ratesVideo, input_image_monitorVideo, input_imageVideo)
        if do_plot:
            input_image_monitorVideo = update_input(ratesVideo, input_image_monitorVideo, input_imageVideo)
            input_image_monitorAudio = update_inputAudio(ratesAudio, input_image_monitorAudio, input_imageAudio)

        # sets the input firing rates
        input_groups['Xe'].rate = ratesVideo.reshape(n_inputVideo)

        # get next frame

        single_example_timev = 0.05 * b.second
        b.run(single_example_timev)

        # get new neuron label assignments every 'update_interval'
        '''if y % update_interval == 0 and y > 0:
            assignments = assign_labels(result_monitorVideo[:], input_numbersVideo[y - update_interval: y])'''

        # get count of spikes over the past iteration
        current_spike_countVideo = np.copy(spike_counters['Ve'].count[:]).reshape(
            (conv_featuresVideo, n_eVideo)) - previous_spike_countVideo
        previous_spike_countVideo = np.copy(spike_counters['Ve'].count[:]).reshape((conv_featuresVideo, n_eVideo))

        if test_mode:
            Framecounter = testingFrameCounter[y]
        else:
            Framecounter = trainingFrameCounter[y]

            # after all frames of a video get audio spikes
        current_spike_countAudio = np.copy(spike_counters['Ae'].count[:]).reshape(
            (conv_featuresAudio, n_eAudio))

        if CounterFrames == int(Framecounter):
            # run the network for a single example time
            # print('video')

            # en of one video

            print('frm:' + str(CounterFrames))
            if test_mode:
                ratesAudio = (dataAudio['x'][y % data_sizeAudio, :, :] / 8.0) * input_intensityAudio
            else:
                # ensure weights don't grow without bound
                normalize_weights()
                # get the firing rates of the next input example
                ratesAudio = (dataAudio['x'][y % data_sizeAudio, :, :] / 8.0) * input_intensityAudio

            # ratesAudio = abs(ratesAudio)
            input_groups['Ye'].rate = ratesAudio.reshape(n_inputAudio)

            test = spike_monitors['Ae']

            if y % update_interval == 0 and y > 0:
                assignments = assign_labels(result_monitorAudio[:], input_numbersAudio[y - update_interval: y])

            single_example_timea = 0.15 * b.second
            b.run(single_example_timea)

            current_spike_countAudio = np.copy(spike_counters['Ae'].count[:]).reshape(
                (conv_featuresAudio, n_eAudio)) - previous_spike_countAudio
            previous_spike_countAudio = np.copy(spike_counters['Ae'].count[:]).reshape((conv_featuresAudio, n_eAudio))

            # neuron_groups['Te'].rate = spike_monitors['Ae'].source[:]

            # set weights to those of the most-fired neuron
            if not test_mode and weight_sharing == 'weight_sharing':
                if CounterFrames == int(Framecounter):
                    set_weights_most_fired(current_spike_countAudio, current_spike_countVideo)
                else:

                    set_weights_most_fired(current_spike_countAudio, current_spike_countVideo)

            current_spike_countConv = np.copy(spike_counters['Te'].count[:]).reshape(
                (conv_features, n_e)) - previous_spike_countConv
            previous_spike_countConv = np.copy(spike_counters['Te'].count[:]).reshape((conv_features, n_e))

            current_spike_countConv = current_spike_countAudio
            # update weights every 'weight_update_interval'
            if y % weight_update_interval == 0 and not test_mode and do_plot:
                update_2d_input_weights(input_weight_monitorAudio, fig_weightsAudio)
                if connectivity != 'none':
                    update_patch_weights(patch_weight_monitor, fig2_weights)

            if not test_mode and do_plot:
                update_neuron_votes(neuron_rectsAudio, fig_neuron_votesAudio, result_monitorAudio[:])

            # if the neurons in the network didn't spike more than four times
            t = np.sum(current_spike_countAudio)
            # current_spike_countAudio = previous_spike_countConv
            print (t)
            if np.sum(current_spike_countConv) < 5 and num_retries < 6:
                # increase the intensity of input
                input_intensity += 2
                num_retries += 1

                # set all network firing rates to zero
                for name in input_population_names:
                    input_groups[name + 'e'].rate = 0

                # let the network relax back to equilibrium
                b.run(resting_time)

            # otherwise, record results and continue simulation
            else:
                num_retries = 0
                # record the current number of spikes
                result_monitorVideo[j % update_interval, :] = current_spike_countVideo

                # decide whether to evaluate on test or training set
                if test_mode:
                    input_numbersVideo[j] = dataVideo['y'][j % data_sizeVideo]
                else:
                    input_numbersVideo[j] = dataVideo['y'][j % data_sizeVideo]

                # if CounterFrames == int(Framecounter):

                # reset frame counter
                #    CounterFrames = 0

                result_monitorAudio[y % update_interval, :] = current_spike_countAudio

                result_monitor[y % update_interval, :] = current_spike_countAudio

                # decide whether to evaluate on test or training set
                if test_mode:
                    input_numbersAudio[y] = dataAudio['y'][y % data_sizeAudio]
                else:
                    input_numbersAudio[y] = dataAudio['y'][y % data_sizeAudio]

                # get the output classifications of the network
                output_numbers['all'][y, :], output_numbers['most_spiked'][y, :], output_numbers['top_percent'][y, :] = \
                    predict_label(assignments, input_numbersAudio[y - update_interval - (y % update_interval): y - \
                                                                                                               (
                                                                                                                       y % update_interval)],
                                  result_monitorAudio[y % update_interval, :])

                # print progress
                if y % print_progress_interval == 0 and y > 0:
                    print 'runs done:', y, 'of', int(
                        num_examplesAudio), '(time taken for past', print_progress_interval, 'runs:', str(
                        timeit.default_timer() - start_time) + ')'
                    start_time = timeit.default_timer()

                # plot performance if appropriate
                print(y)
                if y % update_interval == 0 and y > 0:
                    if not test_mode and do_plot:
                        # updating the performance plot
                        perf_plot, performances = update_performance_plot(performance_monitor, performances, y,
                                                                          fig_performance)
                    else:
                        performances = get_current_performance(performances, y)

                    # pickling performance recording and iteration number
                    # p.dump((j, performances), open(os.path.join(performance_dir, ending + '.p'), 'wb'))

                    for performance in performances:
                        print '\nClassification performance (' + performance + ')', performances[performance][
                                                                                    1:int(y / float(
                                                                                        update_interval)) + 1], \
                            '\nAverage performance:', sum(
                            performances[performance][1:int(y / float(update_interval)) + 1]) / \
                                                      float(len(performances[performance][
                                                                1:int(y / float(update_interval)) + 1])), '\n'

                        # set input firing rates back to zero
                for name in input_population_names:
                    input_groups[name + 'e'].rate = 0

                # run the network for 'resting_time' to relax back to rest potentials
                b.run(resting_time)
                # bookkeeping
                input_intensity = start_input_intensity + 20
                input_intensityAudio = start_input_intensity + 20
                input_intensityVideo = start_input_intensity + 20

            CounterFrames = 0
            y += 1
            # set weights to those of the most-fired neuron
            if not test_mode and weight_sharing == 'weight_sharing':
                set_weights_most_fired(current_spike_countAudio, current_spike_countVideo)

        j += 1
        CounterFrames += 1

    # ensure weights don't grow without bound
    normalize_weights()

    print '\n'


def save_results():
    '''
    Save results of simulation (train or test)
    '''
    print '...Saving results'

    if not test_mode:
        save_connections(connections, input_connections, ending, 'None')
        save_theta(population_names, neuron_groups, ending, 'None')
    else:
        np.save('results_' + str(num_examples) + '_' + ending, result_monitorAudio)
        np.save('input_numbers_' + str(num_examples) + '_' + ending, input_numbersAudio)

    print '\n'


def evaluate_results():
    '''
    Evalute the network using the various voting schemes in test mode
    '''
    global update_interval

    start_time_training = start_time_testing = 0
    end_time_training = end_time_testing = num_examples

    update_interval = end_time_training

    training_result_monitor = testing_result_monitor = result_monitorAudio
    training_input_numbers = testing_input_numbers = input_numbersAudio

    print '...Getting assignments'

    assignments = assign_labels(training_result_monitor, training_input_numbers)

    voting_mechanisms = ['all', 'most-spiked (per patch)', 'most-spiked (overall)', ]

    test_results = {}
    for mechanism in voting_mechanisms:
        test_results[mechanism] = np.zeros((6, num_examples))

    print '\n...Calculating accuracy per voting mechanism'

    # for idx in xrange(end_time_testing - end_time_training):
    for idx in xrange(num_examples):
        for (mechanism, label_ranking) in zip(voting_mechanisms, predict_label(assignments, training_input_numbers,
                                                                               testing_result_monitor[idx, :])):
            test_results[mechanism][:, idx] = label_ranking

    differences = {mechanism: test_results[mechanism][0, :] - testing_input_numbers for mechanism in voting_mechanisms}
    correct = {mechanism: len(np.where(differences[mechanism] == 0)[0]) for mechanism in voting_mechanisms}
    incorrect = {mechanism: len(np.where(differences[mechanism] != 0)[0]) for mechanism in voting_mechanisms}
    accuracies = {mechanism: correct[mechanism] / float(end_time_testing - start_time_testing) * 100 for mechanism in
                  voting_mechanisms}

    for mechanism in voting_mechanisms:
        print '\n-', mechanism, 'accuracy:', accuracies[mechanism]

    for mechanism in voting_mechanisms:
        test = test_results[mechanism][0, :]

        test_results1 = test  # test_results[mechanism]
        # print(test_results)
        test_results1.tofile('testresultsent' + str(mechanism) + '.csv', sep=',')
        print(']]]]]]]]]]]]]]]]]]')
        # to_csv('testresults'+str(mechanism)+'.csv',)

    test_labels = np.array(testing_input_numbers)  # pd.DataFrame(testing_input_numbers)
    test_labels.tofile('testlabelsent.csv', sep=',')  # (to_csv('testlabels.csv',)
    results = pd.DataFrame([accuracies.values()], index=[str(num_examples) + '_' + ending], columns=accuracies.keys())
    if not 'accuracy_resultsent.csv' in os.listdir(results_path):
        results.to_csv(results_path + '.csv', )
    else:
        all_results = pd.read_csv(results_path + '.csv')
        all_results.append(results)
        all_results.to_csv(results_path + '.csv')

    print '\n'


if __name__ == '__main__':

    # results = np.load ('results_35_none_15_15_50_36_weight_dependence_postpre_weight_sharing_4_0.0.npy')

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', default='test', help='Network operating mode: "train" mode learns the synaptic weights of the network, and \
															"test" mode holds the weights fixed and evaluates classification accuracy on the test dataset.')
    parser.add_argument('--connectivity', default='none',
                        help='Between-patch connectivity: choose from "none", "pairs", "linear", and "full".')
    parser.add_argument('--weight_dependence', default='weight_dependence',
                        help='Modifies the STDP rule to either use or not use the weight dependence mechanism.')
    parser.add_argument('--post_pre', default='postpre',
                        help='Modifies the STDP rule to incorporate both post- and pre-synaptic weight updates, rather than just post-synaptic updates.')
    parser.add_argument('--conv_size', type=int, default=10,
                        help='Side length of the square convolution window used by the input -> excitatory layer of the network.')
    parser.add_argument('--conv_stride', type=int, default=10,
                        help='Horizontal, vertical stride of the convolution window used by the input -> excitatory layer of the network.')
    parser.add_argument('--conv_features', type=int, default=40,
                        help='Number of excitatory convolutional features / filters / patches used in the network.')
    parser.add_argument('--weight_sharing', default='weight_sharing',
                        help='Whether to use within-patch weight sharing (each neuron in an excitatory patch shares a single set of weights).')
    parser.add_argument('--lattice_structure', default='4',
                        help='The lattice neighborhood to which connected patches project their connections: one of "none", "4", "8", or "all".')
    parser.add_argument('--random_lattice_prob', type=float, default=0.0, help='Probability with which a neuron from an excitatory patch connects to a neuron in a neighboring excitatory patch \
																													with which it is not already connected to via the between-patch wiring scheme.')
    parser.add_argument('--random_inhibition_prob', type=float, default=0.0, help='Probability with which a neuron from the inhibitory layer connects to any given excitatory neuron with which \
																																it is not already connected to via the inhibitory wiring scheme.')
    parser.add_argument('--top_percent', type=int, default=10,
                        help='The percentage of neurons which are allowed to cast "votes" in the "top_percent" labeling scheme.')
    parser.add_argument('--do_plot', type=str, default='True', help='Whether or not to display plots during network training / testing. Defaults to False, as this makes the network operation \
																																					speedier, and possible to run on HPC resources.')
    parser.add_argument('--sort_euclidean', type=str, default='False', help='When plotting reshaped input -> excitatory weights, whether to plot each row (corresponding to locations in the input) \
																																					sorted by Euclidean distance from the 0 matrix.')
    parser.add_argument('--num_examples', type=int, default=10000,
                        help='The number of examples for which to train or test the network on.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='The random seed (any integer) from which to generate random numbers.')
    parser.add_argument('--reduced_dataset', type=str, default='False',
                        help='Whether or not to use 9-digit reduced-size dataset (900 images).')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes to use in reduced dataset.')
    parser.add_argument('--examples_per_class', type=int, default=20,
                        help='Number of examples per class to use in reduced dataset.')
    # parse arguments and place them in local scope

    test_mode = True

    args = parser.parse_args()
    args = vars(args)
    locals().update(args)

    print '\nOptional argument values:'
    for key, value in args.items():
        print '-', key, ':', value

    print '\n'

    if do_plot == 'True':
        do_plot = True
    elif do_plot == 'False':
        do_plot = False
    else:
        raise Exception('Expecting True or False-valued command line argument "do_plot".')

    if sort_euclidean == 'True':
        sort_euclidean = True
    elif sort_euclidean == 'False':
        sort_euclidean = False
    else:
        raise Exception('Expecting True or False-valued command line argument "sort_euclidean".')

    if reduced_dataset == 'True':
        reduced_dataset = True
    elif reduced_dataset == 'False':
        reduced_dataset = False
    else:
        raise Exception('Expecting True or False-valued command line argument "reduced_dataset".')

    '''infile = open('TestingAudioRAVDES100IJCNN.pickle',
                  'rb')  # 'TrainingAudioEnterface29July1.pickle', 'rb') #'TestingAudioRAVDESbrown.pickle', 'rb') #TestingAudioRAVDES100IJCNN.pickle', 'rb')  # TestingAudioEnterface06DecGPU.pickle' , 'rb')
    testingAudio = p.load(infile)
    infile.close()

    infile = open('TestingVideoRAVDESLaplace100IJCNN.pickle',
                  'rb')  # 'TrainingVideoEnterface29July1.pickle', 'rb') #'TestingVideoRAVDES0.2.pickle', 'rb' ) #TestingVideoRAVDESLaplace100IJCNN.pickle', 'rb')
    testingVideo = pickle.load(infile)
    infile.close()

    #FramecountersTestRAVDES100.pickle'
    infile = open('FramecountersTestRAVDESNOISE.pickle',
                  'rb')  # 'FramecountersTrainEnterface29July.pickle', 'rb') #'FramecountersTestRAVDES100.pickle', 'rb')
    testingFrameCounter = p.load(infile)
    infile.close()'''

    infile = open('FramecountersTestRAVDES100.pickle',  # FramecountersTestEnterNOISE.pickle',
                  'rb')  # 'FramecountersTrainEnterface29July.pickle', 'rb') #'FramecountersTestRAVDES100.pickle', 'rb')
    testingFrameCounter = p.load(infile)
    infile.close()

    # TestingAudioEnterfaceno
    '''infile = open('TestingAudioRAVDES100IJCNN.pickle', #TestingAudioEnterfacepink.pickle',
                  'rb')  # 'TrainingAudioEnterface29July1.pickle', 'rb') #'TestingAudioRAVDESbrown.pickle', 'rb') #TestingAudioRAVDES100IJCNN.pickle', 'rb')  # TestingAudioEnterface06DecGPU.pickle' , 'rb')
    testingAudio = p.load(infile)
    infile.close()

    #TestingVideoEnterface0
    infile = open('TestingVideoRAVDESLaplace100IJCNN.pickle', #TestingVideoEnterface0.pickle',
                  'rb')  # 'TrainingVideoEnterface29July1.pickle', 'rb') #'TestingVideoRAVDES0.2.pickle', 'rb' ) #TestingVideoRAVDESLaplace100IJCNN.pickle', 'rb')
    testingVideo = pickle.load(infile)
    infile.close()'''

    '''infile = open('TestingVideoEnterface0.8.pickle',
                  'rb')  # 'TrainingVideoEnterface29July1.pickle', 'rb') #'TestingVideoRAVDES0.2.pickle', 'rb' ) #TestingVideoRAVDESLaplace100IJCNN.pickle', 'rb')
    testingVideo = p.load(infile)
    infile.close()

    infile = open('FramecountersTestEnterNOISE.pickle',
                  'rb')  # 'FramecountersTrainEnterface29July.pickle', 'rb') #'FramecountersTestRAVDES100.pickle', 'rb')
    testingFrameCounter = p.load(infile)
    infile.close()

    # TestingAudioEnterfaceno
    infile = open('TestingAudioEnterfaceno.pickle',
                  'rb')  # 'TrainingAudioEnterface29July1.pickle', 'rb') #'TestingAudioRAVDESbrown.pickle', 'rb') #TestingAudioRAVDES100IJCNN.pickle', 'rb')  # TestingAudioEnterface06DecGPU.pickle' , 'rb')
    testingAudio = p.load(infile)
    infile.close()'''

    infile = open('FramecountersTestEnterNOISE.pickle',  # FramecountersTestEnterNOISE.pickle',
                  'rb')  # 'FramecountersTrainEnterface29July.pickle', 'rb') #'FramecountersTestRAVDES100.pickle', 'rb')
    testingFrameCounter = p.load(infile)
    infile.close()

    infile = open('TestingAudioEnterfaceno.pickle',  # TestingAudioEnterfaceno.pickle',
                  'rb')  # 'TrainingAudioEnterface29July1.pickle', 'rb') #'TestingAudioRAVDESbrown.pickle', 'rb') #TestingAudioRAVDES100IJCNN.pickle', 'rb')  # TestingAudioEnterface06DecGPU.pickle' , 'rb')
    testingAudio = p.load(infile)
    infile.close()

    infile = open('TestingVideoEnterface0.pickle',  # TestingVideoEnterface0.pickle',
                  'rb')  # 'TrainingVideoEnterface29July1.pickle', 'rb') #'TestingVideoRAVDES0.2.pickle', 'rb' ) #TestingVideoRAVDESLaplace100IJCNN.pickle', 'rb')
    testingVideo = pickle.load(infile)
    infile.close()

    '''infile = open('FramecountersTrainEnterface29July.pickle', 'rb') #FramecountersTrainEnterfac06DecGPUSMALL.pickle' , 'rb') #FramecountersTrainEnterfaceGPU.pickle', 'rb')
    trainingFrameCounter = p.load(infile)
    print(trainingFrameCounter.shape)
    infile.close()

    infile = open('TrainingVideoEnterface29July1.pickle', 'rb')
    #infile = open('TrainingAudioRAVDES388.pickle', 'rb')
    trainingAudio = p.load(infile)
    print(trainingAudio['x'].shape)
    infile.close()'''

    '''infile = open('TrainingVideoEnterface29July1.pickle', 'rb') #TrainingVideoEnterface06DeGPUSMALL.pickleTrainingVideoEnterface06DeGPU.pickle', 'rb')
    #infile = open('TrainingVideoRAVDES.pickle', 'rb')
    trainingVideo = pickle.load(infile)
    infile.close()'''

    '''X = np.memmap('trainingVideo.npy', dtype='int', mode='w+', shape=(1, 1))
    #Y = np.memmap('trainingAudio.npy', dtype='int', mode='w+', shape=(1, 1))
    X = trainingVideo['x']
    print(trainingVideo['x'].shape)
    infile.close()'''

    # training = np.array(training)

    # testing = pickle.load(open('TestingNewLabels6laplaceJAFFE.pickle', 'rb'))

    # testing = training
    # testing = np.array(testing)
    # testing = np.load('predictionall.npy')

    '''numTrainVideo = len(trainingVideo['x'])

    numTrainVideo60 = (numTrain * 60) / 100

    print(numTrainVideo60)'''

    # numTest = len(testing['x'])

    # num_examplesVideo = numTrain60

    numTestVideo = len(testingVideo['x'])

    # numTrainAudio = len(trainingAudio['x'])
    numTestAudio = len(testingAudio['x'])

    # print(numTestAudio)

    # print(numTestVideo)

    num_examplesAudio = numTestAudio  # numTest #numTest  # 2400 #numTest #2400 #900 #1200 #numTrain # 2400 #numTrain
    num_examplesVideo = numTestVideo
    print(num_examplesVideo)
    num_examples = num_examplesAudio
    if reduced_dataset:
        data_size = num_classes * examples_per_class

    elif test_mode:
        data_sizeAudio = numTestAudio
        data_sizeVideo = numTestVideo  # numTest #numTest  # numTest
    else:
        data_sizeAudio = numTestAudio
        data_sizeVideo = numTestVideo  # numTest #numTest  # numTest
        # data_sizeAudio = numTrainAudio
        # data_sizeVideo = numTrainVideo

    # set brian global preferences
    b.set_global_preferences(defaultclock=b.Clock(dt=0.5 * b.ms), useweave=True,
                             gcc_options=['-ffast-math -march=native'], usecodegen=True,
                             usecodegenweave=True, usecodegenstateupdate=True, usecodegenthreshold=False,
                             usenewpropagate=True, usecstdp=True, openmp=False,
                             magic_useframes=False, useweave_linear_diffeq=True)

    # for reproducibility's sake
    np.random.seed(random_seed)

    # test or train mode
    # test_mode = mode == 'test'

    start = timeit.default_timer()
    '''data = get_labeled_data(os.path.join(MNIST_data_path, 'testing' if test_mode else 'training'),
                                                not test_mode, reduced_dataset, num_classes, examples_per_class)'''

    # dataVideo = testingVideo

    # data = X  # training['x']
    dataVideo = testingVideo  # data[0:numTrainVideo60]
    # labels = trainingVideo['y']  # training # training
    # labelsVideo = labels[0:numTrainVideo60]
    # training # training
    dataAudio = testingAudio
    print 'Time needed to load data:', timeit.default_timer() - start

    # set parameters for simulation based on train / test mode
    if test_mode:
        do_plot_performance = False
        record_spikes = True
        ee_STDP_on = False
    else:
        do_plot_performance = True
        record_spikes = True
        ee_STDP_on = True

    # number of inputs to the network
    n_inputVideo = 10000  #
    n_input_sqrtVideo = int(math.sqrt(n_inputVideo))

    n_inputAudio = 40 * 388  # 276  #
    n_input_sqrtAudio = int(math.sqrt(n_inputAudio))

    conv_sizeVideo = 10  # conv_size
    conv_sizeAudio = 10  # conv_size

    conv_strideVideo = 10  # conv_stride
    conv_strideAudio = 10  # conv_stride

    # number of neurons parameters
    if conv_size == 100 and conv_stride == 0:
        n_e = 1
    else:
        n_eVideo = (((n_input_sqrtVideo - conv_sizeVideo) / conv_strideVideo) + 1) ** 2
        n_eAudio = (((n_input_sqrtVideo - conv_sizeAudio) / conv_strideAudio) + 1) ** 2

    conv_featuresVideo = 40
    n_e_totalVideo = n_eVideo * conv_featuresVideo
    n_e_sqrtVideo = int(math.sqrt(n_eVideo))
    n_iVideo = n_eVideo
    conv_features_sqrtVideo = int(math.ceil(math.sqrt(conv_featuresVideo)))

    conv_featuresAudio = 40
    n_e_totalAudio = n_eAudio * conv_featuresAudio
    n_e_sqrtAudio = int(math.sqrt(n_eAudio))
    n_iAudio = n_eAudio
    conv_features_sqrtAudio = int(math.ceil(math.sqrt(conv_featuresAudio)))

    # This is for the convergence layer

    n_e = n_eAudio  # n_eVideo + n_eAudio

    conv_features = conv_featuresAudio  # conv_featuresVideo + conv_featuresAudio

    n_e_total = n_e * conv_features
    # time (in seconds) per data example presentation and rest period in between, used to calculate total runtime
    single_example_time = 0.35 * b.second  # 35
    resting_time = 0.15 * b.second
    runtime = num_examples * (single_example_time + resting_time)

    # set the update interval
    if test_mode:
        update_interval = num_examplesAudio
    else:
        update_interval = 100  # 100

    # weight updates and progress printing intervals
    weight_update_interval = 100  # 50
    print_progress_interval = 100  # 50

    # rest potential parameters, reset potential parameters, threshold potential parameters, and refractory periods
    v_rest_e, v_rest_i = -65. * b.mV, -60. * b.mV
    v_reset_e, v_reset_i = -65. * b.mV, -45. * b.mV
    v_thresh_e, v_thresh_i = -52. * b.mV, -40. * b.mV
    v_thresh_eConv, v_thresh_iConv = -25. * b.mV, -10. * b.mV
    refrac_e, refrac_i = 5. * b.ms, 2. * b.ms

    # dictionaries for weights and delays
    weight, delay = {}, {}

    # populations, connections, saved connections, etc.
    input_population_names = ['X', 'Y']
    #define three population audio A , video V and Multisensory T
    population_names = ['A', 'V', 'T']
    input_connection_names = ['XV', 'YA']
    save_conns = ['XeVe', 'VeVe', 'YeAe', 'AeAe', 'AeTe', 'VeTe']

    # weird and bad names for variables, I think
    input_conn_names = ['ee_input']
    recurrent_conn_names = ['ei', 'ie', 'ee']

    # setting weight, delay, and intensity parameters
    if conv_size == 100 and conv_stride == 0:
        weight['ee_input'] = (conv_size ** 2) * 0.15
    else:
        weight['ee_input'] = (conv_size ** 2) * 0.1625
    delay['ee_input'] = (0 * b.ms, 10 * b.ms)
    delay['ei_input'] = (0 * b.ms, 5 * b.ms)
    delay['ee'] = (0 * b.ms, 5 * b.ms)
    input_intensity = start_input_intensity = 30.0

    input_intensityVideo = start_input_intensity = 30.0

    input_intensityAudio = start_input_intensity = 30.0

    # time constants, learning rates, max weights, weight dependence, etc.
    tc_pre_ee, tc_post_ee = 20 * b.ms, 20 * b.ms
    nu_ee_pre, nu_ee_post = 0.0001, 0.01
    wmax_ee = 1.0
    exp_ee_post = exp_ee_pre = 0.2
    w_mu_pre, w_mu_post = 0.2, 0.2

    # setting up differential equations (depending on train / test mode)
    if test_mode:
        scr_e = 'v = v_reset_e; timer = 0*ms'
    else:
        tc_theta = 1e7 * b.ms
        theta_plus_e = 0.05 * b.mV
        scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

    offset = 20.0 * b.mV
    v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'

    # equation for convergence
    eqs_post = '''
    dv/dt=(n-v)/tau_cd : 1
    dn/dt=-n/tau_n+sigma*(2/tau_n)**.5*xi : 1
    '''

    # equations for neurons
    neuron_eqs_e = '''
			dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / (100 * ms)  : volt
			I_synE = ge * nS *         -v                           : amp
			I_synI = gi * nS * (-100.*mV-v)                          : amp
			dge/dt = -ge/(1.0*ms)                                   : 1
			dgi/dt = -gi/(2.0*ms)                                  : 1
			'''
    if test_mode:
        neuron_eqs_e += '\n  theta      :volt'
    else:
        neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'

    neuron_eqs_e += '\n  dtimer/dt = 100.0 : ms'

    neuron_eqs_i = '''
			dv/dt = ((v_rest_i - v) + (I_synE + I_synI) / nS) / (10*ms)  : volt
			I_synE = ge * nS *         -v                           : amp
			I_synI = gi * nS * (-85.*mV-v)                          : amp
			dge/dt = -ge/(1.0*ms)                                   : 1
			dgi/dt = -gi/(2.0*ms)                                  : 1
			'''

    # STDP rule
    stdp_input = weight_dependence + '_' + post_pre
    if weight_dependence == 'weight_dependence':
        use_weight_dependence = True
    else:
        use_weight_dependence = False
    if post_pre == 'postpre':
        use_post_pre = True
    else:
        use_post_pre = False

    # STDP synaptic traces
    eqs_stdp_ee = '''
				dpre/dt = -pre / tc_pre_ee : 1.0
				dpost/dt = -post / tc_post_ee : 1.0
				'''

    # setting STDP update rule
    if use_weight_dependence:
        if post_pre:
            eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post * w ** exp_ee_pre'
            eqs_stdp_post_ee = 'w += nu_ee_post * pre * (wmax_ee - w) ** exp_ee_post; post = 1.'

        else:
            eqs_stdp_pre_ee = 'pre = 1.'
            eqs_stdp_post_ee = 'w += nu_ee_post * pre * (wmax_ee - w) ** exp_ee_post; post = 1.'

    else:
        if use_post_pre:
            eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post'
            eqs_stdp_post_ee = 'w += nu_ee_post * pre; post = 1.'

        else:
            eqs_stdp_pre_ee = 'pre = 1.'
            eqs_stdp_post_ee = 'w += nu_ee_post * pre; post = 1.'

    print '\n'

    # set ending of filename saves
    ending = connectivity + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(
        n_e) + '_' + \
             weight_dependence + '_' + post_pre + '_' + weight_sharing + '_' + lattice_structure + '_' + str(
        random_lattice_prob)

    b.ion()
    fig_num = 1

    # creating dictionaries for various objects
    neuron_groups, input_groups, connections, input_connections, stdp_methods, \
    rate_monitors, spike_monitors, spike_counters, output_numbers = {}, {}, {}, {}, {}, {}, {}, {}, {}

    # creating convolution locations inside the input image
    convolution_locations = {}
    convolution_locationsVideo = {}
    convolution_locationsAudio = {}
    # convolution locations for each modalities
    for n in xrange(n_eVideo):
        convolution_locationsVideo[n] = [
            ((n % n_e_sqrtVideo) * conv_strideVideo + (n // n_e_sqrtVideo) * n_input_sqrtVideo * \
             conv_strideVideo) + (x * n_input_sqrtVideo) + y for y in xrange(conv_sizeVideo) for x in
            xrange(conv_sizeVideo)]

    # Audio
    for n in xrange(n_eAudio):
        convolution_locationsAudio[n] = [
            ((n % n_e_sqrtAudio) * conv_strideAudio + (n // n_e_sqrtAudio) * n_input_sqrtAudio * \
             conv_strideAudio) + (x * n_input_sqrtAudio) + y for y in xrange(conv_sizeAudio) for x in
            xrange(conv_sizeAudio)]

    # instantiating neuron "vote" monitor
    result_monitorVideo = np.zeros((update_interval, conv_featuresVideo, n_eVideo))

    result_monitorAudio = np.zeros((update_interval, conv_featuresAudio, n_eAudio))

    result_monitor = np.zeros((update_interval, conv_features, n_e))

    # build the spiking neural network
    build_network()

    # bookkeeping variables
    previous_spike_countVideo = np.zeros((conv_featuresVideo, n_eVideo))
    previous_spike_countAudio = np.zeros((conv_featuresAudio, n_eAudio))
    previous_spike_countConv = np.zeros((conv_features, n_e))

    assignments = np.zeros((conv_featuresVideo, n_eVideo))

    # TODO Check
    assignments = np.zeros((conv_featuresAudio, n_eAudio))

    input_numbersVideo = [0] * num_examplesVideo

    input_numbersAudio = [0] * num_examplesAudio

    input_numbers = input_numbersAudio

    ratesVideo = np.zeros((n_input_sqrtVideo, n_input_sqrtVideo))

    ratesAudio = np.zeros((40, 388))  # 276))


    output_numbers['most_spiked'] = np.zeros((num_examplesAudio, 6))

    # run the simulation of the network
    run_simulation()

    # save and plot results
    save_results()

    # evaluate results

    if test_mode:
        evaluate_results()
