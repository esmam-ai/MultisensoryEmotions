'''
Spiking Neural network implementation for FER tasks.
Based on code by  https://github.com/peter-u-diehl/stdp-mnist



'''

import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import brian_no_units
import networkx as nx
import cPickle as p
import pandas as pd
import cPickle as pickle
import numpy as np
import brian as b
import argparse
import timeit
import math
import os

from sklearn.cluster import KMeans
from struct import unpack
from brian import *

from util import *



def set_weights_most_fired(current_spike_count):
    '''
    For each convolutional patch, set the weights to those of the neuron which
    fired the most in the last iteration.
    '''

    for conn_name in input_connections:
        for feature in xrange(conv_features):
            # count up the spikes for the neurons in this convolution patch
            column_sums = np.sum(current_spike_count[feature: feature + 1, :], axis=0)

            # find the excitatory neuron which spiked the most
            most_spiked = np.argmax(column_sums)

            # create a "dense" version of the most spiked excitatory neuron's weight
            most_spiked_dense = input_connections[conn_name][:, feature * n_e + most_spiked].todense()

            # set all other neurons' (in the same convolution patch) weights the same as the most-spiked neuron in the patch
            for n in xrange(n_e):
                if n != most_spiked:
                    other_dense = input_connections[conn_name][:, feature * n_e + n].todense()
                    other_dense[convolution_locations[n]] = most_spiked_dense[convolution_locations[most_spiked]]
                    input_connections[conn_name][:, feature * n_e + n] = other_dense


def normalize_weights():
    '''
    Squash the input -> excitatory weights to sum to a prespecified number.
    '''
    for conn_name in input_connections:
        connection = input_connections[conn_name][:].todense()
        for feature in xrange(conv_features):
            feature_connection = connection[:, feature * n_e: (feature + 1) * n_e]
            column_sums = np.sum(np.asarray(feature_connection), axis=0)
            column_factors = weight['ee_input'] / column_sums

            for n in xrange(n_e):
                dense_weights = input_connections[conn_name][:, feature * n_e + n].todense()
                dense_weights[convolution_locations[n]] *= column_factors[n]
                input_connections[conn_name][:, feature * n_e + n] = dense_weights

    for conn_name in connections:
        if 'AeAe' in conn_name and lattice_structure != 'none' and lattice_structure != 'none':
            connection = connections[conn_name][:].todense()
            for feature in xrange(conv_features):
                feature_connection = connection[feature * n_e: (feature + 1) * n_e, :]
                column_sums = np.sum(feature_connection)
                column_factors = weight['ee_recurr'] / column_sums

                for idx in xrange(feature * n_e, (feature + 1) * n_e):
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


def update_input(rates, im, fig):
    '''
    Update the input image to use for input plotting.
    '''
    im.set_array(rates.reshape((100, 100)))
    fig.canvas.draw()
    return im


def get_2d_input_weights():
    '''
    Get the weights from the input to excitatory layer and reshape it to be two
    dimensional and square.
    '''
    # specify the desired shape of the reshaped input -> excitatory weights
    rearranged_weights = np.zeros((conv_features * conv_size, conv_size * n_e))

    # get the input -> excitatory synaptic weights
    connection = input_connections['XeAe'][:]

    if sort_euclidean:
        # for each excitatory neuron in this convolution feature
        euclid_dists = np.zeros((n_e, conv_features))
        temps = np.zeros((n_e, conv_features, n_input))
        for n in xrange(n_e):
            # for each convolution feature
            for feature in xrange(conv_features):
                temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)].todense()
                if feature == 0:
                    if n == 0:
                        euclid_dists[n, feature] = 0.0
                    else:
                        euclid_dists[n, feature] = np.linalg.norm(
                            temps[0, 0, convolution_locations[n]] - temp[convolution_locations[n]])
                else:
                    euclid_dists[n, feature] = np.linalg.norm(
                        temps[n, 0, convolution_locations[n]] - temp[convolution_locations[n]])

                temps[n, feature, :] = temp.ravel()

            for idx, feature in enumerate(np.argsort(euclid_dists[n])):
                temp = temps[n, feature]
                rearranged_weights[idx * conv_size: (idx + 1) * conv_size, n * conv_size: (n + 1) * conv_size] = \
                    temp[convolution_locations[n]].reshape((conv_size, conv_size))

    else:
        for n in xrange(n_e):
            for feature in xrange(conv_features):
                temp = connection[:, feature * n_e + (n // n_e_sqrt) * n_e_sqrt + (n % n_e_sqrt)].todense()
                #print('print')
                #print(temp)
                rearranged_weights[feature * conv_size: (feature + 1) * conv_size, n * conv_size: (n + 1) * conv_size] = \
                    temp[convolution_locations[n]].reshape((conv_size, conv_size))

    # return the rearranged weights to display to the user
    if n_e == 1:
        ceil_sqrt = int(math.ceil(math.sqrt(conv_features)))
        square_weights = np.zeros((100 * ceil_sqrt, 100 * ceil_sqrt))
        for n in xrange(conv_features):
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


def plot_2d_input_weights():
    '''
    Plot the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weights()



    if n_e != 1:
        fig = plt.figure(fig_num, figsize=(10, 10))  #fig = plt.figure(fig_num, figsize=(18, 9))
    else:
        fig = plt.figure(fig_num, figsize=(6, 6))

    im = plt.imshow(weights, interpolation='nearest', vmin=0, vmax=wmax_ee, cmap=cmap.get_cmap('hot_r')) #hot_r'))

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
        plt.xticks(xrange(conv_size, conv_size * (conv_features + 1), conv_size), xrange(1, conv_features + 1))
        plt.yticks(xrange(conv_size, conv_size * (n_e + 1), conv_size), xrange(1, n_e + 1))

        #(conv_features * conv_size, conv_size * n_e)
        plt.xlabel('Convolution feature')
        plt.ylabel('Location in input') # (from top left to bottom right')

    fig.canvas.draw()
    return im, fig


def update_2d_input_weights(im, fig):
    '''
    Update the plot of the weights from input to excitatory layer to view during training.
    '''
    weights = get_2d_input_weights()
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

    for i in xrange(6):
        num_assignments[i] = len(np.where(assignments == i)[0])
        if num_assignments[i] > 0:
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
        #print(assignments.shape)
        #print(most_spiked_array.shape)
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


def assign_labels(result_monitor, input_numbers):
    '''
    Based on the results from the previous 'update_interval', assign labels to the
    excitatory neurons.
    '''
    assignments = np.ones((conv_features, n_e))
    input_nums = np.asarray(input_numbers)
    maximum_rate = np.zeros(conv_features * n_e)

    for j in xrange(6):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            #print('result_monitor')
            #print(input_nums) #np.sum(result_monitor[input_nums == j], axis=0))
            rate = np.sum(result_monitor[input_nums == j], axis=0) / num_assignments
            #print('rate')
            #print(rate)
            for i in xrange(conv_features * n_e):
                if rate[i // n_e, i % n_e] > maximum_rate[i]:
                    maximum_rate[i] = rate[i // n_e, i % n_e]
                    assignments[i // n_e, i % n_e] = j

    return assignments


def save_results():
    '''
    Save results of simulation (train or test)
    '''
    print '...Saving results'

    if not test_mode:
        save_connections( connections, input_connections, ending, 'None')
        save_theta( population_names, neuron_groups, ending, 'None')
    else:
        np.save( 'results_' + str(num_examples) + '_' + ending, result_monitor)
        np.save( 'input_numbers_' + str(num_examples) + '_' + ending, input_numbers)

    print '\n'


def evaluate_results():
    '''
    Evalute the network using the various voting schemes in test mode
    '''
    global update_interval

    start_time_training = start_time_testing = 0
    end_time_training = end_time_testing = num_examples

    update_interval = end_time_training

    training_result_monitor = testing_result_monitor = result_monitor
    training_input_numbers = testing_input_numbers = input_numbers

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

        test_results1 = test # test_results[mechanism]
        #print(test_results)
        test_results1.tofile('testresultsRAVDESImageN5'+str(mechanism)+'.csv', sep=',')
        print(']]]]]]]]]]]]]]]]]]')
        #to_csv('testresults'+str(mechanism)+'.csv',)

    test_labels = np.array(testing_input_numbers) #pd.DataFrame(testing_input_numbers)
    test_labels.tofile('testlabelsRAVDESImageN5.csv',sep=',') #(to_csv('testlabels.csv',)
    results = pd.DataFrame([accuracies.values()], index=[str(num_examples) + '_' + ending], columns=accuracies.keys())
    if not 'accuracy_resultsRAVDESImageN5.csv' in os.listdir(results_path):
        results.to_csv(results_path + '.csv', )
    else:
        all_results = pd.read_csv(results_path + '.csv')
        all_results.append(results)
        all_results.to_csv(results_path + '.csv')

    print '\n'


def initialise_network():
    global fig_num

    neuron_groups['e'] = b.NeuronGroup(n_e_total, neuron_eqs_e, threshold=v_thresh_e, refractory=refrac_e, reset=scr_e,
                                       compile=True, freeze=True)
    neuron_groups['i'] = b.NeuronGroup(n_e_total, neuron_eqs_i, threshold=v_thresh_i, refractory=refrac_i,
                                       reset=v_reset_i, compile=True, freeze=True)

    for name in population_names:
        print '...Creating neuron group:', name

        # get a subgroup of size 'n_e' from all exc
        neuron_groups[name + 'e'] = neuron_groups['e'].subgroup(conv_features * n_e)
        # get a subgroup of size 'n_i' from the inhibitory layer
        neuron_groups[name + 'i'] = neuron_groups['i'].subgroup(conv_features * n_e)

        # start the membrane potentials of these groups 40mV below their resting potentials
        neuron_groups[name + 'e'].v = v_rest_e - 40. * b.mV
        neuron_groups[name + 'i'].v = v_rest_i - 40. * b.mV

    print '...Creating recurrent connections'

    for name in population_names:
        # if we're in test mode / using some stored weights
        if test_mode:
            # load up adaptive threshold parameters
            # neuron_groups['e'].theta = np.load(os.path.join(weights_dir, 'theta_A' + '_' + ending +'.npy'))
            neuron_groups['e'].theta = np.load('theta_A' + '_' + ending + '.npy')
        else:
            # otherwise, set the adaptive additive threshold parameter at 20mV
            neuron_groups['e'].theta = np.ones((n_e_total)) * 20.0 * b.mV

        for conn_type in recurrent_conn_names:
            if conn_type == 'ei':
                # create connection name (composed of population and connection types)
                conn_name = name + conn_type[0] + name + conn_type[1]
                # create a connection from the first group in conn_name with the second group
                connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]],
                                                      structure='sparse', state='g' + conn_type[0])
                # instantiate the created connection
                for feature in xrange(conv_features):
                    for n in xrange(n_e):
                        connections[conn_name][feature * n_e + n, feature * n_e + n] = 10.4

            elif conn_type == 'ie':
                # create connection name (composed of population and connection types)
                conn_name = name + conn_type[0] + name + conn_type[1]
                # create a connection from the first group in conn_name with the second group
                connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]],
                                                      structure='sparse', state='g' + conn_type[0])
                # instantiate the created connection
                for feature in xrange(conv_features):
                    for other_feature in xrange(conv_features):
                        if feature != other_feature:
                            for n in xrange(n_e):
                                connections[conn_name][feature * n_e + n, other_feature * n_e + n] = 17.4


            elif conn_type == 'ee':
                # create connection name (composed of population and connection types)
                conn_name = name + conn_type[0] + name + conn_type[1]
                # get weights from file if we are in test mode
                if test_mode:
                    weight_matrix = np.load( conn_name + '_' + ending + '.npy')
                    connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]],
                                                  weight_matrix)
                # create a connection from the first group in conn_name with the second group
                #connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]],
                #                                      structure='sparse', state='g' + conn_type[0])

                else:
                    connections[conn_name] = b.Connection(neuron_groups[conn_name[0:2]], neuron_groups[conn_name[2:4]],
                                                         structure='sparse', state='g' + conn_type[0])

                    #instantiate the created connection


        # if STDP from excitatory -> excitatory is on and this connection is excitatory -> excitatory
        if ee_STDP_on and 'ee' in recurrent_conn_names:
            stdp_methods[name + 'e' + name + 'e'] = b.STDP(connections[name + 'e' + name + 'e'], eqs=eqs_stdp_ee,
                                                           pre=eqs_stdp_pre_ee, post=eqs_stdp_post_ee, wmin=0.,
                                                           wmax=wmax_ee)

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
        b.figure(fig_num, figsize=(8, 6))

        fig_num += 1

        b.ion()
        b.subplot(211)
        b.raster_plot(spike_monitors['Ae'], refresh=1000 * b.ms, showlast=1000 * b.ms,
                      title='Excitatory spikes per neuron')
        b.subplot(212)
        b.raster_plot(spike_monitors['Ai'], refresh=1000 * b.ms, showlast=1000 * b.ms,
                      title='Inhibitory spikes per neuron')
        b.tight_layout()

    # creating lattice locations for each patch

    lattice_locations = {}


    # setting up parameters for weight normalization between patches
    num_lattice_connections = sum([len(value) for value in lattice_locations.values()])
    weight['ee_recurr'] = (num_lattice_connections / conv_features) * 0.15

    # creating Poission spike train from input image (784 vector, 100x100 image)
    for name in input_population_names:
        input_groups[name + 'e'] = b.PoissonGroup(n_input, 0)
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
                weight_matrix = np.load(conn_name + '_' + ending + '.npy')
            # weight_matrix[weight_matrix < 0.20] = 0

            # create connections from the windows of the input group to the neuron population
            input_connections[conn_name] = b.Connection(input_groups['Xe'], neuron_groups[name[1] + conn_type[1]],
                                                        structure='sparse', state='g' + conn_type[0], delay=True,
                                                        max_delay=delay[conn_type][1])

            if test_mode:
                for feature in xrange(conv_features):
                    for n in xrange(n_e):
                        for idx in xrange(conv_size ** 2):
                            input_connections[conn_name][convolution_locations[n][idx], feature * n_e + n] = \
                            weight_matrix[convolution_locations[n][idx], feature * n_e + n]
            else:
                for feature in xrange(conv_features):
                    for n in xrange(n_e):
                        for idx in xrange(conv_size ** 2):
                            input_connections[conn_name][convolution_locations[n][idx], feature * n_e + n] = (
                                                                                                             b.random() + 0.01) * 0.3

            if test_mode:
                # normalize_weights()
                if do_plot:
                    plot_2d_input_weights()
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


def run_snn():
    '''
    Logic for running the simulation itself.
    '''
    global fig_num, input_intensity, previous_spike_count, rates, assignments, clusters, cluster_assignments, \
        kmeans, kmeans_assignments, simple_clusters, simple_cluster_assignments, index_matrix

    # plot input weights
    #print(test_mode)
    if not test_mode and do_plot:
        input_weight_monitor, fig_weights = plot_2d_input_weights()
        fig_num += 1
        if connectivity != 'none':
            patch_weight_monitor, fig2_weights = plot_patch_weights()
            fig_num += 1
        neuron_rects, fig_neuron_votes = plot_neuron_votes(assignments, result_monitor[:])
        fig_num += 1

    # plot input intensities
    if do_plot:
        input_image_monitor, input_image = plot_input(rates)
        fig_num += 1

    # set up performance recording and plotting
    num_evaluations = int(num_examples / update_interval)+1
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
    #spikemon = []

    while j < num_examples:
        # fetched rates depend on training / test phase, and whether we use the
        # testing dataset for the test phase
        if test_mode:
            rates = (data['x'][j % data_size, :, :] / 8.0) * input_intensity
        else:
            # ensure weights don't grow without bound
            normalize_weights()
            # get the firing rates of the next input example
            rates = (data['x'][j % data_size, :, :] / 8.0) * input_intensity

        # plot the input at this step
        if do_plot:
            input_image_monitor = update_input(rates, input_image_monitor, input_image)

        # sets the input firing rates
        input_groups['Xe'].rate = rates.reshape(n_input)

        # run the network for a single example time
        b.run(single_example_time)

        # get new neuron label assignments every 'update_interval'
        if j % update_interval == 0 and j > 0:
            #print('resultmon')
            #print(result_monitor)
            assignments = assign_labels(result_monitor[:], input_numbers[j - update_interval: j])

        sp = spike_monitors['Ae']  # [:].reshape(conv_features, n_e)#spike_counters['Ae']#spike_monitors['Ae']
        spikemon.append(sp.it)  # .count)

        # get count of spikes over the past iteration
        current_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape(
            (conv_features, n_e)) - previous_spike_count
        previous_spike_count = np.copy(spike_counters['Ae'].count[:]).reshape((conv_features, n_e))

        # set weights to those of the most-fired neuron
        if not test_mode and weight_sharing == 'weight_sharing':
            set_weights_most_fired(current_spike_count)

        # update weights every 'weight_update_interval'
        if j % weight_update_interval == 0 and not test_mode and do_plot:
            update_2d_input_weights(input_weight_monitor, fig_weights)
            if connectivity != 'none':
                update_patch_weights(patch_weight_monitor, fig2_weights)

        if not test_mode and do_plot:
            update_neuron_votes(neuron_rects, fig_neuron_votes, result_monitor[:])

        # if the neurons in the network didn't spike more than four times
        t = np.sum(current_spike_count)
        print (t)
        if np.sum(current_spike_count) < 5 and num_retries < 6:
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
            result_monitor[j % update_interval, :] = current_spike_count

            # decide whether to evaluate on test or training set
            if test_mode:
                input_numbers[j] = data['y'][j % data_size] #[0]
            else:
                input_numbers[j] = data['y'][j % data_size] #[0]

            # get the output classifications of the network
            output_numbers['all'][j, :], output_numbers['most_spiked'][j, :], output_numbers['top_percent'][j, :] = \
                predict_label(assignments, input_numbers[j - update_interval - (j % update_interval): j - \
                                                                                                      (
                                                                                                      j % update_interval)],
                              result_monitor[j % update_interval, :])

            # print progress
            if j % print_progress_interval == 0 and j > 0:
                print 'runs done:', j, 'of', int(
                    num_examples), '(time taken for past', print_progress_interval, 'runs:', str(
                    timeit.default_timer() - start_time) + ')'
                start_time = timeit.default_timer()

            # plot performance if appropriate
            if j % update_interval == 0 and j > 0:
                if not test_mode and do_plot:
                    # updating the performance plot
                    perf_plot, performances = update_performance_plot(performance_monitor, performances, j,
                                                                      fig_performance)
                else:
                    performances = get_current_performance(performances, j)

                # pickling performance recording and iteration number
                p.dump((j, performances), open(os.path.join(performance_dir, ending + '.p'), 'wb'))

                for performance in performances:
                    print '\nClassification performance (' + performance + ')', performances[performance][
                                                                                1:int(j / float(update_interval)) + 1], \
                        '\nAverage performance:', sum(
                        performances[performance][1:int(j / float(update_interval)) + 1]) / \
                                                  float(len(performances[performance][
                                                            1:int(j / float(update_interval)) + 1])), '\n'

            # set input firing rates back to zero
            for name in input_population_names:
                input_groups[name + 'e'].rate = 0

            # run the network for 'resting_time' to relax back to rest potentials
            b.run(resting_time)
            # bookkeeping
            input_intensity = start_input_intensity
            j += 1

    # set weights to those of the most-fired neuron
    if not test_mode and weight_sharing == 'weight_sharing':
        set_weights_most_fired(current_spike_count)

    # ensure weights don't grow without bound
    normalize_weights()

    print '\n'



if __name__ == '__main__':


    #results = np.load ('results_35_none_15_15_50_36_weight_dependence_postpre_weight_sharing_4_0.0.npy')

    parser = argparse.ArgumentParser()

    conv_size = 40
    conv_stride = 40
    conv_features = 20
    weight_sharing = 'weight_sharing'
    do_plot = 'False'
    sort_euclidean = True
    num_classed = 6
    example_per_class = 20
    reduced_dataset = False
    weight_dependence = 'weight_dependence'
    post_pre = 'post_pre'



    test_mode = False






    if do_plot == 'True':
        do_plot = True
    elif do_plot == 'False':
        do_plot = False
    else:
        raise Exception('Expecting True or False-valued command line argument "do_plot".')


    '''training = pickle.load(open('TrainingNewLabels6laplaceJAFFE.pickle', 'rb'))

    training = pickle.load(open('TrainingNewLabels6laplaceDB.pickle', 'rb'))

    testing = pickle.load(open('TestingNewLabels6laplaceDB.pickle', 'rb'))'''

    '''infile = open('TrainingVideoRAVDES2EM.pickle', 'rb')
    training = pickle.load(infile)
    infile.close()

    infile = open('TestingVideoRAVDES2EM.pickle', 'rb')
    testing = pickle.load(infile)
    infile.close()'''

    infile = open('TestingVideoRAVDESLaplace100IJCNN.pickle' , 'rb') #TrainingVideoRAVDESLaplace100.pickle', 'rb') #TrainingVideoRAVDESLaplace100.pickle', 'rb')  # CK+LAPLACIANTrain.pickle', 'rb')
    training = pickle.load(infile)
    #training = training[0:500]
    infile.close()




    '''infile = open('TestingVideoRAVDESLaplace100.pickle', 'rb') #TestingCOHEN25FEBNoise4.pickle' ,'rb') #TestingVideoRAVDESLaplace100.pickle', 'rb')
    testing = pickle.load(infile)
    print(len(testing['x']))
    infile.close()'''

    # training = np.array(training)

    #testing = pickle.load(open('TestingNewLabels6laplaceJAFFE.pickle', 'rb'))

    #testing = training
    # testing = np.array(testing)

    numTrain = len(training['x'])
    #numTest = len(testing['x'])
    num_examples = numTrain #numTest #numTest  # 2400 #numTest #2400 #900 #1200 #numTrain # 2400 #numTrain
    if reduced_dataset:
        data_size = num_classes * examples_per_class

    elif test_mode:
        data_size = numTest # numTest #numTest  # numTest
    else:
        data_size = numTrain

    # set brian global preferences
    b.set_global_preferences(defaultclock=b.Clock(dt=0.5 * b.ms), useweave=True,
                             gcc_options=['-ffast-math -march=native'], usecodegen=True,
                             usecodegenweave=True, usecodegenstateupdate=True, usecodegenthreshold=False,
                             usenewpropagate=True, usecstdp=True, openmp=False,
                             magic_useframes=False, useweave_linear_diffeq=True)


    # test or train mode
    #test_mode = mode == 'test'

    start = timeit.default_timer()

    data = training  # training # training

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

    n_input = 10000  #
    n_input_sqrt = int(math.sqrt(n_input))

    # number of neurons parameters
    if conv_size == 100 and conv_stride == 0:
        n_e = 1
    else:
        n_e = (((n_input_sqrt - conv_size) / conv_stride) + 1) ** 2

    print(n_e)
    n_e_total = n_e * conv_features
    n_e_sqrt = int(math.sqrt(n_e))
    n_i = n_e
    conv_features_sqrt = int(math.ceil(math.sqrt(conv_features)))

    # time (in seconds) per data example presentation and rest period in between, used to calculate total runtime
    single_example_time = 0.35 * b.second # 35
    resting_time = 0.15 * b.second
    runtime = num_examples * (single_example_time + resting_time)

    # set the update interval
    if test_mode:
        update_interval = num_examples
    else:
        update_interval =  50 # 100

    # weight updates and progress printing intervals
    weight_update_interval =  50 #50
    print_progress_interval = 50 #50

    # rest potential parameters, reset potential parameters, threshold potential parameters, and refractory periods
    v_rest_e, v_rest_i = -65. * b.mV, -60. * b.mV
    v_reset_e, v_reset_i = -65. * b.mV, -45. * b.mV
    v_thresh_e, v_thresh_i = -52. * b.mV, -40. * b.mV
    refrac_e, refrac_i = 5. * b.ms, 2. * b.ms

    # dictionaries for weights and delays
    weight, delay = {}, {}

    # populations, connections, saved connections, etc.
    input_population_names = ['X']
    population_names = ['A']
    input_connection_names = ['XA']
    save_conns = ['XeAe', 'AeAe']

    # weird and bad names for variables, I think
    input_conn_names = ['ee_input']
    recurrent_conn_names = ['ei', 'ie'] #, 'ee']

    # setting weight, delay, and intensity parameters

    weight['ee_input'] = (conv_size ** 2) * 0.1625
    delay['ee_input'] = (0 * b.ms, 10 * b.ms)
    delay['ei_input'] = (0 * b.ms, 5 * b.ms)
    input_intensity = start_input_intensity = 5.0

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


    eqs_stdp_pre_ee = 'pre = 1.; w -= nu_ee_pre * post * w ** exp_ee_pre'
    eqs_stdp_post_ee = 'w += nu_ee_post * pre * (wmax_ee - w) ** exp_ee_post; post = 1.'



    # set ending of filename saves
    ending = 'FER' + '_' + str(conv_size) + '_' + str(conv_stride) + '_' + str(conv_features) + '_' + str(
        n_e)

    b.ion()
    fig_num = 1


    neuron_groups, input_groups, connections, input_connections, stdp_methods, \
    rate_monitors, spike_monitors, spike_counters, output_numbers = {}, {}, {}, {}, {}, {}, {}, {}, {}

    # creating convolution locations inside the input image
    convolution_locations = {}
    for n in xrange(n_e):
        convolution_locations[n] = [((n % n_e_sqrt) * conv_stride + (n // n_e_sqrt) * n_input_sqrt * \
                                     conv_stride) + (x * n_input_sqrt) + y for y in xrange(conv_size) for x in
                                    xrange(conv_size)]


        print(len(convolution_locations[n]))

    # instantiating neuron "vote" monitor
    result_monitor = np.zeros((update_interval, conv_features, n_e))

    # build the spiking neural network
    initialise_network()

    # bookkeeping variables
    previous_spike_count = np.zeros((conv_features, n_e))
    assignments = np.zeros((conv_features, n_e))
    input_numbers = [0] * num_examples
    rates = np.zeros((n_input_sqrt, n_input_sqrt))

    output_numbers['all'] = np.zeros((num_examples, 6))
    output_numbers['most_spiked'] = np.zeros((num_examples, 6))
    output_numbers['top_percent'] = np.zeros((num_examples, 6))

    # run the simulation of the network
    spikemon = []
    run_snn()

    #save spiking history
    np.save('spikefileImage.npy', spikemon)

    # save and plot results
    save_results()

    # evaluate results

    if test_mode:
        evaluate_results()
