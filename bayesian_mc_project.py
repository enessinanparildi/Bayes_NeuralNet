import numpy as np
import preprocessing

from math import log, exp, pi, sqrt
from scipy.stats import multivariate_normal, uniform
from itertools import accumulate, tee, repeat
from joblib import Parallel, delayed

from utils import softmax, save_dataset, run_error_stats
import samplers


class Network:

    def __init__(self, layer_info_tuple, input_train, input_test, labels):
        self.layer_info_tuple = layer_info_tuple
        self.layer_num = len(layer_info_tuple)
        self.input_train = input_train
        self.input_feature_size = input_train.shape[1]
        self.labels = labels
        self.input_test = input_test

    def one_layer_forward_pass(self, input, weights, bias):
        input = input
        lin_output = np.dot(input, weights) + np.reshape(bias, (1, bias.shape[0]))
        output = self.tanh(lin_output)
        return output

    def softmax(self, input):
        return (np.exp(input).T / np.sum(np.exp(input), axis=1)).T

    @staticmethod
    def sigmoid(input):
        return 1 / (1 + np.exp(-input))

    def tanh(self, input):
        return np.tanh(input)

    def forward_neural_network_calculation(self, all_weight_set, all_bias_set, input):
        first_output = self.one_layer_forward_pass(input, all_weight_set[0], all_bias_set[0])
        second_output = self.one_layer_forward_pass(first_output, all_weight_set[1], all_bias_set[1])
        third_output = self.one_layer_forward_pass(second_output, all_weight_set[2], all_bias_set[2])
        return third_output

    def open_vals_set(self, vals):

        if len(vals) != self.get_total_weight_dimension():
            print('invalid_shape')

        lenw1 = self.input_feature_size * self.layer_info_tuple[0]
        lenw2 = self.layer_info_tuple[0] * self.layer_info_tuple[1]
        lenw3 = self.layer_info_tuple[1] * self.layer_info_tuple[2]
        lenb1 = self.layer_info_tuple[0]
        lenb2 = self.layer_info_tuple[1]
        lenb3 = self.layer_info_tuple[2]

        allsizes = (self.input_feature_size,) + self.layer_info_tuple
        lengths = [lenw1, lenw2, lenw3, lenb1, lenb2, lenb3]
        accleng = tuple(accumulate(lengths))
        splittedvals = [vals[i: j] for i, j in zip((0,) + accleng, accleng)]
        splitted_weigths = splittedvals[:3]
        splitted_biases = splittedvals[3:]

        a, b = tee(list(allsizes))
        next(b, None)
        shaple_tuples = list(zip(a, b))

        shaped_weigths = [np.array(weigth).reshape((shape_tuple[0], shape_tuple[1])) for weigth, shape_tuple in
                          zip(splitted_weigths, shaple_tuples)]
        shaped_biases = [np.array(bias) for bias in splitted_biases]

        return shaped_weigths, shaped_biases

    def get_total_weight_dimension(self):
        weight_dim_num = self.input_feature_size * self.layer_info_tuple[0] + \
                         self.layer_info_tuple[0] * self.layer_info_tuple[1] + \
                         self.layer_info_tuple[1] * self.layer_info_tuple[2]
        bias_dim_num = sum(self.layer_info_tuple)
        return weight_dim_num + bias_dim_num

    def label_eval(self, score_label_tuple):
        score, label = score_label_tuple
        score = score[0]
        if label == 1:
            return log(self.sigmoid(np.array(score[1]).reshape((1, -1))))
        else:
            return log(1 - self.sigmoid(np.array(score[0]).reshape((1, -1))))

    def label_eval_v2(self, whole_data_tuple):
        all_weight_instance, all_bias_instance, label, x = whole_data_tuple
        score = self.forward_neural_network_calculation(all_weight_instance, all_bias_instance, x)
        score = score[0]

        if label == 1:
            return log(self.sigmoid(np.array(score[1]).reshape((1, -1))))
        else:
            return log(1 - self.sigmoid(np.array(score[0]).reshape((1, -1))))

    def likehood_sample_v2(self, all_weight_instance, all_bias_instance):
        # #scores = Parallel(n_jobs=-1,prefer='threads')(delayed(self.forward_neural_network_calculation)(
        # all_weight_instance, all_bias_instance, x) for x in  self.input_train) logits = sum(Parallel(n_jobs=-1,
        # prefer='threads' )(delayed(self.label_eval)(x) for x in zip(scores, self.labels)))
        tuple_set = zip(repeat(all_weight_instance, len(self.labels)), repeat(all_bias_instance, len(self.labels)),
                        self.labels, self.input_train)
        logits = sum(map(self.label_eval_v2, tuple_set))
        # logits = sum(Parallel(n_jobs=-1,prefer='threads' )(delayed(self.label_eval_v2)(x) for x in tuple_set))
        return logits

    def standard_multivariate_log_normal(self, input):
        inputflat = input.flatten()
        dim = len(inputflat)
        res = exp(-0.5 * np.dot(inputflat, inputflat)) / (sqrt(pow(2 * pi, dim)))
        return log(res)

    def test_likehood_query(self, instance_parameter, test_query_labels, test_query_data):
        all_weights_instance, all_biases_instance = self.open_vals_set(instance_parameter)
        scores = self.forward_neural_network_calculation(all_weights_instance, all_biases_instance, test_query_data)
        prob_scores = np.apply_along_axis(softmax, arr=scores, axis=1)
        print(prob_scores)
        return prob_scores[0, int(test_query_labels)]

    def get_predictive_dist_estimate(self, samples, test_query_labels, test_query_data):
        # num_of_piece = cpu_count()-1
        # split_data = [ samples[val:val+int(len(samples)/num_of_piece)] for val in range(0, len(samples), int(len(samples)/num_of_piece))]
        scores = Parallel(n_jobs=-1, verbose=0)(
            delayed(self.test_likehood_query)(instance_val, test_query_labels, test_query_data) for instance_val in
            samples)

        return (sum(scores) / len(scores), scores)

    def likehood_sample(self, all_weight_instance, all_bias_instance):
        #
        scores = self.forward_neural_network_calculation(all_weight_instance, all_bias_instance, self.input_train)
        prob_scores = np.apply_along_axis(softmax, arr=scores, axis=1)
        logits = np.log(np.array([prob_scores[i, int(label)] for i, label in enumerate(self.labels)]))
        # print('sum_of_logit ' + str(np.sum(logits)))
        return np.sum(logits)

    def log_prior_sample(self, all_weights, all_biases):
        all_weights.extend(all_biases)
        scores = sum([self.standard_multivariate_log_normal(x) for x in all_weights])
        # print('sum_of_scores ' + str(scores))

        return scores

    def get_unnormalized_weight_posterior(self, instance_parameter):
        all_weights_instance, all_biases_instance = self.open_vals_set(instance_parameter)
        # return exp(self.likehood_sample(all_weights_instance, all_biases_instance)
        #           + self.log_prior_sample(all_weights_instance, all_biases_instance))

        return exp(self.likehood_sample(all_weights_instance, all_biases_instance)
                   + self.log_prior_sample(all_weights_instance, all_biases_instance))

    # def calculate_target_mean(self):


def main():
    indut_dim = 4
    stdval_has = 1
    stdval_rwm = 0.04
    num_of_run_rwm = 100000
    size = 300
    save_new = False
    name = 11
    namefiles = 7
    run_sampler_rwm = False
    run_sampler_hastings = False
    run_sampler_importance = False
    run_sampler_independence = True
    run_sampler_adaptive = False

    if save_new:
        save_dataset(indut_dim, size, name)

    X_train = np.load('mcdatatrain' + str(namefiles) + '.npy')
    X_test = np.load('mcdatatest' + str(namefiles) + '.npy')
    y_train = np.load('mcdatatrainlabels' + str(namefiles) + '.npy')
    y_test = np.load('mcdatatestlabels' + str(namefiles) + '.npy')

    print(X_train.shape)

    bayes_net = Network((2, 5, 2), X_train, X_test, y_train)
    if run_sampler_rwm:
        print('dimensionality = ' + str(bayes_net.get_total_weight_dimension()))
        rwm_sampler = samplers.RandomWalkMetropolis(stdval=stdval_rwm, num_of_run=num_of_run_rwm, network=bayes_net,
                                                    name=name)
        rwm_sampler.main_loop()
        print('acceptence_rate_rwm = ' + str(rwm_sampler.get_acceptence_rate()))

    if run_sampler_hastings:
        print('dimensionality = ' + str(bayes_net.get_total_weight_dimension()))
        mh_sampler = samplers.MetropolisHastings(stdval=stdval_has, num_of_run=num_of_run_rwm, network=bayes_net,
                                                 name=name)
        mh_sampler.main_loop()
        print('acceptence_rate_mh = ' + str(mh_sampler.get_acceptance_rate()))

    if run_sampler_importance:
        imp_sampler = samplers.ImportanceSampler(num_of_run=num_of_run_rwm, network=bayes_net)
        result = imp_sampler.run(X_test[20], y_test[20])
        print('importance sampling result = ' + str(result))

    if run_sampler_independence:
        ind_sampler = samplers.IndependenceSampler(stdval=stdval_has, num_of_run=num_of_run_rwm, network=bayes_net,
                                                   name=name)
        ind_sampler.main_loop()
        print('acceptence_rate_indepen = ' + str(ind_sampler.get_acceptance_rate()))
        print(ind_sampler.get_acceptance_rate())

    if run_sampler_adaptive:
        adap_sampler = samplers.AdaptiveMonteCarlo(stdval=stdval_has, num_of_run=num_of_run_rwm, network=bayes_net,
                                                   name=name)
        adap_sampler.main_loop()
        print('acceptence_rate_indepen = ' + str(adap_sampler.get_acceptance_rate()))

    # print(X_test[15])
    ind = 20
    run_error_stats(bayes_net, X_test[ind], y_test[ind], ind)


if __name__ == "__main__":
    main()
