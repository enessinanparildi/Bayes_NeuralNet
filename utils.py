import numpy as np

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

from math import sqrt

import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer

import matplotlib.pyplot as plt


def save_dataset(indut_dim, size, name):
    X_train, X_test, y_train, y_test = prapare_dataset(indut_dim=indut_dim, size=size)
    np.save('mcdatatrain' + str(name) + '.npy', X_train)
    np.save('mcdatatest' + str(name) + '.npy', X_test)
    np.save('mcdatatrainlabels' + str(name) + '.npy', y_train)
    np.save('mcdatatestlabels' + str(name) + '.npy', y_test)


def softmax(scores):
    e_x = np.exp(scores - np.max(scores))
    return e_x / e_x.sum()


def thin_samples(samples, step=100):
    samples = samples[::step, :]
    return samples


def iid_error(sample_set):
    return np.std(sample_set) / sqrt(len(sample_set))


def autocorr_graph(sample_set):
    plot_acf(sample_set, lags=100)
    plt.show()


def varfact(sample_set):
    acfset = acf(x=sample_set)
    return 2 * np.sum(acfset) - 1


def load_samples(thin=False):
    samples = np.load('ind_sample_set11.npy')
    print('sample_set_size:' + str(samples.shape))
    if thin:
        samples = thin_samples(samples[2000:, :])
    return samples


def run_error_stats(network, test_data, test_label, ind):
    samples = load_samples()
    estimate_mean, result_set = network.get_predictive_dist_estimate(samples, test_label, test_data)
    print('ind_sample_set11')
    print(ind)
    print('ground truth label : ' + str(test_label))
    print('uncertainty estimate for test instance : ' + str(estimate_mean))
    varf = varfact(result_set)
    iiderr = iid_error(result_set)
    print('varfact : ' + str(varf))
    print('iid_error : ' + str(iiderr))
    print('true_error : ' + str(iiderr * sqrt(varf)))
    print('confidence_interval: (' + str(estimate_mean - 1.96 * iiderr) + ' -- ' + str(
        estimate_mean + 1.96 * iiderr) + ')')

    print(samples[:, 8])
    histogram_graph(samples[:, 1])
    autocorr_graph(result_set)


def histogram_graph(values):
    print(values.shape)
    plt.hist(values, density=True, facecolor='g', bins=1000)
    plt.ylabel('Probability')
    plt.title('Posterior distribution of a weight')
    plt.text(1.0, 1.0, ' posterior mean = ' + str(np.mean(values)) + '  posterior std = ' + str(np.std(values)))
    plt.grid(True)
    plt.show()


def choose_fraction_dataset(dataset, labels, size):
    class_zero_inds = np.where(labels == 0)[0]
    class_one_inds = np.where(labels == 1)[0]
    class_zero_chosen_inds = np.random.choice(class_zero_inds, size)
    class_one_chosen_inds = np.random.choice(class_one_inds, size)
    all_inds = np.concatenate((class_one_chosen_inds, class_zero_chosen_inds), axis=0)
    np.random.shuffle(all_inds)
    chosen_dataset = dataset[all_inds, :]
    chosen_labels = labels[all_inds]
    return chosen_dataset, chosen_labels


def tfidf_transform(frequency_data, use_idf):
    transformer = TfidfTransformer(norm="l2",
                                   use_idf=use_idf,
                                   smooth_idf=True,
                                   sublinear_tf=False)
    return transformer.fit_transform(frequency_data)


def get_freq_dataset():
    frequency_data_dir = "C:\\Users\\Ben\\Google Drive\\bigproject\\dataset\\all_malware_benign_master.csv"
    frequency_data, frequency_datalabels, frequency_numberlabels, uniquelabels, classsizes = preprocessing.process_data_v2(
        frequency_data_dir)
    binary_labels = np.ones(len(frequency_numberlabels))
    binary_labels[np.where(np.array(frequency_numberlabels) == 3)[0]] = 0
    inds = np.arange(len(frequency_numberlabels))
    np.random.shuffle(inds)
    inds.astype(np.int_)
    return frequency_data[inds, :], np.array(frequency_numberlabels)[inds], binary_labels[inds]


def prapare_dataset(indut_dim, size):
    frequency_data, frequency_numberlabels, binary_labels = get_freq_dataset()
    norm_frequency_data = tfidf_transform(frequency_data, use_idf=False)
    whole_dataset, whole_labels = choose_fraction_dataset(norm_frequency_data, binary_labels, size=size)
    whole_dataset = whole_dataset.toarray()
    reduced_whole_dataset = PCA(n_components=indut_dim).fit_transform(whole_dataset)
    whole_dataset = StandardScaler().fit_transform(reduced_whole_dataset)
    X_train, X_test, y_train, y_test = train_test_split(whole_dataset, whole_labels, test_size=.1, random_state=42)

    return X_train, X_test, y_train, y_test
