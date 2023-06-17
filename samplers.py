from scipy.stats import multivariate_normal
from scipy.stats import uniform

import numpy as np

from math import log
from math import exp


class RandomWalkMetropolis:

    def __init__(self, stdval, num_of_run, network, name, burn_length=1000):
        self.stdval = stdval
        self.num_of_run = num_of_run
        self.burn_length = burn_length
        self.network = network
        self.input_dimension = network.get_total_weight_dimension()
        self.sample_set = np.zeros((self.num_of_run + 1, self.input_dimension))
        self.samplename = name
        self.acceptence_rate = 0

    def metropolis_main_op(self, sample):
        proposal_val = multivariate_normal.rvs(mean=sample, cov=self.stdval)

        try:
            val = self.network.get_unnormalized_weight_posterior(
                proposal_val) / self.network.get_unnormalized_weight_posterior(sample)
        except:
            print('exception')
            val = 2

        uniform_sample = uniform.rvs()
        if uniform_sample < val:
            self.acceptence_rate = self.acceptence_rate + 1
            return proposal_val
        else:
            return sample

    def main_loop(self):
        initial_sample = multivariate_normal.rvs(mean=np.zeros(self.input_dimension), cov=1)
        self.sample_set[0, :] = initial_sample
        for i in range(self.num_of_run):
            self.sample_set[i + 1, :] = self.metropolis_main_op(self.sample_set[i, :])
            print('one_loop_of metropolis step number=' + str(i) + '\033[94m')
        np.save('rwn_sample_set' + str(self.samplename), self.get_samples())

    def get_acceptence_rate(self):
        return self.acceptence_rate / self.num_of_run

    def get_samples(self):
        return self.sample_set[self.burn_length:, :]

    # def calculate_target_mean(self):


class MetropolisHastings:

    def __init__(self, stdval, num_of_run, network, name, burn_length=1000):
        self.stdval = stdval
        self.num_of_run = num_of_run
        self.burn_length = burn_length
        self.network = network
        self.input_dimension = network.get_total_weight_dimension()
        self.sample_set = np.zeros((self.num_of_run + 1, self.input_dimension))
        self.samplename = str(name) + 'hastings'
        self.acceptance_rate = 0

    def metropolis_main_op(self, sample):
        # proposal multivariate normal with identity covariance matrix

        norm = np.linalg.norm(sample)
        print(norm)
        sigmaval1 = self.stdval * ((1 + norm) ** 2) * np.diagflat(np.ones(self.input_dimension))

        proposal_val = multivariate_normal.rvs(mean=sample, cov=sigmaval1)

        norm2 = np.linalg.norm(proposal_val)
        sigmaval2 = self.stdval * ((1 + norm2) ** 2) * np.diagflat(np.ones(self.input_dimension))

        q1 = multivariate_normal.logpdf(proposal_val, mean=sample, cov=sigmaval1)
        q2 = multivariate_normal.logpdf(sample, mean=proposal_val, cov=sigmaval2)
        # multivariate gausssian density # Ratio and random uniform

        try:
            A = log(self.network.get_unnormalized_weight_posterior(proposal_val)) + q2 - \
                log(self.network.get_unnormalized_weight_posterior(sample)) - q1
        except:
            A = 2

        U = log(uniform.rvs())
        if U < A:
            self.acceptance_rate = self.acceptance_rate + 1
            return proposal_val
        else:
            return sample

    def main_loop(self):
        initial_sample = multivariate_normal.rvs(mean=np.zeros(self.input_dimension), cov=1)
        self.sample_set[0, :] = initial_sample
        for i in range(self.num_of_run):
            self.sample_set[i + 1, :] = self.metropolis_main_op(self.sample_set[i, :])
            print('one_loop_of metropolis-hastings step number=' + str(i) + '\033[92m')
        np.save('rwn_sample_set' + str(self.samplename), self.get_samples())

    def get_acceptance_rate(self):
        return self.acceptance_rate / self.num_of_run

    def get_samples(self):
        return self.sample_set[self.burn_length:, :]

    def thin_samples(self, step=200):
        self.sample_set = self.sample_set[:len(self.sample_set):step, :]


class ImportanceSampler:

    def __init__(self, num_of_run, network):
        self.num_of_run = num_of_run
        self.input_dimension = network.get_total_weight_dimension()
        self.network = network
        self.sample_set = np.zeros((self.num_of_run, self.input_dimension))

    def importance_main_op(self, sample_val, test_data, test_label):
        dist_vals_denom = multivariate_normal.logpdf(sample_val, mean=np.zeros(self.input_dimension), cov=1)
        unnormlized_posterior_vals = self.network.get_unnormalized_weight_posterior(sample_val)
        likehood_vals = self.network.test_likehood_query(sample_val, test_label, test_data)

        numeratorator = log(likehood_vals) + unnormlized_posterior_vals - (dist_vals_denom)

        demoninator = unnormlized_posterior_vals - (dist_vals_denom)

        self.num_of_run = self.num_of_run - 1
        print('one_loop_of importance-sampling step number =  ' + str(self.num_of_run) + '\033[0;35m')
        print(exp(numeratorator))
        print(exp(demoninator))

        return exp(numeratorator), exp(demoninator)

    def importance_mapping(self, test_data, test_label):
        vals = multivariate_normal.rvs(size=self.num_of_run, mean=np.zeros((self.input_dimension)), cov=1)
        setofresults = [self.importance_main_op(val, test_data, test_label) for val in vals]
        # setofresults = Parallel(n_jobs=-1,verbose=30)(delayed(self.importance_main_op)(val, test_data, test_label) for val in vals)
        np.save('importance_sampling_set', setofresults)
        numval, den_val = list(zip(*setofresults))
        finalresult = sum(numval) / sum(den_val)
        return finalresult

    def run(self, test_data, test_label):
        return self.importance_mapping(test_data, test_label)


class AdaptiveMonteCarlo:

    def __init__(self, stdval, num_of_run, network, name, burn_length=20000):
        self.stdval = stdval
        self.num_of_run = num_of_run
        self.burn_length = burn_length
        self.network = network
        self.input_dimension = network.get_total_weight_dimension()
        self.mult = ((2.38) ** 2) / self.input_dimension
        self.sample_set = np.zeros((self.num_of_run + 1, self.input_dimension))
        self.samplename = str(name) + 'adaptive'
        self.acceptance_rate = 0
        self.counter = 0
        self.epsilon = 0.1

    def adaptive_main_op(self, sample):
        # proposal multivariate normal with identity covariance matrix

        self.counter = self.counter + 1

        if (self.counter < self.input_dimension ** 2):
            propSigma = 0.1 * np.eye(self.input_dimension)
        else:
            covsofar = np.cov(np.transpose(self.sample_set[:self.counter, :]))
            propSigma = self.mult * covsofar + self.epsilon * np.eye(self.input_dimension)

        proposal_val = multivariate_normal.rvs(mean=sample, cov=propSigma)  # proposal value
        # multivariate gausssian density # Ratio and random uniform
        try:
            A = log(self.network.get_unnormalized_weight_posterior(proposal_val)) - \
                log(self.network.get_unnormalized_weight_posterior(sample))
        except:
            A = 2

        U = log(uniform.rvs())
        if U < A:
            self.acceptence_rate = self.acceptance_rate + 1
            return proposal_val
        else:
            return sample

    def main_loop(self):
        initial_sample = multivariate_normal.rvs(mean=np.zeros(self.input_dimension), cov=1)
        self.sample_set[0, :] = initial_sample
        for i in range(self.num_of_run):
            self.sample_set[i + 1, :] = self.adaptive_main_op(self.sample_set[i, :])
            print('one_loop_of adaptive step number=' + str(i) + '\033[92m')
        np.save('rwn_sample_set' + str(self.samplename), self.get_samples())

    def get_acceptance_rate(self):
        return self.acceptence_rate / self.num_of_run

    def get_samples(self):
        return self.sample_set[self.burn_length:, :]

    def thin_samples(self, step=200):
        self.sample_set = self.sample_set[:len(self.sample_set):step, :]


class IndependenceSampler:

    def __init__(self, stdval, num_of_run, network, name, burn_length=1000):
        self.stdval = stdval
        self.num_of_run = num_of_run
        self.burn_length = burn_length
        self.network = network
        self.input_dimension = network.get_total_weight_dimension()
        self.sample_set = np.zeros((self.num_of_run + 1, self.input_dimension))
        self.samplename = name
        self.acceptance_rate = 0

    def main_op(self, sample):
        proposal_val = multivariate_normal.rvs(mean=np.zeros(self.input_dimension), cov=self.stdval)

        q1 = multivariate_normal.logpdf(proposal_val, mean=np.zeros(self.input_dimension), cov=self.stdval)
        q2 = multivariate_normal.logpdf(sample, mean=np.zeros(self.input_dimension), cov=self.stdval)

        try:
            val = log(self.network.get_unnormalized_weight_posterior(proposal_val)) + q2 - \
                log(self.network.get_unnormalized_weight_posterior(sample)) - q1

        except:
            print('exception')
            val = 2

        log_uniform_sample = log(uniform.rvs())
        if log_uniform_sample < val:
            self.acceptance_rate = self.acceptance_rate + 1
            return proposal_val
        else:
            return sample

    def main_loop(self):
        initial_sample = multivariate_normal.rvs(mean=np.zeros(self.input_dimension), cov=1)
        self.sample_set[0, :] = initial_sample
        for i in range(self.num_of_run):
            self.sample_set[i + 1, :] = self.main_op(self.sample_set[i, :])
            print('one_loop_of metropolis step number=' + str(i) + '\033[94m')
        np.save('ind_sample_set' + str(self.samplename), self.get_samples())

    def get_acceptance_rate(self):
        return self.acceptance_rate / self.num_of_run

    def get_samples(self):
        return self.sample_set[self.burn_length:, :]
