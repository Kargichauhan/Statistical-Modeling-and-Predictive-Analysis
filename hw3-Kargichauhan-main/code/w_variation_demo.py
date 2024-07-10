# w_variation_demo.py
# Adaptation of w_variation_demo.m by Clayton T. Morrison 2023
#     From A First Course in Machine Learning, Chapter 2.
#     Simon Rogers, 01/11/11 [simon.rogers@glasgow.ac.uk]
# The bias in the estimate of the variance
# Generate lots of datasets and look at how the average fitted variance
# agrees with the theoretical value
import numpy as np
import matplotlib.pyplot as plt


def run_demo(dim=2):
    # Generate the data sets and fit the parameters

    # true_w = np.array([-2, 3])  # The true model
    true_w = np.zeros(dim)  # The true model
    N_dataset_sizes = np.linspace(20, 1000, 50)  # data set sizes
    N_experiment_repetitions = 10000  # number of fit/predict repetitions

    # Store sum-squared error for each experiment repetition (columns)
    # at each data set size (rows)
    all_ss = np.zeros((N_dataset_sizes.shape[0], N_experiment_repetitions))
    all_ss_corrected = np.zeros((N_dataset_sizes.shape[0], N_experiment_repetitions))

    # The true noise variance
    noisevar = 0.5**2

    # Total number of experiments at different data set sizes
    total = N_dataset_sizes.shape[0]

    print(f'NOTE: this will take a bit of time to run... generating {total} datasets')

    # Generate experiments
    for j in range(total):
        N = int(N_dataset_sizes[j])  # Number of observations in this data set
        print(f'Dim={dim} : processing data set {j + 1} (of {total} ), for data set of size {N}')
        x = np.random.rand(N)
        X = np.zeros((x.shape[0], dim))
        for i in range(dim):
            X[:, i] = np.power(x, i)
        for i in range(N_experiment_repetitions):
            t = np.dot(X, true_w) + np.random.randn(N)*np.sqrt(noisevar)
            w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, t))
            ss = (1.0 / N)*(np.dot(t, t) - np.dot(t, np.dot(X, w)))
            all_ss[j, i] = ss
            ss_corrected = (1.0 / (N - dim)) * (np.dot(t, t) - np.dot(t, np.dot(X, w)))
            all_ss_corrected[j, i] = ss_corrected


    # The expected value of the fitted variance is equal to:
    # $\sigma^2\left(1-\frac{D}{N}\right)$
    # where $D$ is the number of dimensions (2) and $\sigma^2$ is the true
    # variance.
    # Plot the average empirical value of the variance against the
    # theoretical expected value as the size of the datasets increases
    plt.figure()
    plt.scatter(N_dataset_sizes, np.mean(all_ss, 1), color='white', s=40,
                edgecolor='black', label='Mean Estimated from Sample (Uncorrected)')
    plt.scatter(N_dataset_sizes, np.mean(all_ss_corrected, 1), color='white', s=40,
                edgecolor='g', label='Mean Estimated from Sample (Corrected)')
    plt.plot(N_dataset_sizes, noisevar * (1 - dim / N_dataset_sizes),  # 2.0
             color='r', linewidth=2, label='Theoretical Uncorrected')
    plt.plot(N_dataset_sizes, [0.25]*len(N_dataset_sizes),
             color='b', linestyle='dashed', label='Actual Variance')
    plt.xlabel('$N$ = number of samples')
    plt.ylabel(r'$\mathbf{E}\{\widehat{\sigma^2}\}$')
    plt.title(f'Dimensions = {dim}')
    plt.legend(loc=4)

    # uncomment the following line to save the plot
    plt.savefig(f'w_variance_demo_dim={dim}.png', format='png')


if __name__ == "__main__":
    run_demo(dim=2)

    # uncomment the following to run demo for dimensions 0-5
    # for d in range(6):
    #     run_demo(dim=d)

    plt.show()
