import predictive_variance
import pytest
import numpy
import os


X = numpy.array([[ 1.,          1.42525101],
                 [ 1.,          1.53322081],
                 [ 1.,          0.60933506],
                 [ 1.,         -0.6096406 ],
                 [ 1.,          5.45990756],
                 [ 1.,          1.22234057],
                 [ 1.,          5.64133452],
                 [ 1.,         -1.66809007],
                 [ 1.,          6.19463911],
                 [ 1.,         -0.29796103],
                 [ 1.,          6.57939482],
                 [ 1.,          5.45185819],
                 [ 1.,         -0.42518873],
                 [ 1.,          0.1971685 ],
                 [ 1.,          1.14307983],
                 [ 1.,          5.91440535],
                 [ 1.,          1.22688258],
                 [ 1.,          0.02437595],
                 [ 1.,         -1.10799651],
                 [ 1.,         -0.74301144],
                 [ 1.,          6.54345734],
                 [ 1.,          6.55392953],
                 [ 1.,          5.01681306],
                 [ 1.,          5.48088847],
                 [ 1.,          5.37850252],
                 [ 1.,         -1.49433942],
                 [ 1.,          0.92653472],
                 [ 1.,         -1.36712804],
                 [ 1.,          5.48355717],
                 [ 1.,         -1.85881724]])

w = numpy.array([ 1.27818407, -1.49782412])

t = numpy.array([  0.44079819,  -6.76323709,   2.69317774,   5.84717065, -14.38040541,
                   1.00820624, -13.57717539,  -0.89883863,  -4.70829286,   8.26206432,
                   1.72470151, -14.54524638,   5.92425056,   5.83775495,   0.32436047,
                   -9.36530044,  2.83670195,   8.45518683,   2.73483539,   4.11611837,
                   6.24500092,    3.6600345, -15.98205868, -12.42270341, -16.99273446,
                   -0.32113962,   4.17571024,  -0.70172683, -13.45334347,  -4.33149775])


@pytest.fixture
def exercise_results_ex5c_predictive_variance():
    return predictive_variance.calculate_cov_w(X, w, t)


def test_ex5c_predictive_variance(exercise_results_ex5c_predictive_variance):
    ex5c_cov_w = exercise_results_ex5c_predictive_variance
    ex5c_covw_solution = numpy.array([[ 2.35598884, -0.37435139], [-0.37435139,  0.16410594]])
    assert ex5c_cov_w == pytest.approx(ex5c_covw_solution)


DATA_ROOT = 'data'
PATH_TO_SYNTH_DATA = os.path.join(DATA_ROOT, 'synth_data.csv')

FIGURES_ROOT = 'figures'
PATH_TO_EX5C_FN_NAME_BASE = os.path.join(FIGURES_ROOT, 'ex5c_sample_fn_order')

# 2023
XMIN = -2.
XMAX = 7.
XMIN_REMOVE = 2.5
XMAX_REMOVE = 4.5
### SOLUTION START ###
# 2021
# xmin = -4.
# xmax = 5.
# xmin_remove = -1
# xmax_remove = 1
### SOLUTION END ###

x, t = predictive_variance.read_data_from_file(PATH_TO_SYNTH_DATA)
pos = ((x >= XMIN_REMOVE) & (x <= XMAX_REMOVE)).nonzero()
x = numpy.delete(x, pos, 0)
t = numpy.delete(t, pos, 0)
orders = (1, 3, 5, 9)
num_function_samples = 20


# The following test is here just to generate your figures when the tests are run
# This is expected to pass right a way

@pytest.fixture
def exercise_results_ex5c():
    return predictive_variance.plot_functions_sampling_from_covw\
        (x, t, orders, num_function_samples,
         XMIN, XMAX, PATH_TO_EX5C_FN_NAME_BASE)


def test_ex5c_plot_functions_sampling_from_covw(exercise_results_ex5c):
    _ = exercise_results_ex5c
    assert os.path.exists('figures/ex5c_sample_fn_order-1.png')
    assert os.path.exists('figures/ex5c_sample_fn_order-3.png')
    assert os.path.exists('figures/ex5c_sample_fn_order-5.png')
    assert os.path.exists('figures/ex5c_sample_fn_order-9.png')
