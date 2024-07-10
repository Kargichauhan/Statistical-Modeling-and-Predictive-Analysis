import predictive_variance
import pytest
import numpy
import os


DATA_ROOT = 'data'
PATH_TO_SYNTH_DATA_SETS = os.path.join(DATA_ROOT, 'synth_data_sets.csv')


FIGURES_ROOT = 'figures'
PATH_TO_EX6_FN_NAME_BASE = os.path.join(FIGURES_ROOT, 'ex6_sample_fn_order')


# 2023
XMIN = -2.
XMAX = 7.
NOISE_VAR = 6.
### SOLUTION START ###
# 2021
# XMIN = -4.
# XMAX = 5.
# NOISE_VAR = 6.
### SOLUTION END ###

@pytest.fixture
def exercise_results():
    return predictive_variance.exercise_6(orders=(1, 3, 5, 9),
                                          noise_var=NOISE_VAR,
                                          xmin=XMIN,
                                          xmax=XMAX,
                                          data_sets_source_path=PATH_TO_SYNTH_DATA_SETS,
                                          sampled_fn_figure_path_base=PATH_TO_EX6_FN_NAME_BASE)


SOLUTION = \
    ((numpy.array([-1.00710708, -0.42106081]), -0.16498545396456532),
     (numpy.array([ 4.35275546, -2.39401441, -2.79961287,  0.48347419]), -5.925460729020613),
     (numpy.array([ 4.64597994, -1.18590451, -3.11068649,  0.16822632,  0.11162946, -0.00920029]),
      -4.690286958747697),
     (numpy.array([ 4.89843317e+00,  2.89992839e+00, -4.12019538e+00, -3.90667353e+00,
                    1.76530160e+00,  4.90612333e-01, -3.88232956e-01,  8.03808872e-02,
                    -6.95189056e-03,  2.13134315e-04]),
      -10.608036999656179))


def test_ex6_sampled_fns(exercise_results):
    ex6_results = exercise_results
    for (w_ex6, last_t_ex6), (w_sol, last_t_sol) in zip(ex6_results, SOLUTION):
        if w_ex6 is None:
            assert False
        else:
            assert numpy.allclose(w_ex6, w_sol)
        assert last_t_ex6 == pytest.approx(last_t_sol)
