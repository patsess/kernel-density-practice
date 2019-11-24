
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, poisson
from itertools import product
import statsmodels.api as sm
from kde_multivariate_cond_wrapper import KDEMultiCondWrapper
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

"""note: results are VERY unsatisfactory!!"""


class SimpleCondKernelDensityExample(object):
    def __init__(self, poisson=False):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info('initialising a {} instance'
                         .format(self.__class__.__name__))

        # set attributes for easy access
        self._poisson = poisson

        # helpers for properties
        self._random_state = None
        self._params = None

        # initialise attributes
        self._x = None
        self._y = None

    @property
    def random_state(self):
        if self._random_state is None:
            self._random_state = 666

        return self._random_state

    @property
    def params(self):
        if self._params is None:
            self._params = {
                'x': {
                    'exp': 3.7,
                    'std': 2.1,
                },
            }

            if self._poisson:
                self._params['x']['y_coef'] = 0.3
            else:
                self._params.update({
                    'y_given_x': {
                        'std': 3.3
                    },
                })
                self._params['x']['y_coef'] = 9.5

        return self._params

    def load_dataset(self, n_samples=1000, plot=False):
        self.logger.info('loading data set ({} samples)...'.format(n_samples))

        # generate feature
        x = norm(loc=self.params['x']['exp'],
                 scale=self.params['x']['std']).rvs(
            n_samples, random_state=self.random_state)

        # evaluate deterministic part (without any link function initially)
        mu = self.params['x']['y_coef'] * x

        # generate target variable
        if self._poisson:
            mu = np.exp(mu)
            y = poisson(mu).rvs(random_state=self.random_state)

        else:
            # generate residuals for conditional target
            y_resid = norm(loc=0.,
                           scale=self.params['y_given_x']['std']).rvs(
                n_samples, random_state=self.random_state)

            y = mu + y_resid

        if plot:
            plt.plot(self._x, self._y, 'bo')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        # load attributes for data set
        self._x = x
        self._y = y
        self.logger.info('...attributes for data set loaded')

    def model_using_sm_kdemc(self, bw='cv_ml', use_sklearn_cv=False):
        if self._poisson:
            raise Exception('functionality does not seem to work with '
                            'discrete data; only setting dep_type to '
                            'continuous (which makes no sense) leads to '
                            'reasonable results, and even then using cv_ml '
                            'for bw gives silly results (perhaps due to '
                            'poor estimates for the feature). It may be that '
                            'playing with specified bws (e.g. bw=[0.01, 2.]) '
                            'could gives sensible results')

        # note: from my experience, bw=='normal_reference' underestimates the
        # underlying conditional variance in the target, whereas 'cv_ml' does
        # not
        self.logger.info('modelling using conditional kernel density '
                         'estimation from the statsmodels library...')

        dep_type = ('o' if self._poisson else 'c')
        dens_y = sm.nonparametric.KDEMultivariate(data=self._y,
                                                  var_type=dep_type, bw=bw)
        # note: probability mass can be given to minus numbers, so this
        # mass should be given to positive values (or only zero?) for
        # non-negative targets like Poisson variables
        self.logger.info('unconditional kernel density fitted with bw {} '
                         '(using method {})'.format(dens_y.bw, bw))

        if use_sklearn_cv:
            dens_y_given_x = KDEMultiCondWrapper(dep_type=dep_type,
                                                 indep_type='c')
            param_grid = {'bw': list(product([0.01, 5., 6., 7., 8.],
                                             [0.01, 0.5, 0.6, 0.7, 0.8]))}
            # param_grid = {'bw': list(product([5e-3, 8e-3, 1e-2, 2e-1],
            #                                  [5e-3, 8e-3, 1e-2, 2e-1]))}
            cv_ = GridSearchCV(dens_y_given_x, param_grid, cv=5, verbose=0)
            self.logger.info('fitting kernel density with hyper-parameter '
                             'candidates {}'.format(param_grid))
            cv_.fit(X=self._x, y=self._y)
            self.logger.info('kernel density fitted with hyper-parameters: {}'
                             .format(cv_.best_params_))
            dens_y_given_x = cv_.best_estimator_.model_
        else:
            dens_y_given_x = sm.nonparametric.KDEMultivariateConditional(
                endog=self._y, exog=self._x, dep_type=dep_type, indep_type='c',
                bw=bw)
            self.logger.info('conditional kernel density fitted with bw {} '
                             '(using method {})'.format(dens_y_given_x.bw, bw))

        if self._poisson:
            eval_ys = np.array(range(-4, self._y.max() + 5))
        else:
            eval_ys = np.linspace(start=self._y.min() - 3.,
                                  stop=self._y.max() + 3., num=1000)

        eval_dens_y = dens_y.pdf(eval_ys)
        # note: the sm.nonparametric.KDEMultivariate.pdf() method gives
        # probability mass to non-integer values even for count data (e.g.
        # Poisson variables) when pdf() evaluates a probability, BUT in these
        # circumstances this does not prevent the integer-only evaluations
        # summing to (effectively) one, so it does not seem a problem IF bw
        # is selected well (e.g. not using bw='normal_reference')

        plt.plot(eval_ys, eval_dens_y, label='unconditional')
        for eval_x_ in np.linspace(start=self._x.min(), stop=self._x.max(),
                                   num=10):
            eval_x = np.repeat(eval_x_, repeats=eval_ys.size)
            eval_dens_y_given_x = dens_y_given_x.pdf(endog_predict=eval_ys,
                                                     exog_predict=eval_x)

            plt.plot(eval_ys, eval_dens_y_given_x,
                     label='given_x_{}'.format(np.round(eval_x_, 1)))

            self._plot_expected(eval_x_=eval_x_, eval_ys=eval_ys)

        plt.ylim(bottom=0., top=np.minimum(0.15, plt.ylim()[1]))  # cap v-axis
        plt.legend()
        plt.show()

        # TODO: try method with bi-variate example (2 targets and 2 features)

    def _plot_expected(self, eval_x_, eval_ys):
        if self._poisson:
            expected_y_ = np.exp(self.params['x']['y_coef'] * eval_x_)
            plt.axvline(expected_y_, color='grey', linestyle='--', alpha=0.5)
            plt.plot(
                eval_ys, poisson(expected_y_).pmf(eval_ys),
                color='grey', linestyle='--', alpha=0.5)

        else:
            expected_y_ = self.params['x']['y_coef'] * eval_x_
            plt.axvline(expected_y_, color='grey', linestyle='--', alpha=0.5)
            plt.plot(
                eval_ys, norm(loc=expected_y_, scale=self.params[
                    'y_given_x']['std']).pdf(eval_ys),
                color='grey', linestyle='--', alpha=0.5)

    def model_using_guassian_processes(self, param_grid=None):
        raise Exception('it seems that GPs are used to model flexible '
                        'relationships between features and a target, not '
                        'to produce a non-parametic conditional distribution '
                        'for the target. Unless there is another way to use '
                        'them')

        self.logger.info('modelling using Gaussian processes from sklearn...')

        reg = GaussianProcessRegressor(random_state=self.random_state)
        # TODO: research important hyper-parameters to optimise and cv

        X_ = self._x.reshape((-1, 1))
        pipeline_list = [
            # ('scaler', StandardScaler()),
            ('model', reg)
        ]  # TODO: is scaling advantageous with GPs?

        if param_grid is None:
            param_grid = {
                'model__alpha': [1e-10, 1e-5, 1e-1],
            }

        reg = GridSearchCV(Pipeline(pipeline_list), param_grid=param_grid,
                           cv=3, refit=True, verbose=0)

        self.logger.info('fitting pipeline with hyper-parameter candidates {}'
                         .format(param_grid))
        reg.fit(X=X_, y=self._y)
        self.logger.info('pipeline fitted with hyper-parameters: {}'
                         .format(reg.best_params_))
        reg = reg.best_estimator_.named_steps['model']

        eval_ys = np.linspace(start=self._y.min() - 10.,
                              stop=self._y.max() + 10., num=1000)

        sns.kdeplot(self._y, label='unconditional')
        for eval_x in np.linspace(start=self._x.min(), stop=self._x.max(),
                                  num=10):
            X_ = np.array(eval_x).reshape((-1, 1))
            y_cond_sample = reg.sample_y(X=X_, n_samples=1000,
                                         random_state=self.random_state)

            sns.kdeplot(y_cond_sample.flatten(),
                        label='given_x_{}'.format(np.round(eval_x, 1)))

            self._plot_expected(eval_x_=eval_x_, eval_ys=eval_ys)

        plt.legend()
        plt.show()


if __name__ == '__main__':
    ckd = SimpleCondKernelDensityExample(poisson=False)
    ckd.load_dataset()
    ckd.model_using_sm_kdemc(bw='normal_reference', use_sklearn_cv=True)
    # ckd.model_using_guassian_processes()

    # TODO: try sklearn's kde even though it's not conditional? Check that
    # TODO I can conditional from it!!!
