# -*- coding: utf-8 -*-
import random
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from . import funcs

# Default settings.
MAX_SEARCHES = 25
SOLVER = 'Nelder-Mead'

class LPPLS:
    """Class for Log-Periodic Power Law Singularity Model.

    LPPLS is a model that captures feedback phenomena such as herding and
    imitation. It can provide an early warning of market instabilities by
    estimating indicators for positive and negative bubbles.
    The model comes as a combination of mathematical and statistical physics of
    phase transitions, behavorial finance, and economic theory of bubbles.

    For details about the model see for example:
      - Sornette, Johansen & Bouchaud (1996), *"Stock market crashes, precursors
        and replicas"*, Journal de Physique I 6(1)
      - Sornette, Demos, Zhang, Cauwels, Filimonov & Zhang (2015), *"Real-time
      prediction and post-mortem analysis of the shanghai 2015 stock market
      bubble and crash"*, Swiss Finance Institute Research Paper (15-31)

    For implementation details and estimation of confidence indicators see for
    example:
      - Jeremy (2020), *"Prediction of financial bubbles and backtesting of a
        trading strategy"*, Master Thesis at Imperial College London

    Attributes
    ----------
    tc_ : float
        Critical time, i.e. when the bubble crashes.

    m_ : float
        Coefficient that determines the existence of price singularity.

    w_ : float
        Pulsation of the oscillations.

    a_ : float
        Logarithmic price at critical time.

    b_ : float
        Amplitude of power law. B < 0 represents a positive bubble, B > 0
        represents a negative bubble.

    c_ : float
        Magnitude of the oscillations.

    c1_ : float
        First term that replaces magnitude and phase of oscillations.

    c2_ : float
        Second term that replaces magnitude and phase of oscillations.

    damp : float
        Damping over the observation window.

    osc : float
        Number of oscillations over the observation window.

    Examples
    --------
    >>> import markets
    >>> model = markets.LPPLS()
    >>> model.fit(X, y)
    [1]

    """

    def __init__(self):
        """Initializes the LPPLS model."""

        # Parameters of model.
        self.tc_ = None
        self.m_ = None
        self.w_ = None
        self.a_ = None
        self.b_ = None
        self.c_ = None
        self.c1_ = None
        self.c2_ = None
        self.damp = None
        self.osc = None

    def fit(self, time, price, max_searches = MAX_SEARCHES,
        solver = SOLVER, persist = True):
        """Estimates parameters of LPPLS model with least-square method.

        For details see:
            Filimonov & Sornette, A stable and robust calibration scheme of the
            log-periodic power law model, 2013.

        Parameters
        ----------
        time : {array-like} of shape (n_samples)
            Sequence of time points, where `n_samples` is the number of samples.

        price : {array-like} of shape (n_samples)
            Sequence of price values, where `n_samples` is the number of samples.

        max_searches : int, default=25
            Number of attempts to fit the model. If eventually the fitting
            process is not successful model parameters are set to zeros.

        solver : str, default='Nelder-Mead'
            Solver used to minimize the non linear parameters of the model,
            see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
            for the available solvers. Suggested solvers are Nelder-Mead or
            Levenberg-Marquardt (least-squares).

        persist: bool, default=True
            If true, the estimated coefficients are stored as internal
            attributes.

        Returns
        -------
        coeffs : {array-like} of shape (10)
            Array with the parameters estimated for the model.

        """

        obs = np.array([time, price])

        t1 = time[0]
        t2 = time[-1]

        search_count = 0
        coeffs = np.zeros(10)

        # Start fitting process.
        while search_count < max_searches:

            # Set random guessing limits for non-linear parameters.
            guess_bounds = [

                # Boundaries for tc.
                (t2 - 0.2 * (t2 - t1), t2 + 0.2 * (t2 - t1)),

                # Boundaries for m.
                (0.0, 2.0),

                # Boundaries for ω.
                (1.0, 50.0)
            ]

            # Initial random seed for non linear parameters (tc, m, w).
            initial_guess = np.array([random.uniform(a[0], a[1]) for a in guess_bounds])

            # Increment search count on SVD convergence error, but raise all other exceptions.
            try:

                res = minimize(args = obs, fun = self.f_objective,
                    x0 = initial_guess, method = solver)

                if res.success:

                    tc = res.x[0]
                    m = res.x[1]
                    w = res.x[2]

                    # Solve for linear parameters.
                    a, b, c1, c2 = funcs.solve_matrix_equation(obs, tc, m, w)

                    # Get other parameters.
                    c = funcs.get_c(c1, c2)
                    osc = funcs.get_oscillations(w, tc, t1, t2)
                    damp = funcs.get_damping(m, w, b, c)

                    coef = np.array([tc, m, w, a, b, c, c1, c2, osc, damp])

                else:
                    raise ValueError('Optimizer did not exit successfully.')

                if persist is True:

                    # Store fitted parameters.
                    self.tc_ = tc
                    self.m_ = m
                    self.w_ = w
                    self.a_ = a
                    self.b_ = b
                    self.c_ = c
                    self.c1_ = c1
                    self.c2_ = c2
                    self.osc = osc
                    self.damp = damp

                coeffs = np.array([tc, m, w, a, b, c, c1, c2, osc, damp])

                # Exit from while loop.
                break

            except Exception as err:
                #print(str(err))
                # Start a new iteration.
                search_count += 1

        # If eventually not successful just return zero values.
        return coeffs

    def fit_nested(self, args):
        """Performs sequential fitting of multiple observation intervals in
        order to estimate LPPLS indicators."""

        # Unpack coordinates for fitting.
        obs_all, ii0, ii1, win_size, idx_stop, step_inner, max_searches = args

        # Select subset of observations under test.
        obs = obs_all[:, ii0 : ii1]
        t1 = obs[0][0]
        t2 = obs[0][-1]
        p2 = obs[1][-1]

        fittings = []

        # Now we fit multiple times on different observation samples.
        for jj in range(0, idx_stop, step_inner):

            # Prepare data for fitting.
            time = obs[0, jj : win_size]
            price = obs[1, jj : win_size]

            # Fit the model to the data and get back the params.
            tc, m, w, a, b, c, c1, c2, osc, damp = self.fit(time, price,
                max_searches = max_searches, persist = False)

            # Add fitting parameters.
            fittings.append([tc, m, w, a, b, c, c1, c2, osc, damp, time[0],
                time[-1]])

        return {'t1': t1, 't2': t2, 'p2': p2, 'fittings': fittings}

    def f_objective(self, x, *args):
        """Objective function to minimize for model calibration.

        For details about model calibration see:
            Filimonov & Sornette, A stable and robust calibration scheme of the
            log-periodic power law model, 2013.

        For minimization problem with scipy see:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html

        Args:
            x(np.ndarray):  1-damp array with shape (n,).
            args:           Tuple of the fixed parameters needed to completely specify the function.
        Returns:
            (float)
        """

        # Parse function variables and data observations.
        tc = x[0]
        m = x[1]
        w = x[2]
        obs = args[0]

        # Solve for linear parameters.
        a, b, c1, c2 = funcs.solve_matrix_equation(obs, tc, m, w)

        # Compute the sum of square residuals of function.
        f_values = [funcs.f_eval(t, tc, m, w, a, b, c1, c2) for t in obs[0]]
        errs = np.subtract(f_values, obs[1])
        errs = np.power(errs, 2)

        return np.sum(errs)

    def __call__(self, time):
        """Evaluates the function at a given time point."""

        return funcs.f_eval(time, self.tc_, self.m_, self.w_, self.a_, self.b_,
            self.c1_, self.c2_)

    def estimate_indicators(self, time_in, price_in, filters = None,
        parallel = True, n_workers = 4, win_size = 120, win_margin = 30,
        step_outer = 1, step_inner = 10, max_searches = MAX_SEARCHES):
        """
        Computes indicators of confidence for positive and negative bubbles.

        For each interval ending with `t2`, confidence values are estimated as the
        ratio between the number of windows where fitting parameters match
        filtering conditions (for both positive and negative bubbles) and the
        total number of windows considered for that `t2`.

        Parameters
        ----------
        time : {array-like} of shape (n_samples)
            Array of time points, where `n_samples` is the number of
            observations.

        price : {array-like} of shape (n_samples)
            Array of price values, where `n_samples` is the number of
            observations.

        filters : dict,default=None
            If provided, a dictionary with filtering conditions for m, w, osc
            and damp parameters. If not provided default values are used for
            estimation.

        parallel: bool, default=True
            If True, multiple fittings are executed in parallel with a
            multiprocessing pool.

        n_workers: int, default=4
            Number of workers to use for parallel execution. If parallel=False
            it is ignored.

        win_size : int, default=120
            Size of window used for nested fitting of price observations.

        win_margin : int, default=30
            Size of margin used for the moving window used in nested fitting.

        step_outer : int,default=1
            Step used for selecting time/price windows over the observations.

        step_inner : int,default=10
            Step used for the estimation of confidence values within a time
            window.

        max_searches : int,default=25
            Same as in `fit` method.

        Returns
        -------
        time : {array-like} of shape (n_samples - win_size + 1)
            Array of time points for which confidence values have been
            estimated, where `n_samples` is the number of
            observations.

        price : {array-like} of shape (n_samples - win_size + 1)
            Array of price points for which confidence values have been
            estimated, where `n_samples` is the number of
            observations.

        pos_confidences : {array-like} of shape (n_samples - win_size + 1)
            Array of confidence values for positive bubbles, where `n_samples`
            is the number of observations.

        neg_confidences : {array-like} of shape (n_samples - win_size + 1)
            Array of confidence values for negative bubbles, where `n_samples`
            is the number of observations.

        """

        obs = np.array([time_in, price_in])
        n_obs = obs.shape[1]

        # Safe margin for observation windows.
        idx_stop = win_size - win_margin

        # Prepare parameters for sequence of fitting.
        fitting_args = []
        for ii in range(0, n_obs - win_size + 1, step_outer):
            fitting_args.append((obs, ii, ii + win_size, win_size, idx_stop,
                step_inner, max_searches))

        n_fittings = len(fitting_args)

        if parallel is True:
            # Start multiprocessing pool for parallel fitting.
            with Pool(processes = n_workers) as pool:
                nested_fits = list(tqdm(pool.imap(self.fit_nested, fitting_args),
                    total = n_fittings))

        else:
            # No multiprocessing, so start sequential fitting.
            nested_fits = []
            for ii in tqdm(range(n_fittings)):
                nested_fits.append(self.fit_nested(fitting_args[ii]))

        # Set filters for valid values of parameters.
        if filters is None:

            # Default filters.
            m_min, m_max = 0.01, 1.2
            w_min, w_max = 2.0, 25.0
            osc_min = 2.5
            damp_min = 0.8

        else:

            # Try to parse user parameters.
            m_min = filters['m_min']
            m_max = filters['m_max']
            w_min = filters['w_min']
            w_max = filters['w_max']
            osc_min = filters['osc_min']
            damp_min = filters['damp_min']

        # Init output time/price.
        price = np.empty(n_fittings)
        time = np.empty(n_fittings)

        # Init array of confidence values.
        pos_confidences = np.empty(n_fittings)
        neg_confidences = np.empty(n_fittings)

        for ii, res in enumerate(nested_fits):

            # Update time/price.
            time[ii] = res['t2']
            price[ii] = res['p2']

            # Init counters.
            n_pos_candidates = 0
            n_neg_candidates = 0
            n_pos = 0
            n_neg = 0

            for fitting in res['fittings']:

                # Parse fitted parameters.
                tc = fitting[0]
                m = fitting[1]
                w = fitting[2]
                b = fitting[4]
                c = fitting[5]
                osc = fitting[8]
                damp = fitting[9]
                t1 = fitting[10]
                t2 = fitting[11]

                if b == 0 or c == 0:
                    osc = np.inf

                # Check for filtering conditions.
                tc_in_range = t2 - 0.05 * (t2 - t1) <= tc <=  t2 + 0.1 * (t2 - t1)
                m_in_range = m_min <= m <= m_max
                w_in_range = w_min <= w <= w_max
                osc_in_range = osc > osc_min
                damp_in_range = damp > damp_min

                # Mark whether filtering conditions are matched or not.
                if all((tc_in_range, m_in_range, w_in_range, osc_in_range, damp_in_range)):
                    is_candidate = True
                else:
                    is_candidate = False

                # Value of b determines the nature of the bubble.
                if b < 0:
                    n_pos += 1
                    if is_candidate:
                        n_pos_candidates += 1
                elif b > 0:
                    n_neg += 1
                    if is_candidate:
                        n_neg_candidates += 1

            # Compute confidence value for positive bubbles.
            if n_pos > 0:
                pos_confidence = n_pos_candidates / n_pos
            else:
                pos_confidence = 0

            # Compute confidence value for negative bubbles.
            if n_neg > 0:
                neg_confidence = n_neg_candidates / n_neg
            else:
                neg_confidence = 0

            # Add current confidence value.
            pos_confidences[ii] = pos_confidence
            neg_confidences[ii] = neg_confidence

        return time, price, pos_confidences, neg_confidences
