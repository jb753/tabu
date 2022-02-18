"""Functions for multiobjective tabu search."""
import numpy as np
import matplotlib.pyplot as plt


def hj_move(x, dx):
    """Generate a set of Hooke and Jeeves moves about a point.

    For a design vector with M variables, return a 2MxM matrix, each variable
    having being perturbed by elementwise +/- dx."""
    return x + np.concatenate((np.diag(dx), np.diag(-dx)))


def objective(x):
    return np.stack((x[:, 0], (1.0 + x[:, 1]) / x[:, 0]), axis=1)


def constrain_input(x):
    return np.all(
        (
            x[:, 1] + 9.0 * x[:, 0] >= 6.0,
            -x[:, 1] + 9.0 * x[:, 0] >= 1.0,
            x[:, 0] >= 0.0,
            x[:, 0] <= 1.0,
            x[:, 1] >= 0.0,
            x[:, 1] <= 5.0,
        ),
        axis=0,
    )


def find_rows(A, B, atol=None):
    """Get matching rows in matrices A and B.

    Return:
        logical same shape as A, True where A is in B
        indices same shape as A, of the first found row in B for each A row."""

    # Arrange the A points along a new dimension
    A1 = np.expand_dims(A, 1)

    # NA by NB mem logical where all elements match
    if atol:
        b = np.isclose(A1, B, atol=atol).all(axis=-1)
    else:
        b = (A1 == B).all(axis=-1)

    # A index is True where it matches any of the B points
    ind_A = b.any(axis=1)

    # Use argmax to find first True along each row
    loc_B = np.argmax(b, axis=1)

    # Where there are no matches, override to a sentinel value -1
    loc_B[~ind_A] = -1

    return ind_A, loc_B


class Memory:
    def __init__(self, nx, ny, max_points):
        """Store a set of design vectors and their objective functions."""

        # Record inputs
        self.nx = nx
        self.ny = ny
        self.max_points = max_points

        # Initialise points counter
        self.npts = 0

        # Preallocate matrices for design vectors and objectives
        # Private because the user should not have to deal with empty slots
        self._X = np.empty((max_points, nx))
        self._Y = np.empty((max_points, ny))

    # Public read-only properties for X and Y
    @property
    def X(self):
        """The current set of design vectors."""
        return self._X[: self.npts, :]

    @property
    def Y(self):
        """The current set of objective functions."""
        return self._Y[: self.npts, :]

    def contains(self, Xtest):
        """Boolean index for each row in Xtest, True if x already in memory."""
        if self.npts:
            return find_rows(Xtest, self._X[: self.npts])[0]
        else:
            return np.zeros((Xtest.shape[0],), dtype=bool)

    def add(self, xa, ya=None):
        """Add a point to the memory."""
        xa = np.atleast_2d(xa)
        if ya is None:
            ya = xa
        else:
            ya = np.atleast_2d(ya)

        # Only add new points
        i_new = ~self.contains(xa)
        n_new = np.sum(i_new)
        xa = xa[i_new]
        ya = ya[i_new]

        # Roll downwards and overwrite
        self._X = np.roll(self._X, n_new, axis=0)
        self._X[:n_new, :] = xa
        self._Y = np.roll(self._Y, n_new, axis=0)
        self._Y[:n_new, :] = ya

        # Update points counter
        self.npts = np.min((self.max_points, self.npts + n_new))

    def lookup(self, Xtest):
        """Return objective function for design vector already in mem."""

        # Check that the requested points really are available
        if np.any(~self.contains(Xtest)):
            raise ValueError("The requested points have not been previously evaluated")

        return self._Y[: self.npts][find_rows(Xtest, self._X[: self.npts])[1]]

    def delete(self, ind_del):
        """Remove points at given indexes."""

        # Set up boolean mask for points to keep
        b = np.ones((self.npts,), dtype=bool)
        b[ind_del] = False
        n_keep = np.sum(b)

        # Reindex everything so that spaces appear at the end of memory
        self._X[:n_keep] = self._X[: self.npts][b]
        self._Y[:n_keep] = self._Y[: self.npts][b]

        # Update number of points
        self.npts = n_keep

    def update_front(self, X, Y):
        """Add or remove points to maintain a Pareto front."""
        Yopt = self._Y[: self.npts]

        # Arrange the test points along a new dimension
        Y1 = np.expand_dims(Y, 1)

        # False where an old point is dominated by a new point
        b_old = ~(Y1 < Yopt).all(axis=-1).any(axis=0)

        # False where a new point is dominated by an old point
        b_new = ~(Y1 >= Yopt).all(axis=-1).any(axis=1)

        # False where a new point is dominated by a new point
        b_self = ~(Y1 > Y).all(axis=-1).any(axis=1)

        # We only want new points that are non-dominated
        b_new_self = np.logical_and(b_new, b_self)

        # Delete old points that are now dominated by new points
        self.delete(~b_old)

        # Add new points
        self.add(X[b_new_self], Y[b_new_self])

        # Flag is true if we added new points
        flag = np.sum(b_new_self) > 0

        return flag

    def update_best(self, X, Y):
        """Add or remove points to keep the best N in memory."""

        X, Y = np.atleast_2d(X), np.atleast_2d(Y)

        # Join memory and test points into one matrix
        Yall = np.concatenate((self._Y[: self.npts],Y),axis=0)
        Xall = np.concatenate((self._X[: self.npts],X),axis=0)

        # Sort by objective, truncate to maximum number of points
        isort = np.argsort(Yall[:,0],axis=0)[:self.max_points]
        Xall, Yall = Xall[isort], Yall[isort]

        # Reassign to the memory
        self.npts = len(isort)
        self._X[: self.npts] = Xall
        self._Y[: self.npts] = Yall


    def generate_sparse(self, nregion):
        """Return a random design vector in a underexplored region."""

        # Loop over each variable
        xnew = np.empty((1, self.nx))
        for i in range(self.nx):

            # Bin the design variable
            hX, bX = np.histogram(self._X[: self.npts, i], nregion)

            # Random value in least-visited bin
            bin_min = hX.argmin()
            bnds = bX[bin_min : bin_min + 2]
            xnew[0, i] = np.random.uniform(*bnds)

        return xnew

    def sample_random(self):
        """Choose a random design point from the memory."""
        i_select = np.random.choice(self.npts, 1)
        return self._X[i_select], self._Y[i_select]

    def sample_sparse(self):
        """Choose a design point from sparse region of the memory."""

        # Arbitrarily bin on first design variable
        hX, bX = np.histogram(self._X[: self.npts, 0], 5)

        # Override count in empty bins so we do not pick them
        hX[hX==0] = hX.max()+1

        # Choose sparsest bin, breaking ties randomly
        i_bin = np.random.choice(np.flatnonzero(hX == hX.min()))

        # Logical indexes for chosen bin
        log_bin = np.logical_and(
                self._X[:self.npts,0] >= bX[i_bin] ,
                self._X[:self.npts,0] <= bX[i_bin+1]
                )
        # Choose randomly from sparsest bin
        i_select = np.atleast_1d(np.random.choice(np.flatnonzero(log_bin)))

        return self._X[i_select], self._Y[i_select]

    def clear(self):
        """Erase all points in memory."""
        self.npts = 0


class TabuSearch:
    def __init__(self, objective, constraint, nx, ny):
        """Maximise an objective function using Tabu search."""

        # Store objective and constraint functions
        self.objective = objective
        self.constraint = constraint

        # Default memory sizes
        self.n_short = 20
        self.n_med = 1000
        self.n_long = 2000
        self.nx = nx
        self.ny = ny

        # Default iteration counters
        self.i_diversify = 5
        self.i_intensify = 20
        self.i_restart = 50
        self.i_pattern = 2

        # Misc algorithm parameters
        self.x_regions = 2
        self.max_fevals = 2000
        self.fac_restart = 0.5
        self.fac_pattern = 2.0

        # Initialise counters
        self.fevals = 0

        # Initialise memories
        self.mem_short = Memory(nx, ny, self.n_short)
        self.mem_med = Memory(nx, ny, self.n_med)
        self.mem_long = Memory(nx, ny, self.n_long)
        self.mem_int = Memory(nx, ny, self.n_med)
        self.mem_all = (self.mem_short, self.mem_med, self.mem_long, self.mem_int)

    def clear_memories(self):
        """Erase all memories"""
        for mem in self.mem_all:
            mem.clear()

    def initial_guess(self, x0):
        """Reset memories, set current point, evaluate objective."""
        self.clear_memories()
        y0 = objective(x0)
        for mem in self.mem_all:
            mem.add(x0, y0)
        return y0

    def evaluate_moves(self, x0, dx):
        """From a given start point, evaluate permissible candidate moves."""

        # Generate candidate moves
        X = hj_move(x0, dx)

        # Filter by input constraints
        X = X[self.constraint(X)]

        # Filter against short term memory
        X = X[~self.mem_short.contains(X)]

        # Re-use previous objectives from long-term mem if possible
        Y = np.nan * np.ones((X.shape[0], self.ny))
        ind_seen = self.mem_long.contains(X)
        Y[ind_seen] = self.mem_long.lookup(X[ind_seen])

        # Evaluate objective for unseen points
        Y[~ind_seen] = self.objective(X[~ind_seen])

        # Increment function evaluation counter
        self.fevals += np.sum(~ind_seen)

        return X, Y

    def select_move(self, x0, y0, X, Y):
        """Choose next move given starting point and list of candidate moves."""

        # Categorise the candidates for next move with respect to current
        b_dom = (Y < y0).all(axis=1)
        b_non_dom = (Y > y0).all(axis=1)
        b_equiv = ~np.logical_and(b_dom, b_non_dom)

        # Convert to indices
        i_dom = np.where(b_dom)[0]
        i_non_dom = np.where(b_non_dom)[0]
        i_equiv = np.where(b_equiv)[0]

        # Choose the next point
        if len(i_dom) > 0:
            # If we have dominating points, randomly choose from them
            np.random.shuffle(i_dom)
            x1, y1 = X[i_dom[0]], Y[i_dom[0]]
            # Put spare dominating points into intensification memory
            if len(i_dom) > 1:
                self.mem_int.update_front(X[i_dom[1:]], Y[i_dom[1:]])
        elif len(i_equiv) > 0:
            # Randomly choose from equivalent points
            np.random.shuffle(i_equiv)
            x1, y1 = X[i_equiv[0]], Y[i_equiv[0]]
        elif len(i_non_dom) > 0:
            # Randomly choose from non-dominating points
            np.random.shuffle(i_non_dom)
            x1, y1 = X[i_non_dom[0]], Y[i_non_dom[0]]
        else:
            raise Exception("No valid points to pick next move from")

        # Keep in matrix form
        x1 = np.atleast_2d(x1)
        y1 = np.atleast_2d(y1)

        return x1, y1

    def pattern_move(self, x0, y0, x1, y1):
        """If this move is in a good direction, increase move length."""
        x1a = x0 + self.fac_pattern * (x1 - x0)
        y1a = self.objective(x1a)
        if (y1a < y1).all():
            return x1a
        else:
            return x1

    def search(self, x0, dx, dx_min):
        """Perform a search with given intial point and step size."""

        y0 = ts.initial_guess(x0)

        i = 0
        while self.fevals < self.max_fevals and np.any(dx > dx_min):

            # Evaluate objective for all permissible candidate moves
            X, Y = ts.evaluate_moves(x0, dx)

            # Put new results into long-term memory
            self.mem_long.add(X, Y)

            # Put Pareto-equivalent results into medium-term memory
            # Flag true if we sucessfully added a point
            flag = self.mem_med.update_front(X, Y)

            # If we did not add to medium memory, increment local search counter
            if flag:
                i = 0
            else:
                i += 1

            # Choose next point based on local search counter
            if i == self.i_restart:
                # Reduce step sizes and randomly select from medium-term
                dx = dx * self.fac_restart
                x1, y1 = self.mem_med.sample_sparse()
                i = 0
            elif i == self.i_intensify:
                x1, y1 = self.mem_int.sample_sparse()
            elif i == self.i_diversify or X.shape[0] == 0:
                # Generate a new point in sparse design region,
                # If we have reached i_diversify or all moves are tabu
                x1 = self.mem_long.generate_sparse(self.x_regions)
                y1 = self.objective(x1)
            else:
                # Normally, choose the best candidate move
                x1, y1 = ts.select_move(x0, y0, X, Y)
                # Check for a pattern move every i_pattern steps
                if np.mod(i, self.i_pattern):
                    x1 = ts.pattern_move(x0, y0, x1, y1)

            # Add chosen point to short-term list (tabu)
            self.mem_short.add(x1)

            # Update current point before next iteration
            x0, y0 = x1, y1


if __name__ == "__main__":

    ts = TabuSearch(objective, constrain_input, 1, 1)

    x0 = 1.
    y0 = 2.

    ts.mem_med.add(1.,4.)
    ts.mem_med.add(2.,3.)
    ts.mem_med.add(3.,2.)
    ts.mem_med.add(5.,2.)
    ts.mem_med.update_best([[2.],[3.],[1.]],[[3.5],[5.],[1.]])


    # x0 = np.atleast_2d([0.5, 2.0])
    # dx = np.array([0.1, 0.5])

    # ts.search(x0, dx, dx / 64.0)

#     print("%d evals" % ts.fevals)
#     print(ts.mem_med.npts)
#     print(ts.mem_long.npts)
#     f, a = plt.subplots()
#     a.plot(ts.mem_long.Y[:, 0], ts.mem_long.Y[:, 1], ".")
#     a.plot(ts.mem_med.Y[:, 0], ts.mem_med.Y[:, 1], "o")
#     a.plot(ts.mem_int.Y[:, 0], ts.mem_int.Y[:, 1], "x")
#     a.set_xlim((0.1, 1.0))
#     a.set_ylim((0.0, 10.0))
#     plt.show()
