"""Functions for multiobjective tabu search."""
import numpy as np
import matplotlib.pyplot as plt

def hj_move(x, dx):
    """Generate a set of Hooke and Jeeves moves about a point.

    For a design vector with M variables, return a 2MxM matrix, each variable
    having being perturbed by elementwise +/- dx."""
    return x + np.concatenate((np.diag(dx),np.diag(-dx)))

def objective(x):
    return np.stack((x[:,0], (1.+x[:,1])/x[:,0]),axis=1)

def constrain_input(x):
    return np.all(
        (
            x[:,1]+9.*x[:,0]>=6.,
            -x[:,1]+9.*x[:,0]>=1.,
            x[:,0]>=0.,
            x[:,0]<=1.0,
            x[:,1]>=0.,
            x[:,1]<=5.,
        )
        ,axis=0)

def add_short(X,x):
    X = np.roll(X,1,axis=0)
    X[0,:] = x
    return X

def add_med(X, Y, XYopt):
    # Split args
    Xopt, Yopt = XYopt

    # Arrange the test points along a new dimension
    Y1 = np.expand_dims(Y,1)

    # False where an old point is dominated by a new point
    b_old = ~(Y1<Yopt).all(axis=-1).any(axis=0)

    # False where a new point is dominated by an old point
    b_new = ~(Y1>=Yopt).all(axis=-1).any(axis=1)

    # False where a new point is dominated by a new point
    b_self = ~(Y1>Y).all(axis=-1).any(axis=1)

    # We only want new points that are non-dominated
    b_new_self = np.logical_and(b_new,b_self)

    # Return non-dominated points from both sets
    Ym = np.concatenate((Y[b_new_self],Yopt[b_old]))
    Xm = np.concatenate((X[b_new_self],Xopt[b_old]))

    # Flag is true if we added new point
    flag = np.sum(b_new_self)>0

    return (Xm, Ym), flag

def add_long(X, Y, X_long, Y_long):
    # Only add unique points
    i, il = find_rows(X, X_long)
    X = X[~i]
    Y = Y[~i]

    # Return combination
    X_new = np.append(X_long, X, axis=0)
    Y_new = np.append(Y_long, Y, axis=0)

    return X_new, Y_new

def sample_mem(XY):
    X = XY[0]
    return X[np.random.choice(X.shape[0],1)]

def find_rows(A,B):
    """Get matching rows in matrices A and B.

    Return:
        logical same shape as A, True where A is in B
        indices same shape as A, of the first found row in B for each A row."""

    # Arrange the A points along a new dimension
    A1 = np.expand_dims(A,1)

    # NA by NB mem logical where all elements match
    b = (A1==B).all(axis=-1)

    # Lay out index arrays
    jA, jB = np.meshgrid(range(A.shape[0]),range(B.shape[0]),indexing='ij')

    # A index is True where it matches any of the B points
    ind_A = b.any(axis=1)

    # Use argmax to find first True along each row
    loc_B = np.argmax(b,axis=1)

    # Where there are no matches, override to a sentinel value -1
    loc_B[~ind_A] = -1

    return ind_A, loc_B

def sample_long(XY, nregion):
    """Return a point in under-explored region of design space."""

    X = XY[0]

    # Loop over each variable
    xnew = np.empty((1,M))
    for m in range(M):

        # Bin the design variables
        hX, bX = np.histogram(X[:,m], nregion)

        # Random value in least-visited bin
        imin = hX.argmin()
        bnds = bX[imin:imin+2]
        xnew[0,m] = np.random.uniform(*bnds)

    return xnew

def sample_front(XY, nregion):
    """Return a point in under-explored region of Pareto front."""

    Y = XY[1]
    X = XY[0]

    # Arbitrarily bin using the first objective
    hY, bY = np.histogram(Y[:,0], nregion)

    # Select least-visited bin
    imin = hY.argmin()
    bnds = bY[imin:imin+2]

    # Select a random point in this bin
    Xb = X[np.logical_and( Y[:,0] > bnds[0] , Y[:,0] <= bnds[1])]
    xnew = Xb[np.random.choice(Xb.shape[0])]

    return xnew

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
        self.X = np.empty((max_points, nx))
        self.Y = np.empty((max_points, ny))

    def add(self, xa, ya=None):
        """Add a point to the memory."""
        if ya is None:
            ya=xa

        # Only add new points
        i_new = ~self.contains(xa)
        n_new = np.sum(i_new)
        xa = xa[i_new]
        ya = ya[i_new]

        # Roll downwards and overwrite
        self.X = np.roll(self.X,n_new,axis=0)
        self.X[:n_new,:] = xa
        self.Y = np.roll(self.Y,n_new,axis=0)
        self.Y[:n_new,:] = ya

        # Update points counter
        self.npts = np.min((self.max_points, self.npts+n_new))

    def contains(self, Xtest):
        """Boolean index for each row in Xtest, True if x already in memory."""
        if self.npts:
            return find_rows(Xtest, self.X[:self.npts])[0]
        else:
            return np.zeros((Xtest.shape[0],),dtype=bool)

    def lookup(self, Xtest):
        """Return objective function for design vector already in mem."""

        # Check that the requested points really are available
        if np.any(~self.contains(Xtest)):
            raise ValueError('The requested points have not been previously evaluated')

        return self.Y[:self.npts][find_rows(Xtest, self.X[:self.npts])[1]]

    def delete(self, ind_del):
        """Remove points at given indexes."""

        # Set up boolean mask for points to keep
        b = np.ones((self.npts,),dtype=bool)
        b[ind_del] = False
        n_keep = np.sum(b)

        # Reindex everything so that spaces appear at the end of memory
        self.X[:n_keep] = self.X[:self.npts][b]
        self.Y[:n_keep] = self.Y[:self.npts][b]

        # Update number of points
        self.npts = n_keep


    def update_front(self, X, Y):
        """Add or remove test points to maintain a Pareto front."""
        Yopt = self.Y[:self.npts]

        # Arrange the test points along a new dimension
        Y1 = np.expand_dims(Y,1)

        # False where an old point is dominated by a new point
        b_old = ~(Y1<Yopt).all(axis=-1).any(axis=0)

        # False where a new point is dominated by an old point
        b_new = ~(Y1>=Yopt).all(axis=-1).any(axis=1)

        # False where a new point is dominated by a new point
        b_self = ~(Y1>Y).all(axis=-1).any(axis=1)

        # We only want new points that are non-dominated
        b_new_self = np.logical_and(b_new,b_self)

        # Delete old points that are now dominated by new points
        self.delete(~b_old)

        # Add new points
        self.add(X[b_new_self], Y[b_new_self])

        # Flag is true if we added new points
        flag = np.sum(b_new_self)>0

        return flag


    def generate_sparse(self, nregion):
        """Return a random design vector in a underexplored region."""

        # Loop over each variable
        xnew = np.empty((1,self.nx))
        for i in range(self.nx):

            # Bin the design variable
            hX, bX = np.histogram(self.X[:self.npts,i], nregion)

            # Random value in least-visited bin
            bin_min = hX.argmin()
            bnds = bX[bin_min:bin_min+2]
            xnew[0,i] = np.random.uniform(*bnds)

        return xnew


    def sample_random(self):
        """Choose a random design point from the memory."""
        return self.X[np.random.choice(self.npts,1)]

if __name__=="__main__":

    n_short = 20
    n_med = 1000
    n_long = 2000

    n_region = 2
    nx = 2
    ny = 2
    diversify = 10
    intensify = 20
    restart = 50
    n_pattern = 2

    x0 = np.atleast_2d([0.5,2.])
    y0 = objective(x0)

    dx = np.array([.1,.5])
    dx_tol = dx/64.

    # Short term memory only needs to store input x
    mem_short = Memory(nx, ny, n_short)
    mem_med = Memory(nx, ny, n_med)
    mem_long = Memory(nx, ny, n_long)
    mem_int = Memory(nx, ny, n_long)

    for mem in (mem_short, mem_long, mem_int):
        mem.add(x0, y0)

    # Initialise local index
    i_local = 0

    # Main loop until step sizes smaller than a tolerance
    niter = 0
    maxiter = np.Inf
    fevals = 0
    while np.any( dx > dx_tol):

        niter += 1

        # Generate candidate moves
        X = hj_move(x0,dx)

        # Filter by input constraints
        X = X[constrain_input(X)]

        # Filter against short term memory
        X = X[~mem_short.contains(X)]

        # Re-use previous objectives from long-term mem if possible
        Y = np.nan*np.ones((X.shape[0], ny))

        # Indexes for candidate points 
        ind_seen = mem_long.contains(X)

        # Evaluate objective for unseen points
        Y[ind_seen] = mem_long.lookup(X[ind_seen])
        Y[~ind_seen] = objective(X[~ind_seen])
        fevals += np.sum(~ind_seen)

        # Put new results into long-term memory
        mem_long.add(X, Y)

        # Put Pareto-equivalent results into medium-term memory
        # Flag true if we sucessfully added a point
        flag = mem_med.update_front(X, Y)

        # If we did not add to medium memory, increment local search counter
        if flag:
            i_local = 0
        else:
            i_local += 1

        # Categorise the candidates for next move with respect to current
        b_dom = (Y<y0).all(axis=1)
        b_non_dom= (Y>y0).all(axis=1)
        b_equiv = ~np.logical_and(b_dom, b_non_dom)

        # Convert to indices
        i_dom = np.where(b_dom)[0]
        i_non_dom = np.where(b_non_dom)[0]
        i_equiv = np.where(b_equiv)[0]

        # Choose the next point
        if len(i_dom)==1:
            x1 = X[i_dom]
        elif len(i_dom)>1:
            # Randomly choose from multiple dominating points
            np.random.shuffle(i_dom)
            x1 = X[i_dom[0]]
            y1 = Y[i_dom[0]]
            # Put spare points into intensification memory
            mem_int.add(X[i_dom[1:]], Y[i_dom[1:]])
        elif len(i_equiv)>0:
            # Randomly choose from equivalent points
            np.random.shuffle(i_equiv)
            x1 = X[i_equiv[0]]
            y1 = Y[i_equiv[0]]
        elif len(i_non_dom)>0:
            # Randomly choose from non-dominating points
            np.random.shuffle(i_non_dom)
            x1 = X[i_non_dom[0]]
            y1 = Y[i_non_dom[0]]

        x1 = np.atleast_2d(x1)
        y1 = np.atleast_2d(y1)

        # Test for pattern move
        if np.mod(niter,n_pattern):
            x1a = x0 + 2.*(x1-x0)
            y1a = objective(x1a)
            if (y1a<y1).all():
                # print('Pattern move')
                x1 = x1a

        # Choose next point based on local search counter
        if i_local == diversify:
            # print('Diversifying')
            x1 = mem_long.generate_sparse(n_region)
        elif i_local == intensify:
            # print('Intensifying')
            x1 = mem_int.sample_random()
        elif i_local == restart:
            # print('Restarting')
            # Reduce step sizes and randomly select from medium-term
            dx = dx/2.
            x1 = mem_med.sample_random()
            i_local = 0

        # Add chosen point to short-term list (tabu)
        mem_short.add(x1)

        # Update current point before next iteration
        x0 = x1


    print('%d evals' % fevals)
    print(mem_med.npts)
    print(mem_long.npts)
    f, a = plt.subplots()
    a.plot(mem_long.Y[:mem_long.npts,0],mem_long.Y[:mem_long.npts,1],'.')
    a.plot(mem_med.Y[:mem_med.npts,0],mem_med.Y[:mem_med.npts,1],'o')
    a.set_xlim((0.1,1.))
    a.set_ylim((0.,10.))
    plt.show()
