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

    Return two indices iA and iB such that matched rows are A[iA] and B[iB]."""

    # Arrange the A points along a new dimension
    A1 = np.expand_dims(A,1)

    # NA by NB mem logical where all elements match
    b = (A1==B).all(axis=-1)

    # A index is True where it matches any of the B points
    ind_A = b.any(axis=1)
    # Vice versa for B indec
    ind_B = b.any(axis=0)

    return ind_A, ind_B

if __name__=="__main__":

    n_short = 20
    M = 2
    N = 2
    diversify = 10
    intensify = 20
    restart = 50
    n_pattern = 2

    x0 = np.atleast_2d([0.5,2.])
    y0 = objective(x0)

    dx = np.array([.1,.5])
    dx_tol = dx/64.

    # Short term memory only needs to store input x
    X_short = np.nan*np.ones((n_short, M))
    X_short[0,:] = x0

    # Medium and long memories need x and y
    XY_med = (x0, y0)
    XY_long = (x0, y0)
    XY_int = (x0, y0)

    # Initialise local index
    i_local = 0

    # Main loop until step sizes smaller than a tolerance
    niter = 0
    maxiter = np.Inf
    while np.any( dx > dx_tol) and niter < maxiter:

        niter += 1

        # Generate candidate moves
        X = hj_move(x0,dx)

        # Filter by input constraints
        X = X[constrain_input(X)]

        # Filter against short term memory
        X = X[~find_rows(X,X_short)[0]]

        # Re-use previous objectives from long-term mem if possible
        Y = np.nan*np.ones((X.shape[0], N))

        # Indexes for candidate points 
        ind_X, ind_X_long = find_rows(X, XY_long[0])

        # Evaluate objective for unseen points
        Y[ind_X] = XY_long[1][ind_X_long]
        Y[~ind_X] = objective(X[~ind_X])

        # Put new results into long-term memory
        XY_long = add_long(X, Y, *XY_long)

        # Put Pareto-equivalent results into medium-term memory
        # Flag true if we sucessfully added a point
        XY_med, flag = add_med(X, Y, XY_med)

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
            XY_int, _ = add_med(X[i_dom[1:]], Y[i_dom[1:]], XY_int)
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
            # Random selection from long-term memory
            x1 = sample_mem(XY_long)
        elif i_local == intensify:
            # print('Intensifying')
            # Random selection from intesification memory
            x1 = sample_mem(XY_int)
        elif i_local == restart:
            # print('Restarting')
            # Reduce step sizes and randomly select from medium-term
            dx = dx/2.
            x1 = sample_mem(XY_med)
            i_local = 0

        # Add chosen point to short-term list (tabu)
        X_short = add_short(X_short, x1)

        # Update current point before next iteration
        x0 = x1


    print(XY_med[0].shape)
    print(XY_long[0].shape)
    X_long_unique = np.unique(XY_long[0],axis=0)
    print(X_long_unique.shape)
    f, a = plt.subplots()
    a.plot(*XY_long[1].T,'.')
    a.plot(*XY_med[1].T,'o')
    a.set_xlim((0.1,1.))
    a.set_ylim((0.,10.))
    plt.show()
