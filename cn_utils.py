import numpy as np
import math
import sympy as sp
#import jax
import plotly.graph_objects as go

x, y, z, a = sp.symbols('x y z a')

def dec2bin(x):
    """
    Takes a number is the decimal format a return the number in binary format as a string
    """
    binary = ''
    while x > 1:
        binary = str(x%2) + binary
        x = x//2
    return str(x) + binary

def bin2dec(x):
  """
  x: Takes a binary number in the string format
  """
  res = 0
  l = list(x)
  for idx, item in enumerate(l):
    res += int(item)* 2**(len(l)-idx - 1)

  return res


# Taylor expansion of a funtion
def taylor_expansion(func, x0, order):
    """
    func: a sympy univariable function with x as its argument. 
    order: the order of expansion
    -----
    returns the expansion of the `func` around the point `a` at the order `order` 
    """
    series = 0
    for i in range(order+1):
        series += (func.diff(x, i).subs(x,x0)/sp.factorial(i))*(x-x0)**i
    
    return series

# Bisection method
def bisection(func, xl, xu, tol=1e-3, max_iter=100):
    """
    func: a usual or lambda function
    xl: lower bound
    xu: upper bound
    tolerance: desired erro for termination
    max_iter: maximum iteration before stop
    """
    assert func(xu) * func(xl) < 0, 'func(xu) * func(xl) must be a negative value'

    iter = 0
    error = abs((xu-xl)/xu) * 100
    xr = (xl + xu)/2 + 0.1 * (xl + xu)/2

    while error > tol and iter < max_iter:
        xr_old = xr
        xr = (xl + xu)/2
        
        f_xr = func(xr)
        f_xl = func(xl)

        if f_xl * f_xr < 0:
            xu = xr
        elif f_xl * f_xr > 0:
            xl = xr
        else:
            return xr
            
        error = abs((xr-xr_old)/xr) * 100
        #print('xr', xr, 'error', error)

        iter += 1

    return xr, error

# Modified False Position

def false_position(func, xl, xu, tol=1e-3, max_iter=100):
    """
    func: a usual or lambda function
    xl: lower bound
    xu: upper bound
    tol: desired erro for termination
    max_iter: maximum iteration before stop
    """
    iteration = 0
    iu= il = 0
    error = abs((xu-xl)/xu) * 100
    xr = (xl + xu)/2 + 0.1 * (xl + xu)/2
    f_xu = func(xu)
    f_xl = func(xl)
    while error > tol and iteration < max_iter:
        
        
        xr_old = xr
        xr = xu - (f_xu*(xl-xu))/(f_xl - f_xu)
        f_xr = func(xr)

        test = f_xr * f_xl

        if test < 0:
            xu = xr
            f_xu = func(xu)
            iu = 0
            il = il + 1
            if il >= 2: 
                f_xl /= 2
        elif test > 0:
            xl = xr
            f_xl = func(xl)
            il = 0
            iu = iu + 1
            if iu >=2:
                f_xu /= 2
        else:
            return xr
        
        error = abs((xr-xr_old)/xr) * 100
        print('xr', xr, 'error', error)

        iteration += 1
    
    return xr, error

# Iteration of fix point
def fix_point(func, x0, tol=1e-3, max_iter=100):
    """
    func: a usual or lambda function
    x0: initial point
    tol: desired erro for termination
    max_iter: maximum iteration before stop
    """
    x_i = x0
    iteration = 0
    error = 100
    while error > tol and iteration <= max_iter:
        x_new = func(x_i)
        error = abs((x_new - x_i)/x_new) * 100
        #print(f'x_new é {x_new} e o erro é {abs(error)}')
        x_i = x_new
        iteration += 1
    
    return x_new, error

# A modified version of Newton-Raphson

def newtonRaphson(func, dfunc, xl, xu, tol=1.0e-3, max_iter=100):
    """
    Finds a root of f(x) = 0 by combining the Newton-Raphson
    method with bisection. The root must be bracketed in (a,b).
    Calls user-supplied functions f(x) and its derivative df(x).

    func: a usual or lambda function
    dfunc: first derivative of the func
    xl: lower bound
    xu: upper bound
    tol: desired erro for termination
    max_iter: maximum iteration before stop
    """
    f_xl = func(xl)
    f_xu = func(xu)

    assert f_xl * f_xu < 0,  'Root is not bracketed'

    x = 0.5*(xl + xu)
    error = 100
    iteration = 0

    while error > tol and iteration <= max_iter:
        f_x = func(x)

        # Tighten the brackets on the root
        if f_xl * f_x < 0:
            xu = x
        else:
            xl = x

        # Try a Newton-Raphson step
        df_x = dfunc(x)
        # If division by zero, push x out of bounds
        try: 
            dx = -f_x/df_x
        except ZeroDivisionError: 
            dx = xu - xl

        x_new = x + dx
        # If the result is outside the brackets, use bisection
        if (xu - x_new)*(x_new - xl) < 0.0:
            dx = 0.5*(xu - xl)
            x_new = xl + dx
        # Check for convergence
        error = abs((x_new - x)/x_new )* 100

        x = x_new
        iteration += 1

    return x_new, error

# Modified secant method for calculating the root of a function
def modifiedSecant(func, x0, delta=1e-2, tol=1e-3, max_iter=100):
    """
    func: a usual or lambda function
    x0: initial point
    delta: step size for divided fifference derivative
    tol: desired erro for termination
    max_iter: maximum iteration before stop
    """
    iteration = 0
    error = 100

    while error > tol and iteration <= max_iter:
        f = func(x0)
        x_new = x0 - f*delta*x0/(func(x0+delta*x0) - f)
        error = (x_new - x0)/ x_new * 100
        #print(x_new, error)
        x0 = x_new  
    
    return x_new, error

# Calculating the derminent of a 2x2 or 3x3 matrix
def matrix_determinent(a):
    assert a.shape[0] == a.shape[1], 'Matrix must be quadratic'
    assert len(a) == 2 or len(a) == 3, 'The shape of matrix must be 2x2 or 3x3'

    if len(a) == 2:
        return a[0,0]*a[1,1] - a[0,1]*a[1,0]
    if len(a) == 3:
        return a[0,0]*matrix_determinent(a[1:,[1,2]]) - a[0,1]*matrix_determinent(a[1:,[0,2]])+\
            a[0,2]*matrix_determinent(a[1:,[0,1]])


# Calculating a system of three linear equations by Cramer technics
def cramer(a, b):
    assert a.shape[0] == a.shape[1], 'Matrix must be quadratic'
    assert a.shape[0] == b.shape[0], 'The dimension of the matrix must the same as the vector of constants'
    D = matrix_determinent(a)
    temp = a.copy()
    temp[:,0] = b
    x1 = matrix_determinent(temp)/D

    temp = a.copy()
    temp[:,1] = b
    x2 = matrix_determinent(temp)/D

    temp = a.copy()
    temp[:,2] = b
    x3 = matrix_determinent(temp)/D

    return x1, x2, x3

# A minimal gauss elimination for solving a linear system
def gauss_elimination_minimal(A, b):
    """
    A: a n-by-n array
    b: a 1d numpy array
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    assert A.shape[0] == A.shape[1], 'the matrix of coefficients must be squared'

    mat = np.concatenate([A, b.reshape(-1,1)], axis=1)
    for idx_i in range(0, len(mat)-1):
        for idx_j in range(idx_i+1, len(mat)):
            factor = mat[idx_j, idx_i] / mat[idx_i, idx_i] 
            mat[idx_j] = mat[idx_j] - factor * mat[idx_i]

    n = len(A)-1
    xs = np.zeros(len(A))
    xs[-1] = mat[n,n+1]/mat[n,n]

    for i in range(n-1, -1, -1):
        sum = mat[i, -1]
        for j in range(0, len(A)):
            sum -= mat[i, j]* xs[j]
        xs[i] = sum / mat[i,i]

    return xs

# LU decomposion using gauss elimination
def decompose(A):
    """
    A: a n-by-n numpy array
    """
    A = np.array(A, dtype=float)

    assert A.shape[0] == A.shape[1], 'the matrix of coefficients must be squared'

    L = np.eye(len(A))
    U = A.copy()

    n = len(A)
    for idx_i in range(0, n-1):
        for idx_j in range(idx_i+1, n):
            factor = U[idx_j, idx_i] / U[idx_i, idx_i] 
            U[idx_j] = U[idx_j] - factor * U[idx_i]
            L[idx_j, idx_i] = factor
    

    return L, U

# A minimal Gauss-Seidel for linear systems
def gauss_seidel(A, b, x0, tol=1e-5, max_iter=20, return_error=False):
    """
    A: a n-by-n numpy array
    b: a 1d numpy array
    x0: initial guess, the same dimension as `b`
    tol: tolerance
    max_iter: maximum iteration
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(A)
    assert len(x0) == n, 'the size of initial guess must be the same as the system'
    assert all([A[i,i] != 0 for i in range(n)]), 'There is a zero in the diagonal of the matrix'
    iteration = 0
    x_old = xs = np.array(x0, dtype=np.float64)
    error = np.ones(n)*100

    while max(np.abs(error)) > tol and iteration < max_iter:
        
        for i in range(n):
            xs[i] = (b[i] - sum(np.delete(xs, i) * np.delete(A[i, :], i)))/A[i, i]
            error[i] = np.abs((xs[i] - x_old[i])/ xs[i])*100
            x_old = xs.copy()
        
        iteration += 1
    
    if return_error:
        return xs, error
    else:
        return xs

# Chlesky decomposition for symmetric matrices
def cholesky(A):
    """
    A: a symmetric nd array
    Return: L and its transpose such that L.dot(L.T) = A
    """
    A = np.array(A, dtype=float)
    assert A.shape[0] == A.shape[1], 'The matrix must be square'
    assert (A.T == A).all(), 'The matrix must be symmetric'
    
    L = np.zeros(A.shape)

    for i in range(len(A)):
        for j in range(i+1):
            if i == j:
                L[i, j] = np.sqrt(A[i, j] - sum(L[i, :i]**2))
            else:
                L[i, j] = (A[i, j] - sum(L[j, :j] * L[i, :j])) / L[j,j]
    
    return L, L.T


# 1D Linear Regression
def linear_regression(x, y):
    """
    x, y: 1d numpy array or list.
    """
    assert len(x) == len(y), "x and y must have the same size"

    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    n = len(x)
    xy = sum(x*y)
    x_2 = sum(x**2)
    x_bar = x.sum()/n
    y_bar = y.sum()/n
    
    #a0: interseção, a1:inclinação
    a1 = (n * xy - x.sum() * y.sum())/(n * x_2 - x.sum()**2)
    a0 = y_bar - a1 * x_bar

    return a0, a1

# 1D polinomial Regression
def polinomial_regression(X, Y, m, xi=None):
    """This function does a m-degree polinomial regression using the procedure 
    in the section 17.4 page 479 on the Numerical Methods for Engineers, 
    by Chapra, seventh edition

    Args:
        X (numeric iterable): [the independent measurments] 
        Y ([numeric iterable]): [the dependent measurments]
        m ([int]): [the degree on the polinomial to adjust]
        xi ([a number of numeric iterable]): [the point(s) that must be inserted into the calculated polinomial]
    

    Raises:
        TypeError: [if the `xi` is string or non-numeric iterable]

    Returns:
        [float, ndarray]: [the value of the point(s) calculated by the adjusted polinomial]
    """
    assert m <= len(X) - 1, 'the degree of polinomial, n, needs n+1 points'
    assert len(X) == len(Y), 'X and Y must have the same size'

    x = sp.symbols('x', real=True)
    Y = np.array(Y, dtype=float).reshape(len(Y), -1)

    fs = [x**i for i in range(m+1)]
    fs_np = [sp.lambdify(x, f) for f in fs]

    Z = [[f(i) for i in X] for f in fs_np]
    Z = np.array(Z, dtype=float).T

    b = Z.T.dot(Y)
    A = Z.T.dot(Z)
    A_inv = np.linalg.pinv(A)
    coefs = A_inv.dot(b)

    if xi is None:
        return coefs
    elif isinstance(xi, (int, float, complex)) and not isinstance(xi, bool):
        return sum([coefs[i] * xi**i for i in range(m+1)])
    elif isinstance(xi, (list, tuple, np.ndarray)):
        return np.array([sum([coefs[i] * xx**i for i in range(m+1)]) for xx in xi]).flatten()
    else: 
        raise TypeError('xi is not a number of an iterable')


    

# Gauss-Newton algorithm for non-linear regression (non optimized implementation)
def gauss_newton(x, y, func, vars, params, A0, xi=None, tol=1e-5, max_iter=20):
    """
    x: A (n,m) array contains measured values of the independent variables. `n` is the number of measurments and 
    `m` is the number of independent variables.
    y: A (n, 1) array contains the measured values of the dependent variable. 
    func: a sympy function with symbolic variables
    vars: list. the independent variables of the model
    params: list. the parameters of the model to be adjusted
    A0: list or array. An intial guess to the parameters
    xi: a number or numeric iterable. The point(s) that must be inserted into the calculated polinomial


    return the coeficients of the model (func)

    Exemple:

    a_0, a_1, x = sp.symbols('a_0, a_1 x')
    func = a_0*(1 - sp.exp(-a_1*x))
    xx = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
    yy = np.array([0.28, 0.57, 0.68, 0.74, 0.79])

    gauss_newton(xx, yy, func, [x], [a_0, a_1], [1,1])
    or
    gauss_newton(xx, yy, func, [x], [a_0, a_1], [1,1], 2.2)
    or a
    gauss_newton(xx, yy, func, [x], [a_0, a_1], [1,1], [2, 2.1, 2.2])

    """
    A0 = np.array(A0, dtype=np.float64)
    x = np.array(x, dtype=float).reshape(-1,1)
    m = x.shape[1]
    

    assert len(vars) == m, 'The number of independent variables must be the same as the columns of data'
    assert len(params) == len(A0), 'The number of initial values must be the same as the number of parameters'

    # creating a list of derivatives
    
    iteration = 1
    error = np.ones(len(params))*100
    A = A_old = A0

    while max(error) > tol and iteration <= max_iter:
        param_val = {p:v for p,v in zip(params, A)}
        
        dfuncs = [func.diff(var).subs(param_val) for var in params]
        dfuncs_np = [sp.lambdify(vars, dfunc) for dfunc in dfuncs]

        Z = np.array([[df(*i) for df in dfuncs_np] for i in x])
        ZT = Z.T

        func_new = func.subs(param_val)
        func_np = sp.lambdify(vars, func_new)
        D = y - np.array([func_np(*i) for i in x])
        
        DeltaA = np.linalg.pinv(ZT.dot(Z)).dot(ZT.dot(D))
        A_old = A.copy()
        A += DeltaA

        error =  np.abs((A - A_old)/A)*100


        iteration +=1

    if xi is None:
        return A
    elif isinstance(xi, (int, float, list, tuple, np.ndarray)) and not isinstance(xi, bool):
        return func_np(xi)
    else: 
        raise TypeError('xi is not a number of an iterable')


# 1D Lagrange polinomial interpolation
def lagrange_interpolation(x, y, n, xi):
    """
    x: a list of independent variables
    y: a list of dependent variables
    n: degree of polinomial 
    xi: the point of interest to calculate f(xi)
    """

    assert len(x) == len(y), 'The same size'
    assert len(x) > n, 'for the n-degree interpolation n+1 points are needed'

    sum = 0
    for i in range(n+1):
        product = y[i]
        for j in range(n+1):
            if i != j:
                product *= (xi - x[j])/(x[i] - x[j])
        sum += product
    
    return sum

# 1D Newton interpolation
def newton_interpolation(x, y, n, xi, return_fdd=False, return_last=True):
    """
    x: a list of independent variables
    y: a list of dependent variables
    n: degree of polinomial 
    xi: the point of interest to calculate f(xi)
    return_fdd: if True will return the finite divided-diference terms
    return_last: if False will return all of the approximation of f(xi) with polinomials lower than `n`
    """

    assert len(x) == len(y), 'the same size'
    assert len(x) > n, 'for the n-degree interpolation n+1 points are needed'
    
    n += 1
    fdd = np.zeros((len(y), len(y)))
    fdd[:, 0] = y

    for j in range(1, n):
        for i in range(0, n-j):
            fdd[i, j] = (fdd[i+1, j-1] - fdd[i, j-1])/(x[i+j] - x[i])

    
    
    errors = np.zeros(n-1)
    ys = np.zeros(n)
    ys[0] = y[0]
    xterm = 1
    for i in range(1, n):
        xterm *= xi - x[i-1]
        ys[i] = (ys[i-1] + xterm * fdd[0, i])
        errors[i-1] = fdd[0, i] * xterm
    
    if return_last:
        return ys[-1], errors[-1]

    elif return_fdd:
        return ys, errors, fdd
    else:
        return ys, errors

# The first and second derivative of a univarible function
def derivative(func, xi, n=1, h=0.001, method='center'):
    """[The first and second derivative of a univarible function]

    Args:
        func ([function]): [the function to be derivated]
        xi ([int, float]): [the point to calculate the derivative]
        n (int, optional): [the order of derivation 1 or 2]. Defaults to 1.
        h (float, optional): [the step size]. Defaults to 0.001.
        method (str, optional): [the method of derivation: 'prog', 'reg', 'center']. Defaults to 'center'.

    Returns:
        [float]: [the derivative of the `func` at point `xi`]
    """

    if n == 1:
        if method == 'center':
            return (func(xi + h) - func(xi - h))/(2*h)
        elif method == 'prog':
            return (func(xi + h) - func(xi))/(h)
        elif method == 'reg':
            return (func(xi) - func(xi - h))/(h)
        else:
            print('Please choose one of the following methods: center, prog, reg')

    elif n == 2:
        if method == 'center':
            return (func(xi + h) - 2*func(xi) + func(xi - h))/(h**2)
        elif method == 'prog':
            return (func(xi + 2*h) - 2 * func(xi + h) + func(xi))/(h**2)
        elif method == 'reg':
            return (func(xi) - 2*func(xi - h) + func(xi - 2*h))/(h**2)
        else:
            print('Please choose one of the following methods: center, prog, reg')

# integration using multiple application of trapezoidal method
def trapezoidal(func, a, b, n=1):
    """integration using multiple application of trapezoidal method

    Args:
        func (function): the function to be integrated
        a (float): the inferior limit
        b (float): the superior limit
        n (int, optional): the number of intervals. Defaults to 2.

    Returns:
        float: the value of the integration
    """
    assert a < b, 'the inferior limit must be less than superior one'
    h = (b - a)/n
    xi = np.linspace(a, b, n+1)
    return h/2*(func(xi[0]) + 2 * sum(map(func, xi[1:-1])) + func(xi[-1]))


# integration of a uivariate function using multiple application of Simpson 1/3 method
def simpson13(func, a, b, n=2):
    """integration of a uivariate function using multiple application of Simpson 1/3 method

    Args:
        func (function): the function to be integrated
        a (float): the inferior limit
        b (float): the superior limit
        n (int, optional): the number of intervals. Defaults to 2.

    Returns:
        float: the value of the integration
    """

    assert a < b, "The inferior limit must be less than the superior one"
    assert n > 1, "The simpson 1/3 needs at least 3 points, n = points - 1"
    assert n % 2 == 0, "The simpson 1/3 works only for par number of segments "

    xi = np.linspace(a, b, n+1)
    return ((b - a)/(3*n))*(func(xi[0]) + 4 * sum(func(xi[1:-1:2])) \
           + 2 * sum(func(xi[2:-2:2])) + func(xi[-1]))


#integration of a function given four points using Simpson 3/8 method
def _simpson38(func, xi):
    """integration of a function given four points using Simpson 3/8 method

    Args:
        func (function): the function to be integrated
        xi (list, tuple or array): a list of four points to insert into the function for integration

    Returns:
        float: the value of the integration
    """

    assert len(xi) == 4,  "This version is designed only for 4 points integration"

    h = xi[1] - xi[0]
    return 3*h/8 *(func(xi[0]) + 3 * (func(xi[1]) + func(xi[2])) + func(xi[-1]))
    

# integration of a uivariate function using multiple application of Simpson 1/3 method
def simpson38(func, a, b, n=3):
    """integration of a uivariate function using multiple application of Simpson 1/3 method

    Args:
        func (function): the function to be integrated
        a (float): the inferior limit
        b (float): the superior limit
        n (int, optional): the number of intervals. Defaults to 2.

    Returns:
        float: the value of the integration
    """

    assert a < b, "The inferior limit must be less than the superior one"
    assert n > 2, "The simpson 3/8 needs at least 4 points, n = points - 1"
    assert n % 3 == 0, "The simpson 3/8 works only for the multiple of three segments "

    xi = np.linspace(a, b, n+1)
    not_multi_3 = []
    multi_3 = []
    for idx in range(1,n):
        if idx % 3 != 0:
            not_multi_3.append((idx))
        else:
            multi_3.append((idx))

    not_multi_3 = xi[not_multi_3]
    multi_3 = xi[multi_3]    
    return (3*(b - a)/(8*n))*(func(xi[0]) + 3 * sum(func(not_multi_3)) \
        + 2 * sum(func(multi_3)) + func(xi[-1]))

# integration using a mixture of Simpson 1/3 and 3/8
def integrate(func, a, b, n=2):
    """integration using a mixture of Simpson 1/3 and 3/8. 
    The preference is using the Simpson 1/3.

    Args:
        func (function): the function to be integrated
        a (float): the inferior limit
        b (float): the superior limit
        n (int, optional): the number of intervals. Defaults to 2.

    Returns:
        float: the value of the integration
    """
    assert a < b, 'the inferior limit must be less than the superior'
    
    xi = np.linspace(a, b, n+1)
    if n == 1:
        return trapezoidal(func, a, b)
    elif n % 2 == 0:
        return simpson13(func, a, b, n)
    elif n == 3:
        return _simpson38(func, xi)
    else:
        h = (b - a)/n
        return _simpson38(func, xi[-4:]) + simpson13(func, a, b-3*h, n-3 )
