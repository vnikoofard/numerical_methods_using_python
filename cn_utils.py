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
        print('xr', xr, 'error', error)

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
    tolerance: desired erro for termination
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

# Segure version of Newton-Raphson

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