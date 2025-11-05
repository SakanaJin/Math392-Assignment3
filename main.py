from scipy.integrate import quad
from typing import Callable
import math

def f(x: float) -> float:
    return 1 / math.sqrt((x**2) + 1)

n: int = 4
lower_bound: int = 0
upper_bound: int = 2

accepted, err = quad(f, lower_bound, upper_bound)

def linear_transform(f: Callable[[float], float], a: float, b: float) -> Callable[[float], float]:
    def func(u: float) -> float:
        return f(((b-a) * u + (b+a))/2)*(b-a)/2
    return func

def legendre_polyfind(n: int, x: float) -> tuple[float, float]:
    '''Returns the nth and n-1th legendre polynomial evaluated at x'''
    if n == 0:
        return 1.0, 0.0 
    P0 = 1.0
    P1 = x
    for i in range(1, n):
        P2 = ((2 * i + 1) * x * P1 - (i * P0)) / (i+1)
        P0, P1 = P1, P2
    return P1, P0

def simpson(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    h: float = (b-a)/n
    return (h/3)*(f(a) + (2 * sum([f(a + (2 * i * h)) for i in range(1, (int)(n/2))])) + (4 * sum([f(a + ((2 * i) - 1) * h) for i in range(1, (int)(n/2) + 1)])) + f(b))

def gauss_quad(f: Callable[[float], float], a: float, b: float) -> float:
    c1 = 0.3478548451
    c2 = 0.6521451549
    x1 = 0.8611363116
    x2 = 0.3399810436
    if(a != -1.0 or b != 1.0):
      f = linear_transform(f, a, b)
    return c1 * f(-x1) + c2 * f(-x2) + c2 * f(x2) + c1 * f(x1)

def cooler_gauss_quad(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    MAXITER = 10
    TOL = 1e-14
    roots = []
    ci = []
    for i in range(1, n+1):
        x = math.cos(math.pi * (4*i - 1)/(4*n + 2)) #initial guess at the ith root pure black magic I just found this online but it makes sense that it's cos if you look at the graphs of Legendre polynomials
        for _ in range(MAXITER): #we get the nth Legendre polynomial and the n-1 for the derivative then do some newton method
            P1, P0 = legendre_polyfind(n, x)
            P1prime = n * (x*P1 - P0)/(x**2 - 1.0)
            oldx = x
            x -= P1 / P1prime
            if abs(x - oldx) < TOL:
                break
        roots.append(x)
        ci.append(2/((1 - x**2) * P1prime**2))
    if(a != -1.0 or b != 1.0):
        f = linear_transform(f, a, b)
    return sum([c * f(x) for c, x in zip(ci, roots)])

if __name__ == "__main__":
    simpsons = simpson(f, lower_bound, upper_bound, n)
    gauss = gauss_quad(f, lower_bound, upper_bound)

    simpsons_err = abs(accepted - simpsons)
    gauss_err = abs(accepted - gauss)

    print(f"{accepted=:.7f}")
    print(f"{simpsons=:.7f}\t{simpsons_err=:.7f}")
    print(f"{gauss=:.7f}\t{gauss_err=:.7f}")

    for i in range(1, 8):
        cooler_gauss = cooler_gauss_quad(f, lower_bound, upper_bound, i)
        cooler_err = abs(accepted - cooler_gauss)
        print(f"{cooler_gauss=:.7f}\t{cooler_err=:.7f}")
