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
    if n == 0:
        raise ValueError(":( n should be an integer greater than zero")
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

    simpson_relerr = (simpsons_err / accepted) * 100
    gauss_relerr = (gauss_err / accepted) * 100
    
    print(f"{accepted=:.7f}")
    print("="*62)
    print(f"|| {'Method':^11} || {'Value':^11} || {'Abs Error':^11} || {'Rel Error':^11} ||")
    print("="*62)
    print(f"|| {'Simpsons':^11} || {simpsons:^11.7f} || {simpsons_err:^11.7f} || {simpson_relerr:^11.7f} ||")
    print(f"|| {'Gauss n4':^11} || {gauss:^11.7f} || {gauss_err:^11.7f} || {gauss_relerr:^11.7f} ||")
    print(f"{'My Gauss Method':=^62}")    
    for i in range(1, 8):
        cooler_gauss = cooler_gauss_quad(f, lower_bound, upper_bound, i)
        cooler_err = abs(accepted - cooler_gauss)
        cooler_relerr = (cooler_err / accepted) * 100
        print(f"|| {f'Gauss n{i}':^11} || {cooler_gauss:^11.7f} || {cooler_err:^11.7f} || {cooler_relerr:^11.7f} ||")
    print("="*62)

    #Both gauss methods got closer to the true solution than the composite simpsons method.
    #The gauss method with the n4 values already given is definitely the fastest computationally, but simpsons is faster than the general gauss
    #I prefer the general gauss since it finds a more precise solution and you can control n. I think the slight computational difference is fine.
    #To improve my approximation I could increase n for the gauss and or improve how the general gauss is approximating the roots of the legendre polynomial
