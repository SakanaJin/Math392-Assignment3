from scipy.integrate import quad
import math

def f(x: float) -> float:
    return 1 / math.sqrt((x**2) + 1)

n: int = 4
lower_bound: int = 0
upper_bound: int = 2

accepted, err = quad(f, lower_bound, upper_bound)

def linear_transform(f, a: float, b: float) -> callable:
    def func(u: float) -> float:
        return f(((b-a) * u + (b+a))/2)*(b-a)/2
    return func

def simpson(f, a: float, b: float, n: int) -> float:
    h: float = (b - a) / n
    arr1 = [f(a + (2 * i * h)) for i in range(1, (int)(n/2))]
    arr2 = [f(a + ((2 * i) - 1) * h ) for i in range(1, (int)(n/2) + 1)]
    return (h/3)*(f(a) + (2 * sum(arr1)) + (4 * sum(arr2)) + f(b))

def gauss_quad(f: callable, a: float, b: float) -> float:
    c1 = 0.3478548451
    c2 = 0.6521451549
    x1 = 0.8611363116
    x2 = 0.3399810436
    if(a != -1.0 or b != 1.0):
      f = linear_transform(f, a, b)
    return c1 * f(-x1) + c2 * f(-x2) + c2 * f(x2) + c1 * f(x1)

if __name__ == "__main__":
    simpsons = simpson(f, lower_bound, upper_bound, n)
    gaussed = gauss_quad(f, lower_bound, upper_bound)

    simpsons_err = abs(accepted - simpsons)
    gaussed_err = abs(accepted - gaussed)

    print(f"{simpsons=:.4f}\t{simpsons_err=:.4f}")
    print(f"{gaussed=:.4f}\t{gaussed_err=:.4f}")
