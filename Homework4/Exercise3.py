import subprocess
import numpy as np
from NelderMead import minimize

def function_1(X):
    x, y, z = X[0], X[1], X[2]
    return float(subprocess.Popen(["C:\\Users\\marko\\OneDrive\\Desktop\\M2\\Homework4\\hw4_executables\\hw4_win.exe", "63180365", "1", str(x), str(y), str(z)], stdout=subprocess.PIPE).communicate()[0])

def function_2(X):
    x, y, z = X[0], X[1], X[2]
    return float(subprocess.Popen(["C:\\Users\\marko\\OneDrive\\Desktop\\M2\\Homework4\\hw4_executables\\hw4_win.exe", "63180365", "2", str(x), str(y), str(z)], stdout=subprocess.PIPE).communicate()[0])

def function_3(X):
    x, y, z = X[0], X[1], X[2]
    return float(subprocess.Popen(["C:\\Users\\marko\\OneDrive\\Desktop\\M2\\Homework4\\hw4_executables\\hw4_win.exe", "63180365", "3", str(x), str(y), str(z)], stdout=subprocess.PIPE).communicate()[0])


def partial(g, k, X):
    h = 1e-9
    Y = np.copy(X)
    X[k - 1] = X[k - 1] + h
    dp = (g(X) - g(Y)) / h
    return dp

def grad(f, X):
    grd = []
    for i in np.arange(0, len(X)):
        ai = partial(f, i + 1, X)
        grd.append(ai)
    return grd

def GD(f,X0,eta, steps, tolerance=1e-7):

    #iterations
    i=0
    while True:
        i=i+1
        X0=X0-eta*np.array(grad(f,X0))
        if np.linalg.norm(grad(f,X0))<tolerance or i>steps: break
    return X0

'''
X0 = [2, 2, 2]
eta = 0.001
steps = 400
xmin = GD(function_1, X0, eta, steps)
print(xmin)
'''


'''
print(minimize(function_1, np.array([2,2,2]))) #[0.56308601 0.3081082  0.81362386]
print(minimize(function_2, np.array([2,2,2]))) #[0.81353239 0.30808888 0.5630329 ]
print(minimize(function_3, np.array([2,2,2]))) #[0.30807774 0.81361397 0.56291173]

print(function_1([0.56308601, 0.3081082,  0.81362386])) #0.56308136803426
print(function_2([0.81353239, 0.30808888, 0.5630329 ])) #0.563081363402983
print(function_3([0.30807774, 0.81361397, 0.56291173])) #0.563081363426894
'''


