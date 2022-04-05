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


#print(minimize(function_1, np.array([2,2,2]))) #[0.56308601 0.3081082  0.81362386]
print(minimize(function_2, np.array([2,2,2]))) #[0.81353239 0.30808888 0.5630329 ]
print(minimize(function_3, np.array([2,2,2]))) #[0.30807774 0.81361397 0.56291173]
