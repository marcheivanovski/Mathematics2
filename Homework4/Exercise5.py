from re import L
import cvxopt
import numpy as np
import math
import sympy
from scipy.optimize import linprog

def set_weights():
    np.random.seed(42)
    w = np.random.uniform(1,2,840)

    #The added external edges get weight 0
    for i in range(20):
        w[i]=0

    for i in range(20,800,41):
        w[i]=0

    for i in range(40,820,41):
        w[i]=0

    for i in range(820,840):
        w[i]=0

    return w

def relaxMaximalWeightMatching2(output=False):
    w = set_weights()
    G = np.zeros((400,840)) #every row in G represents a condition that no sum of edges from one vertex can be bigger than 1

    row=0
    for i in range(0,780,41):
        for j in range(i,i+20):
            G[row,j]=1
            G[row,j+20]=1
            G[row,j+21]=1
            G[row,j+41]=1
            row+=1

    h = np.ones(400)

    c = -1 * w
    A_ub = G
    b_ub = h
    res = linprog(c, A_ub=A_ub, b_ub=b_ub)
    final_solution = res['x']
    #print(final_solution)
    if output:
        for x, weight in zip(final_solution, w):
            print("Weight:", weight, "soution:", x)
        print("This fractional matching contains", np.sum(final_solution), "egdes")
        print("Cost which we were maximizing is:", np.dot(final_solution,w))

    
    return final_solution


#This was written using cvxopt but the library complaied about something...
def relaxMaximalWeightMatching():
    w = set_weights()
    G = np.zeros((400,840))

    row=0
    for i in range(0,780,41):
        for j in range(i,i+20):
            G[row,j]=1
            G[row,j+20]=1
            G[row,j+21]=1
            G[row,j+41]=1
            row+=1

    h = np.ones(400)

    _, inds = sympy.Matrix(G).T.rref() #inds now contains all rows from A which are linearly independant
    G = G[inds, :]
    h = h[list(inds)]

    c = cvxopt.matrix(w)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)

    #cvxopt.solvers.qp.options['show_progress'] = False
    solution = cvxopt.solvers.lp(c, G, h)
    solution = np.ravel(solution['x'])
    print(solution)

if __name__=="__main__":
    relaxMaximalWeightMatching2(output=True)

