from matplotlib.pyplot import axes
import numpy as np
import sympy 
from scipy.optimize import linprog


A_correct = A = np.array([
        [18.0,  48.0, 5.0, 1.0, 5.0, 0.0, 0.0, 8.0],  
        [2.0,  11.0,  3.0,  1.0,  3.0,  0.0,  15.0, 1.0],  
        [0.0, 6.0, 3.0, 10.0, 3.0, 100.0, 30.0, 1.0],  
        [77.0, 270.0, 60.0, 140.0, 61.0, 880.0, 330.0, 32.0]
    ])

foods=["potatoes", "bread", "milk", "eggs", "yoghurt", "veg oil", "beef", "strawberies"]

def calculate_Aprime_bprime(A,b,c):
    m = A.shape[0]
    n = A.shape[1]

    e = np.ones((n,1))
    U = max(np.abs(b).max(), np.abs(A).max(), np.abs(c).max())
    W = (m*U)**m

    d = b/W
    rho = d - np.dot(A, e)

    Aprime = np.concatenate((A, np.zeros((m,1)), rho), axis=1)
    Aprime = np.vstack((Aprime, np.ones(n+2)))

    bprime = np.vstack((d, n+2))

    return Aprime, bprime


def check_primal_auxilary(Aprime, bprime, x):
    return (np.dot(Aprime, x)==bprime).all()

def check_dual_auxilary(A, b, c, M, y, s):
    m = A.shape[0]
    n = A.shape[1]

    U = np.abs(A).max()
    W = (m*U)**m
    d = b/W
    e = np.ones((n,1)) 
    rho = d - np.dot(A, e)
    

    s_n1 = s[-2][0]
    s_n2 = s[-1][0]
    s = s[:-2]

    y_m1 = y[-1][0]
    y = y[:-1]

    first_equality = (np.dot(A.T, y) + np.ones((n,1))*y_m1 + s == c).all()
    second_eqality = (np.dot(rho.T[0], y)[0] + y_m1 + s_n2 == M)
    third_equality = (y_m1 + s_n1 == 0)

    return first_equality and second_eqality and third_equality

def check_invariant_1 (A,x,b):
    return (np.dot(A,x) == b).all()

def check_invariant_2 (A, y, s, c):
    return (np.dot(A.T, y)+s == c).all()

def check_invariant_3 (x, s, mu):
    return np.sum((x*s/mu -1)**2) < 1/4

def iterative_improvement(A, b, s, x, mu_prime):
    #We have to get rid of the last two vars
    
    #x = x[:-2] 
    #s = s[:-2]

    e = np.ones((len(x),1))
    S = np.diag(s.T[0])
    X = np.diag(x.T[0])
    S_inv = np.linalg.inv(S)

    left = np.linalg.inv(np.linalg.multi_dot([A, S_inv, X, A.T]))
    right = b - mu_prime * np.linalg.multi_dot([A, S_inv, e])
    k = np.dot( left, right)
    
    f = -np.dot(A.T, k)
    h = -np.linalg.multi_dot([X, S_inv, f]) + mu_prime*np.dot(S_inv,e) - x

    return h, k, f


def interior_point_method(c, A, b):
    #1. Remove the linearly dependant rows of matrix A
    _, inds = sympy.Matrix(A).T.rref() #inds now contains all rows from A which are linearly independant
    A = A[inds, :]
    b = b[inds, :]

    #2. Construct the auxilary primal problem
    Aprime, bprime = calculate_Aprime_bprime(A,b,c)

    #3. Construct the auxilary dual problem
    #function check_dual_auxilary(A, b, c, M, y, s)

    # Intermideate step to calculate M,U,W,L,M
    m = A.shape[0]
    n = A.shape[1]

    U = max(np.abs(b).max(), np.abs(A).max(), np.abs(c).max())
    W = (m*U)**m
    L = 1/W**2 * 1/(2*n*((m+1)*U)**(3*(m+1))) #also known as R in [1]
    M = 4*n*U/L

    #4. Initial point
    mu =  2 * np.sqrt( M**2 + np.sum(np.square(c)) )

    x = np.ones((n+2, 1))
    y = np.zeros((m+1, 1))
    y[-1] = -mu

    s = np.zeros((n+2, 1))
    s[:-2] = c + mu
    s[-2] = mu
    s[-1] = M+mu

    
    '''if not check_invariant_3(x/W, s, mu):
        print('Invariant 3 not satisfied.')
        return
    elif not check_invariant_2(A, y, s, c):
        print('Invariant 2 not satisfied.')
        return
    elif not check_invariant_1(A, x, b):
        print('Invariant 1 not satisfied.')
        return'''
    
    delta = 1/4.65 
    while True:
        print("The new mu is:", mu, "and the value objective gap is", np.dot(x.T,s))
        mu = (1-delta)*mu
        h, k, f = iterative_improvement(Aprime, bprime, s, x, mu)

        x = x + h
        y = y + k
        s = s + f

        if mu<=L**2/(32*m**2):
            break

    #print(W)
    optimal_x = x[:len(x) - 2]
    optimal_s = s[:len(s) - 2]
    optimal_y = y[:len(y) - 1]

    #print(optimal_x, optimal_s)
    
    B = []
    N = []
    N_complement = []

    #print(optimal_x)
    optimal_x_old = optimal_x.copy()

    for i in range(len(optimal_x)):
        if optimal_x[i]<L/(4*m): optimal_x[i]=0; N.append(i)
        else: N_complement.append(i)
        if optimal_s[i]<L/(4*m): optimal_s[i]=0; B.append(i)

    optimal_x = optimal_x*W
    optimal_s = optimal_s*W
    optimal_y = optimal_y*W

    print("B set contains:", B)
    print("N set contains:", N)
    print("N complement set contains:", N_complement)

    A_n = A[:, N_complement]
       
    if A_n.shape[0]<A_n.shape[1]: #recursion
        x_final = interior_point_method(c[N_complement], A_n, b[N_complement])
    elif A_n.shape[0]>A_n.shape[1]:
        #in this scenario we have more equations than variables, so no need to do recursion since we do not want to remove more variables
        #system might be unsolvable in such case
        ...
    else:
        #just solve the squared mtx
        optimal_x_final = np.linalg.solve(A_n, b)
        optimal_x[N_complement, :]= optimal_x_final
    
        

        print("Optimal solution is:")

        A=A*1000; b=b*1000; c=c*1000

        for i in range(len(foods)):
            print(f'{foods[i]:>{15}} : {str(round(optimal_x[i][0], 2))}')

        print("Cost of all these items is:", np.dot(c.T, optimal_x)[0][0])
        print("Daily needs (CH,PR,FT,EN):\n", np.dot(A_correct, optimal_x[:8, :]).T)


def bread_my_method():
    c = np.transpose(np.array([[10.0, 22.0, 15.0, 45.0, 40.0, 20.0, 87.0, 21.0]]))
    A = np.array([
        [-18.0,  -48.0, -5.0, -1.0, -5.0, -0.0, -0.0, -8.0],  
        [-2.0,  -11.0,  -3.0,  -1.0,  -3.0,  -0.0,  -15.0, -1.0],  
        [-0.0, -6.0, -3.0, -10.0, -3.0, -100.0, -30.0, -1.0],  
        [-77.0, -270.0, -60.0, -140.0, -61.0, -880.0, -330.0, -32.0],  
        [18.0,  48.0, 5.0, 1.0, 5.0, 0.0, 0.0, 8.0],  
        [2.0,  11.0,  3.0,  1.0,  3.0,  0.0,  15.0, 1.0],  
        [0.0, 6.0, 3.0, 10.0, 3.0, 100.0, 30.0, 1.0],  
        [77.0, 270.0, 60.0, 140.0, 61.0, 880.0, 330.0, 32.0]
    ])
    b = np.transpose([[-250.0, -50.0, -50.0, -2200.0, 370.0, 170.0, 90.0, 2400.0]])
    
    c = np.concatenate((c, np.zeros((len(A), 1))), axis = 0)
    A = np.concatenate((A, np.identity(len(A))), axis = 1)

    U = max(np.abs(b).max(), np.abs(A).max(), np.abs(c).max())
    reducer = 10 ** (len(str(int(U))) - 1)

    print("red", reducer)

    if reducer >= 100:
        print(reducer)
        interior_point_method(c/1000, A/1000, b/1000)
    else:
        interior_point_method(c, A, b)

def bread_commercial():
    c = [10.0, 22.0, 15.0, 45.0, 40.0, 20.0, 87.0, 21.0]
    A_ub = np.array([
        [-18.0,  -48.0, -5.0, -1.0, -5.0, -0.0, -0.0, -8.0],  
        [-2.0,  -11.0,  -3.0,  -1.0,  -3.0,  -0.0,  -15.0, -1.0],  
        [-0.0, -6.0, -3.0, -10.0, -3.0, -100.0, -30.0, -1.0],  
        [-77.0, -270.0, -60.0, -140.0, -61.0, -880.0, -330.0, -32.0],  
        [18.0,  48.0, 5.0, 1.0, 5.0, 0.0, 0.0, 8.0],  
        [2.0,  11.0,  3.0,  1.0,  3.0,  0.0,  15.0, 1.0],  
        [0.0, 6.0, 3.0, 10.0, 3.0, 100.0, 30.0, 1.0],  
        [77.0, 270.0, 60.0, 140.0, 61.0, 880.0, 330.0, 32.0]
    ])
    b_ub = [-250.0, -50.0, -50.0, -2200.0, 370.0, 170.0, 90.0, 2400.0]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub)
    #print(res)
    print("------Commercial solver solution----------")

    print("Optimal solution is:")
    for i in range(len(foods)):
        print(f'{foods[i]:>{15}} : {str(round(res.x[i], 2))}')
    print("Cost of all these items is:", np.dot(c, res.x))


if __name__=='__main__':
    bread_my_method()
    bread_commercial()

        