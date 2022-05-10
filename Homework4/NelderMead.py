import numpy as np
import math

def shrink(vertices):
    #print('shrink')
    x_o = vertices[0]
    vertices_new=np.zeros((vertices.shape[0], vertices.shape[1]))
    vertices_new[0]=x_o
    vertices_new[1:,:] = 1/2*(vertices[1:,:] + x_o)
    return vertices_new

def minimize(fun, initial_point, delta=0.1, max_iterations=10000, tolerance=1e-8):
    n =len(initial_point)
    
    vertices=[np.array(initial_point)]
    for i in range(len(initial_point)):
        displacement = np.zeros(n)
        displacement[i]=delta
        vertices.append(initial_point+displacement)

    vertices=np.array(vertices)
    n_iter=0
    while True:
        #vertices_new=[]

        y = np.apply_along_axis(fun, axis=1, arr=vertices) #axis 1 means apply for every row
        vertices = vertices[y.argsort()]

        if n_iter==max_iterations or fun(vertices[-1])-fun(vertices[0])<tolerance: #check stopping criteria
            #print(n_iter)
            break

        #print(vertices, end='\n--------------------------\n')

        centroid = 1/n * (np.sum(vertices[:-1], axis=0))
        reflect = centroid + (centroid - vertices[-1])

        y_r = fun(reflect)

        if y_r<y[0]: #case 1
            expand = centroid + 2*(centroid-vertices[-1])
            y_e = fun(expand)

            vertices_new=vertices.copy()
            vertices_new[-1] = np.array(expand) if y_e  < y_r else np.array(reflect)
        elif y_r<y[-2]: #case 2
            vertices_new=vertices.copy()
            vertices_new[-1] = np.array(reflect)
        elif y_r<y[-1]: #case 3
            contraction_outside = centroid + 1/2*(centroid-vertices[-1])
            y_co = fun(contraction_outside)
             
            if y_co  < y_r:
                vertices_new=vertices.copy()
                vertices_new[-1] = np.array(contraction_outside)
            else:
                vertices_new=shrink(vertices)
        else: #case 4
            contraction_inside = centroid - 1/2*(centroid-vertices[-1])
            y_ci = fun(contraction_inside)
            if y_ci  < y_r:
                vertices_new=vertices.copy()
                vertices_new[-1] = np.array(contraction_inside)
            else:
                vertices_new=shrink(vertices.copy())

        n_iter+=1

        vertices = vertices_new.copy()
        del vertices_new

    return vertices[0]


def rosenbrock(X):
    x, y = X[0], X[1]
    return 100 * math.pow(y - math.pow(x, 2), 2) + math.pow(1 - x, 2)

if __name__ == "__main__":
    print(minimize(rosenbrock, (100, 100)))