import numpy as np
from Exercise5 import set_weights, relaxMaximalWeightMatching2
import random

#x1 | x2 union
#x1 & x2 intersection
#x1.remove("1")
#x1.add("1")

def check_condition(G, x):
    left_side = np.dot(G,x)
    for i in left_side:
        if i>1:
            return False
    return True

def cost(x,w):
    return np.dot(x,w)


def find_edges_to_add(k, edges_missing, x, G):
    new_edges=[]
    count=0
    while len(new_edges)<k:
        #print("Searching...")
        #currently_inside = np.sum(x)
        
        element_add = random.sample(edges_missing, 1)[0]
        x[element_add]=1
        if check_condition(G, x):
            new_edges.append(element_add)
            edges_missing.remove(element_add)
        else:
            x[element_add]=0

        if count>1000:
            break
        count+=1
        
    for i in new_edges:
        x[i]=0
        edges_missing.add(i)

    return new_edges


def step(k, x, w, edges_added, edges_missing, G):
    x_old = x.copy()
    edges_added_old = edges_added.copy()
    edges_missing_old = edges_missing.copy()

    probability = random.uniform(0,1)

    '''if np.sum(x)>20 and random.uniform(0,1)<0.2:
        for _ in range(3):
            remove_edge_idx = np.argmin(w[list(edges_added)])
            remove_edge = list(edges_added)[remove_edge_idx]
            x[remove_edge] = 0
            edges_added.remove(remove_edge)
            edges_missing.add(remove_edge)'''
        


    if np.sum(x)>10: #we can start removing
        if random.uniform(0,1) < 0.7: #then remove k edges
            for i in range(k):
                element_remove = random.sample(edges_added, 1)[0]
                if probability<0.2: #with 20% prob we remove randomly
                    x[element_remove] = 0
                    edges_added.remove(element_remove)
                    edges_missing.add(element_remove)
                elif w[element_remove]<1.5: #remove this edge only if weight is not big
                    x[element_remove] = 0
                    edges_added.remove(element_remove)
                    edges_missing.add(element_remove)
        if random.uniform(0,1) < 0.5: #then add k edges
            edges_to_add = find_edges_to_add(k, edges_missing, x, G)
            for element_add in edges_to_add:
                if probability<0.2: #with 20% prob we add randomly
                    x[element_add] = 1
                    edges_missing.remove(element_add)
                    edges_added.add(element_add)
                elif w[element_add]>1.7: #add this edge only if weight is big
                    x[element_add] = 1
                    edges_missing.remove(element_add)
                    edges_added.add(element_add) 
    else:
        edges_to_add = find_edges_to_add(k, edges_missing, x, G)
        for element_add in edges_to_add:
            if probability<0.2: #with 20% prob we add randomly
                x[element_add] = 1
                edges_missing.remove(element_add)
                edges_added.add(element_add)
            elif w[element_add]>1.7: #add this edge only if weight is big
                x[element_add] = 1
                edges_missing.remove(element_add)
                edges_added.add(element_add) 


    if cost(x_old, w) > cost(x, w):
        return x_old, w, edges_added_old, edges_missing_old, G
    else:
        return x, w, edges_added, edges_missing, G


def MaxWeightMatching(k):
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

    #To check whether the rules for matching are not violated: Gx<1
    edges_added=set([])
    edges_missing=set([i for i in range(840)])

    x = np.zeros(840)
    for i in range(1000):
        x, w, edges_added, edges_missing, G = step(k, x, w, edges_added, edges_missing, G)
        


    for matching, weight in zip(x, w):
        print("Weight:", weight, "soution:", matching)

    x_relaxed = relaxMaximalWeightMatching2(output = False)
    print("This matching contains", np.sum(x), "egdes")
    print("Cost which we were maximizing is:", cost(x,w))
    print("How far away are we from the relaxed:", np.sum(np.absolute(x_relaxed-x)))
    return x


if __name__=='__main__':
    x = MaxWeightMatching(3)