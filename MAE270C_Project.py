# Solving Traveling Salesperson Problem (TSP) using dynamic programming
import numpy as np
import math
import matplotlib.pyplot as plt
import os, psutil
import time
process = psutil.Process()
init_memory = process.memory_info().rss

def distFunc(L, n):
    # Generate a symmetric, zero-diagonal matrix where element (i,j) is
    # the Euclidean distance between the coorisponding coordinates in L
    Cost = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            Cost[i,j] = math.sqrt((L[i,0]-L[j,0])**2 + (L[i,1]-L[j,1])**2)
    return Cost

def backStep(p,u,S,prevStep):
    # Find the cost-to-go and route from prevStep that starts at u and 
    # goes through every city in S. Return the cost-to-go including the
    # cost from u to p
    for i in range(len(prevStep[0])):
        if (u==prevStep[2][i][-1] and S==prevStep[0][i]):
            return (Cost[u,p]+prevStep[1][i], prevStep[2][i].copy())

def findShortPath(p,S,pik,prevStep):
    # Find the optimal route and cost-to-go given a starting city (p) and 
    # a set of cities (S) the route must go through before getting to the first
    vkk = []
    pikk = []
    for u in pik:
        v, pi = backStep(p,u,S,prevStep)
        pikk.append(pi)
        vkk.append(v)
    ind = vkk.index(min(vkk))
    return vkk[ind], pikk[ind]

def findNextStep(prevStep, n):
    # Each step is a list of three corresponding sets. The first is a
    # binary code which encodes which cities this route has already been 
    # to. The second is the cost-to-go of the route. The third is
    # the route saved in order.

    # At each step, this function will look at set of cities S, and find
    # the optimal route/cost-to-go from {any city not in the route} through
    # that set of cities, to the first index
    nextStep = [[], [], []]
    for i in range(len(prevStep[0])):
        S = prevStep[0][i]
        pik = prevStep[2][i]
        for p in range(1,n):
            if (p not in pik):
                vskk, piskk = findShortPath(p,S,pik,prevStep)
                piskk.append(p)
                if piskk not in nextStep[2]:
                    Sprime = S
                    Sprime = Sprime | (1<<p)
                    nextStep[0].append(Sprime)
                    nextStep[1].append(vskk)
                    nextStep[2].append(piskk)
    return nextStep




# n = number of cities
n = 8

# Generate random city locations
#np.random.seed(0)
cityLocations = np.random.random((n,2))

# Generate the distance matrix
Cost = distFunc(cityLocations, n)


## Method #1: Solving TSP by finding the shortest route from each possible endpoint all at once
tic1 = time.perf_counter()
process = psutil.Process()
m1_init_memory = process.memory_info().rss

# Build the first step by hand
prevStep = [[],[],[]]
for i in range(1,n):
    prevStep[0].append(1<<i)
    prevStep[1].append(Cost[0,i])
    prevStep[2].append([i])

# Call the dynammic program
while(len(prevStep[2][0]) != n-1):
    #print(prevStep, "\n")
    prevStep = findNextStep(prevStep, n)

# The final step should have the shortest path ending at each unique city
#print(prevStep, "\n")

# Add the cost to get back to the starting city
for i in range(len(prevStep[1])):
    prevStep[1][i] += Cost[0,prevStep[2][i][-1]]

# Find the minimum length path
min_cost = min(prevStep[1])
min_cost_route = prevStep[2][prevStep[1].index(min(prevStep[1]))]

print("\nShortest Path Cost: ", min_cost, " Shortest Path Route: ", min_cost_route,"\n")
process = psutil.Process()
m1_final_memory = process.memory_info().rss
toc1 = time.perf_counter()

# Method #2: Held-Karp-Bellman Algorithm

# Adapted from https://www.geeksforgeeks.org/travelling-salesman-problem-using-dynamic-programming/
# I would claim about 30% of the following code is my modifications made
# in order to fix the code from the link above. 

# memoization. This makes an n-by-((2^n)-1) array. Element [i][mask]
# is the stored cheapest cost of a path starting at node 0, ending at
# node i, and traveling through all nodes in mask.
# Mask is a binary code which indicates cities by turning the ith bit
# in mask from a 0 to 1. 'mask' is then a number which is easy to find
# using binary
tic2 = time.perf_counter()
process = psutil.Process()
m2_init_memory = process.memory_info().rss
memo = [[-1]*(1 << (n)) for _ in range(n)]
 
def fun(i, mask):
    # Find the shortest path starting at node 0 and ending
    # at node i that travels through each node in mask. 
    # Mask is a binary inclusion code. Ex: 100110 indicates
    # that the path must travel through nodes 1, 2, and 5.
    # Note that i must be indicated as a node to travel through
    # in mask

    # base case
    # if only ith bit and 0th bit are set in mask,
    # it implies we have visited all other nodes already
    if mask == ((1 << i) | 1):
        return Cost[0][i]
 
    # Check if this path has been found before
    if memo[i][mask] != -1:
        return memo[i][mask]
 
    res = 10**9  # result of this sub-problem

    # For every node j in mask that is not i (our desired final node), 
    # find the smallest path cost using fun(j,mask\i) (where mask\i)
    # is mask with bit i turned from 1 to 0
    for j in range(1, n):
        if (mask & (1 << j)) != 0 and j != i:
            res = min(res, fun(j, mask & (~(1 << i))) + Cost[j][i])
    memo[i][mask] = res  # storing the minimum value
    return res
 
ans = 10**9
for i in range(1, n):
    # try to go from node 1 visiting all nodes in between to i
    # then return from i taking the shortest route to 1
    ans = min(ans, fun(i, (1 << (n))-1) + Cost[i][0])
 
print("The cost of most efficient tour = " + str(ans) + "\n\n")
process = psutil.Process()
m2_final_memory = process.memory_info().rss
toc2 = time.perf_counter()

# Sort the city locations based on the final path and plot with lines connecting the
# cities in the final path order
sortedCityLocations = np.array([cityLocations[0,:]])
for i in range(n-1):
    sortedCityLocations = np.append(sortedCityLocations, [cityLocations[min_cost_route[i],:]], axis=0)
sortedCityLocations = np.append(sortedCityLocations, [cityLocations[0,:]], axis=0)

plt.plot(sortedCityLocations[:,0],sortedCityLocations[:,1],'-o')
plt.xlim([0,1])
plt.ylim([0,1])
plt.show()

# print("t1: ", toc1-tic1, "\nt2: ", toc2-tic2)
#print("t2: ", toc2-tic2)

# # print(Cost)

#print("Method #1 memory (bytes): ", m1_final_memory-m1_init_memory)
#print("Method #2 memory (bytes): ", m2_final_memory-m2_init_memory)
