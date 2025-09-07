import numpy as np
import random as rn
import json
import math
import matplotlib.pyplot as plt

xp,yp = [],[]

with open("data.txt", "r") as f:
    for line in f:
        record = json.loads(line)
        xp.append(record["x"])
        yp.append(record["y"])

def f(theta, x):
    s =  theta[0]
    t1 =  theta[1]
    t2 =  theta[2]
    
    val = x**s 
    return val

def error(y_, y):
    return (y_ - y)**2
    

best_loss = float('inf')
best_model = [0,0,0]

for i in range(10000):
    s =  rn.randint(0,3)
    t1 =  rn.randint(-100,100)
    t2 =  rn.randint(-50,-1)
    loss = 0

    for x, y in zip(xp, yp):
        theta = [s,t1,t2]
        yhat = f(theta,x)
        loss += error(yhat,y)
    
    if(loss<best_loss):
        best_loss = loss
        best_model = [s,t1,t2]
        print("new best loss: ",best_loss)

plt.plot(xp, yp)
plt.xlabel("data")
plt.ylabel("distance")
plt.show()

print(best_loss)
print("s: " + str(best_model[0]))
print("t1: " + str(best_model[1]))
print("t2: " + str(best_model[2]))

