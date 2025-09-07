import numpy as np
import random as rn
import json
import math

xp,yp = [],[]

with open("some_data_for_class.txt", "r") as f:
    for line in f:
        record = json.loads(line)
        xp.append(record["x"])
        yp.append(record["y"])

def f(theta, x):
    s =  theta[0]
    t1 =  theta[1]
    t2 =  theta[2]
    t3 =  theta[3]
    t4 =  theta[4]

    val = ((x[0]**0.5)*t1 + (x[1]**0.5)*t2 + (x[2]**0.1)*5 + s) / t4
    return val

def error(y_, y):
    return (y_ - y)**2
    

best_loss = float('inf')
best_model = [0,0,0,0,0]

for i in range(10000):
    s =  rn.randint(1,80)
    t1 =  rn.randint(-100,100)
    t2 =  rn.randint(-50,-1)
    t3 =  rn.randint(-10,10)
    t4 = rn.randint(1,10)
    loss = 0

    for x, y in zip(xp, yp):
        theta = [s,t1,t2,t3,t4]
        yhat = f(theta,x)
        loss += error(yhat,y)
    
    if(loss<best_loss):
        best_loss = loss
        best_model = [s,t1,t2,t3,t4]
        print("new best loss: ",best_loss)

print(best_loss)
print("s: " + str(best_model[0]))
print("t1: " + str(best_model[1]))
print("t2: " + str(best_model[2]))
print("t3: " + str(best_model[3]))
print("t4: " + str(best_model[4]))
