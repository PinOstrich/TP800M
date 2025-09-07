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
loss_history = []
yp_model = []

for i in range(10000):
    s = rn.uniform(0, 3)
    t1 =  rn.randint(-3,3)
    t2 =  rn.randint(-50,-1)
    loss = 0

    for x, y in zip(xp, yp):
        theta = [s,t1,t2]
        yhat = f(theta,x)
        loss += error(yhat,y)
    if(loss<best_loss):
        best_loss = loss
        best_model = [s,t1,t2]
        loss_history.append(loss)
        print("new best loss: ",best_loss)
    
print(best_loss)
print("s: " + str(best_model[0]))
print("t1: " + str(best_model[1]))
print("t2: " + str(best_model[2]))

input  = float(input("Enter a float: "))
print("Distance is: ", str(f(best_model,input)))

plt.figure()
plt.scatter(xp, yp, s=10, color="blue")
plt.xlabel("data")
plt.ylabel("distance")

x = list(range(len(loss_history)))  # 0 to 10000 inclusive
plt.figure()
plt.plot(x, loss_history, color="red")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss over Iterations")

x = np.arange(0,max(xp),0.1)  # 0 to 10000 inclusive
plt.figure()
plt.plot(x,f(best_model,x), color="green")
plt.xlabel("input")
plt.ylabel("output")
plt.title("fitted function")

plt.show()


