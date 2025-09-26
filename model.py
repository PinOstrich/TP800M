# model.py
import numpy as np
import json
import random as rn


def load_data(filename="data.txt"):
    xp, yp = [], []
    with open(filename, "r") as file:
        for line in file:
            record = json.loads(line)
            xp.append(record["x"])
            yp.append(record["y"])
    return xp, yp


def f(theta, x):
    s = theta[0]
    t1 = theta[1]
    return (s**x) * t1


def error(y_pred, y_true):
    return (y_pred - y_true) ** 2


def train_model(xp, yp, iterations=15000):
    best_loss = float("inf")
    best_model = [0, 0, 0]

    for _ in range(iterations):
        s = rn.uniform(2, 3)
        t1 = rn.uniform(0, 1)
        t2 = rn.uniform(0, 3)
        theta = [s, t1, t2]

        loss = sum(error(f(theta, x), y) for x, y in zip(xp, yp))

        if loss < best_loss:
            best_loss = loss
            best_model = theta

    return best_model


xp, yp = load_data()
model_params = train_model(xp, yp)


def predict(x_val):
    return f(model_params, x_val)
