# def f(theta, x):
#     s = theta[0]
#     t1 = theta[1]
#     # t2 = theta[2]
#     return (s**x) * t1


# # [s,t1]
# model_params = [2.9999732284003557, 0.0923592666634409]
# s: 1.0647244723054965
# t1: -4.442152275531948
# t2: 1.132961517984578


def f(theta, x):
    s = theta[0]
    t1 = theta[1]
    # t2 = theta[2]
    return s ** (t1 * x**2 + 0.999 * (x**3))


# [s,t1]
model_params = [1.047821962275532, -3.812611452575907]


def predict(x_val):
    return f(model_params, x_val)
