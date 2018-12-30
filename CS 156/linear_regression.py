import random
import matplotlib.pyplot as plt
import numpy as np

N = 100

# Target function
def f(x):
	return np.dot(true_weights, x)

def f_line(x):
	return true_weights[0] + true_weights[1] * x

def h_line(w, x):
	return w[0] + w[1] * x

def generate_data(N):
	"""generate data based on target function"""
	data = []
	y = []
	for i in range(N):
		x1 = random.uniform(-1, 1)
		noise = np.random.normal(0, 0.5)
		y_val = f(np.array([1, x1])) + noise
		y.append(y_val)
		data.append((x1, y_val))
	return data, y

x = np.linspace(-1, 1, 1000)

# (p, q) and (r, s) = 2 random points on the target function
p = random.uniform(-1, 1)
q = random.uniform(-1, 1)
r = random.uniform(-1, 1)
s = random.uniform(-1, 1)

# Weights are the coefficients in y = mx + b form, so
# must convert two points into slope-intercept form.
# Point slope form: y - y0 = m(x - x0)
# Converted to general form: y = mx + (-x0m + y0)
# where m = (q - s)/(p - r), y0 = p, x0 = q
# b = -q(q - s)/(p - r) + p

m = (q - s)/(p - r)
b = -q*(q - s)/(p - r) + p

# weights of target function, i.e. target function = ax + by + c
true_weights = np.array([b, m])

data, y = generate_data(N)

X = np.array([])
for x1, _y in data:
	try:
		X = np.vstack((X, np.array([1, x1])))
	except ValueError:
		X = np.array([1, x1])

X_dagger = np.linalg.pinv(X)
w = np.matmul(X_dagger, y)

plt.plot(x, f_line(x))
plt.plot(x, h_line(w, x))

for x1, y in data:
	plt.plot(x1, y, 'ro')

plt.show()
