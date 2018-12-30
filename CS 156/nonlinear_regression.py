import random
import matplotlib.pyplot as plt
import numpy as np

runs = 1000
N = 1000

def f(x1, x2):
	return sign(x1**2 + x2**2 - 0.6)

def h(w, x1, x2):
	return sign(np.dot(w, np.array([1, x1, x2, x1*x2, x1**2, x2**2])))

def sign(n):
	if n > 0:
		return 1
	elif n < 0:
		return -1
	else:
		return 0

def generate_data(N):
	"""generate data based on target function"""
	data = []
	y = []
	for i in range(N):
		x1 = random.uniform(-1, 1)
		x2 = random.uniform(-1, 1)
		y_val = f(x1, x2)

		### add noise
		if random.randint(1, 10) == 1:
			y_val = -y_val

		y.append(y_val)
		data.append((x1, x2, y_val))
	return data, y

def transform_data(data):
	"""circular transformation"""
	X = np.array([])
	for x1, x2, _y in data:
		try:
			X = np.vstack((X, np.array([1, x1, x2, x1*x2, x1**2, x2**2])))
		except ValueError:
			X = np.array([1, x1, x2, x1*x2, x1**2, x2**2])
	return X

x = np.linspace(-1, 1, 1000)
y = np.linspace(-1, 1, 1000)
xx, yy = np.meshgrid(x, y)

w_average = np.array([0, 0, 0, 0, 0, 0])

total_incorrect_out = 0
for i in range(runs):
	data, y = generate_data(N)
	X = transform_data(data)
	X_dagger = np.linalg.pinv(X)
	w = np.matmul(X_dagger, y)

	w_average = np.add(w_average, w)

	data_out, _y = generate_data(1000)
	for x1, x2, _yy in data_out:
		if h(w, x1, x2) != _yy:
			total_incorrect_out += 1

E_out = total_incorrect_out / (runs*1000)
print(E_out)

w_average = np.divide(w_average, runs)
print(w)

def plot_points(points):
	"""
	Parameters:
		points - a list of (x, y, sign) tuples

	Plots (x, y) values, with a point being a red circle if 
	the sign is positive, and a blue square if negative.
	"""
	for p in points:
		if p[2] == 1:
			plt.plot(p[0], p[1], 'ro')
		else:
			plt.plot(p[0], p[1], 'bs')

F = w_average[4]*(xx**2) + w_average[5]*(yy**2) + w_average[3]*xx*yy \
 + w_average[2]*yy + w_average[1]*xx + w_average[0]

plt.contour(xx, yy, F, [0])
#plt.axes(xlim=(-1, 1), ylim=(-1, 1))
plot_points(data)
plt.show()