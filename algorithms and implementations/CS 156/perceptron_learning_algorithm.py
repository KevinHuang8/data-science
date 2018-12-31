import random
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def sign(n):
	if n > 0:
		return 1
	elif n < 0:
		return -1
	else:
		return 0

# Target function
def f(true_weights, x):
	return sign(np.dot(true_weights, x))

# Must be in y = f(x) form, not general form. Given general equatiion
# ax + by + c = 0, corresponding form is y = -a/b*x + c/b
# weights = [c, a, b]
def f_line(true_weights, x):
	return -(true_weights[1]/true_weights[2]*x) - (true_weights[0]/true_weights[2])

# Note, f and h are implemeted here in the exact same way, two functions are
# created to make clear when we are refering to the target function and when to the
# hypothesis

# hypothesis
def h(w, x):
	return sign(np.dot(w, x))

# See f_line
def h_line(w, x):
	return -(w[1]/w[2]*x) - (w[0]/w[2])

def generate_data(true_weights, N):
	"""generate data based on target function"""
	data = []
	for i in range(N):
		x1 = random.uniform(-1, 1)
		x2 = random.uniform(-1, 1)
		data.append((x1, x2, f(true_weights, np.array([1, x1, x2]))))
	return data

def is_correct(w, x1, x2, y):
	"""
	Tests whether the hypothesis function (defined by w), works
	for a given (x1, x2) and correct value y.

	I.e., tests h(w, [1, x1, x2]) == f([1, x1, x2]) = y
	"""

	return h(w, np.array([1, x1, x2])) == y

def percetron_learning_algorithm(N, runs, init_weights=None, visualize=True, in_data=None):
	"""
	Parameters:
		runs - number of trials to simulate
		N - number of data points in training set
		data - training set, if not provided, it is randomly generated from the target function
		visualize - whether to visualize the algorithm or not

	Returns:
		average steps to converge, probability f(x) != h(x), weights of last run
	"""
	x = np.linspace(-1, 1, 1000)

	total_count = 0
	total_incorrect = 0
	for i in range(runs):
		# (p, q) and (r, s) = 2 random points on the target function
		p = random.uniform(-1, 1)
		q = random.uniform(-1, 1)
		r = random.uniform(-1, 1)
		s = random.uniform(-1, 1)

		# Weights are the coefficients in ax + by + c = 0 form, so
		# must convert two points into general form.
		# Point slope form: y - y0 = m(x - x0)
		# Converted to general form: -mx + y + (-y0 + m*x0) = 0
		# where m = (q - s)/(p - r), y0 = q, x0 = p
		# Thus, a = -(q - s)/(p - r), b = 1, c = -q + p*(q - s)/(p - r)

		a = -(q - s)/(p - r)
		b = 1
		c = -q + p*(q - s)/(p - r)

		# weights of target function, i.e. target function = ax + by + c
		true_weights = np.array([c, a, b])

		if not in_data:
			data = generate_data(true_weights, N)
		else:
			data = in_data

		# the weights of the hypothesis. Originally set at 0 and
		# gradually refined. (though initital weight can be set)
		if init_weights is not None:
			w = init_weights
		else:
			w = np.array([0, 0, 0])

		# all_steps keeps track of all of the hypotheses tested during
		# the algorithm. Only useful for animated visualization using
		# matplotlib
		all_steps = []
		# count keeps track of each iteration of the algorithm	
		count = 0
		while True:
			count += 1
			misclassified = []

			for point in data:
				if not is_correct(w, point[0], point[1], point[2]):
					misclassified.append(point)

			if not misclassified:
				# algorithm done when all points classified correctly
				break

			# Take a random misclassified point and adjust the weights
			# based on it. This is the actual PLA
			point = random.choice(misclassified)
			w = w + point[2]*np.array([1, point[0], point[1]])

			all_steps.append(h_line(w, x))

		total_count += count

		## Test Correctness ##

		# Generate new data
		data2 = generate_data(true_weights, N*10)
		for point in data2:
			if not is_correct(w, point[0], point[1], point[2]):
				total_incorrect += 1

	average_steps = total_count / runs
	probability_failure = total_incorrect / (runs*N*10)

	if visualize:
		### Algorithm Visualization ###

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

		fig = plt.figure()
		plt.axes(xlim=(-1, 1), ylim=(-1, 1))
		hypothesis, = plt.plot([], [])

		def animate(i):
			hypothesis.set_data(x, all_steps[i])
			return hypothesis,

		plt.plot(x, f_line(true_weights, x))
		plt.plot(x, h_line(w, x))

		anim = animation.FuncAnimation(fig, animate, frames=len(all_steps), repeat=False, interval=50, blit=True)

		plot_points(data)

		plt.show()

	return average_steps, probability_failure, w

# Number of trials to simulate
runs = 100
# Number of data points in training set
N = 100

if __name__ == '__main__':
	average_steps, probability_failure, _ = percetron_learning_algorithm(N, runs, np.array([0, 0, 0]), visualize=False)
	print("Average iterations to converge: ", average_steps)
	print("P[f(x) != h(x)]: ", probability_failure)


