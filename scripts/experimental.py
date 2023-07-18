from scipy.stats import truncnorm

# Define the lower and upper bounds for the truncated normal distribution
lower_bound = -2
upper_bound = 2

# Define the mean and standard deviation of the desired normal distribution
mean = 0
std = 1

# Calculate the a and b values based on the lower_bound, upper_bound, mean, and std
a = (lower_bound - mean) / std
b = (upper_bound - mean) / std

# Generate a random sample from the truncated normal distribution
sample = truncnorm.rvs(a, b, loc=mean, scale=std, size=10)

print(sample)