# Variational inference exactly recovers the mean when the target density has rotational symmetry
# Code adapted from Charles Margossian

# Clean up the environment
rm(list = ls())

# Load libraries
.libPaths("~/Rlib/")
library(cmdstanr)
library(scales)
library(VGAM)
library(posterior)
library(bayesplot)
library(sn)
library(matrixStats)
library(MASS)
library(tidyverse)
library(latex2exp)





# Adjust library path, cmdstan path, and working directory to your settings
set_cmdstan_path("/Users/isaacrankin/CmdStan/cmdstan-2.37.0")
setwd("~/Desktop/QP1/VI")

# Make folders for models and plots
if (!dir.exists("models")) {dir.create("models")}
if (!dir.exists("plots")) {dir.create("plots")}




# CmdStanR guide:
# https://mc-stan.org/cmdstanr/articles/cmdstanr.html

# For configuration options for mod$variational see:
# https://mc-stan.org/docs/cmdstan-guide/variational_config.html



# Make an isotropic Gaussian location family
# For Stan instructions see:
# https://mc-stan.org/docs/cmdstan-guide/example_model_data.html

stan_code <- '
data {
  int<lower=1> N;              // Size of dataset
  array[N] vector[2] z_obs;   // Bivariate coordinates 
}

parameters {
  vector[2] mu;              // Parameters are only (mu_1, mu_2)
}

model {
  mu ~ normal(0, 10);    // Suppose mu_1, mu_2 ~ Normal(mean = 0, sd = 10)

  for (n in 1:N)          // Assume each datapoint is independent N(mu, I_2)
    z_obs[n] ~ normal(mu, rep_vector(1, 2));
}
'

# Make it a file
writeLines(stan_code, "models/gaussian_vi.stan")

# Make it a model
mod <- cmdstan_model("models/gaussian_vi.stan")






#---------------------------------------------------- Example 1
# Make synthetic data
# Cylindrical coordinates parametrized by q

# Reproducible (548 for the class number)
set.seed(548)

# Number of points
n <- 10000

# Choose density of q values
m <- 3
s <- 2
q <- rgamma(n, shape = m, scale = s)

# Randomly choose an angle
theta <- runif(n, 0, 2*pi)

# Choose the radius function
r <- q * sin(3*theta)

# Choose mean of z1
a <- 2

# Choose mean of z2
b <- 1

# Convert back to Cartesian coordinates
z1 <- r * cos(theta) + a
z2 <- r * sin(theta) + b



# Make a Stan dataset
stan_data <- list(
  N = n,
  z_obs = lapply(1:n, function(i) c(z1[i], z2[i]))
)



# Fit a meanfield VI, we've already set sigma = 1
fit_vi <- mod$variational(
  data = stan_data,
  seed = 548,
  algorithm = "meanfield",
  elbo_samples = 100,
  grad_samples = 1,
  iter = 1000,
  output_samples = 1000
)



# Get VI mu draws
vi_draws <- as_draws_df(fit_vi$draws())
z_vi <- data.frame(
  z1 = vi_draws$`mu[1]`,
  z2 = vi_draws$`mu[2]`
)
mu_mean <- colMeans(z_vi)
mu_mean

# Make n points from Normal(mu_mean, 1)
set.seed(548)
pred_points <- data.frame(
  z1 = rnorm(n, mean = mu_mean[1], sd = 1),
  z2 = rnorm(n, mean = mu_mean[2], sd = 1)
)

# Combine synthethic and VI points
plot.data <- data.frame(
  z1 = c(pred_points$z1, z1),
  z2 = c(pred_points$z2, z2),
  type = c(rep("VI", n), rep("Synthetic", n))
)

# Means of synthetic and VI
plot.data2 <- data.frame(
  mean_z1 = c(mu_mean[1], mean(z1)),
  mean_z2 = c(mu_mean[2], mean(z2)),
  type = c("VI", "Synthetic")
)

# Plot points, contours, means
ggplot(data = plot.data, aes(x = z1, y = z2, color = type)) +
  geom_density_2d(alpha = 1, linewidth = 1) +
  geom_point(alpha = 0.2, size = 1) +
  geom_point(data = plot.data2, aes(x = mean_z1, y = mean_z2, color = type),
             size = 8, shape = 10, stroke = 1) +
  xlim(-6, 10) +
  ylim(-7, 9) +
  xlab(TeX("$z_1$")) + ylab(TeX("$z_2$")) +
  theme_bw() +
  theme(text = element_text(size = 15), legend.position = c(0.95, 0.95), legend.justification = c(1, 1)) +
  scale_color_discrete(name = "Points") +
  geom_vline(xintercept = a, linetype = "dashed", color = "grey10") +
  geom_hline(yintercept = b, linetype = "dashed", color = "grey10")
ggsave("plots/VI_ex1.png", width = 8, height = 8)











#### Plot the level curves for r as a function of theta and q

# Make 500 values of theta
theta <- seq(0, 2*pi, length.out = 500)
# Choose 4 values of q
q_vals <- c(1, 2, 3, 4)
q_colours <- c("#46B987", "#46A0B9", "#4665B9", "#8746B9")

# Make dataset for the level curves
# Converted to Cartesian coordinates and shifted to a and b
polar_data <- expand.grid(theta = theta, q = q_vals)
polar_data <- polar_data %>%
  mutate(
    r = q * sin(3*theta),
    x = r * cos(theta) + a,
    y = r * sin(theta) + b
    )
# Plot levels curves r(theta, q)
ggplot(polar_data, aes(x = x, y = y, color = factor(q))) +
  geom_path(size = 1) +
  coord_equal() +
  theme_minimal() +
  scale_color_manual(values = q_colours) +
  labs(color = "q", x = "", y = "", title = "")
ggsave("plots/level_curves_ex1.png", width = 6, height = 6)


#### Plot the density of q
q_vals <- seq(0, 15, length.out = 500)
density_vals <- dgamma(q_vals, shape = m, scale = s)
q_df <- data.frame(q = q_vals, density = density_vals)
ggplot(q_df, aes(x = q, y = density)) +
  geom_line(color = "#788FAD", size = 1) +
  theme_minimal() +
  theme(text = element_text(size = 15)) +
  labs(title = "", x = "q", y = "Density")
ggsave("plots/density_q_ex1.png", width = 6, height = 6)












#---------------------------------------------------- Example 2
# Make synthetic data
# Cylindrical coordinates parametrized by q

# Reproducible (548 for the class number)
set.seed(548)

# Number of points
n <- 10000

# Choose density of q values
m <- 0
s <- 2
q <- abs(rnorm(n, mean = m, sd = s))

# Randomly choose an angle
theta <- runif(n, 0, 2*pi)

# Choose the radius function
r <- q * abs(cos(4*theta))

# Choose mean of z1
a <- 2

# Choose mean of z2
b <- 0

# Convert back to Cartesian coordinates
z1 <- r * cos(theta) + a
z2 <- r * sin(theta) + b



# Make a Stan dataset
stan_data <- list(
  N = n,
  z_obs = lapply(1:n, function(i) c(z1[i], z2[i]))
)



# Fit a meanfield VI, we've already set sigma = 1
fit_vi <- mod$variational(
  data = stan_data,
  seed = 548,
  algorithm = "meanfield",
  elbo_samples = 100,
  grad_samples = 5,
  iter = 1000,
  output_samples = 1000
)



# Get VI mu draws
vi_draws <- as_draws_df(fit_vi$draws())
z_vi <- data.frame(
  z1 = vi_draws$`mu[1]`,
  z2 = vi_draws$`mu[2]`
)
mu_mean <- colMeans(z_vi)
mu_mean

# Make n points from Normal(mu_mean, 1)
set.seed(548)
pred_points <- data.frame(
  z1 = rnorm(n, mean = mu_mean[1], sd = 1),
  z2 = rnorm(n, mean = mu_mean[2], sd = 1)
)

# Combine synthethic and VI points
plot.data <- data.frame(
  z1 = c(pred_points$z1, z1),
  z2 = c(pred_points$z2, z2),
  type = c(rep("VI", n), rep("Synthetic", n))
)

# Means of synthetic and VI
plot.data2 <- data.frame(
  mean_z1 = c(mu_mean[1], mean(z1)),
  mean_z2 = c(mu_mean[2], mean(z2)),
  type = c("VI", "Synthetic")
)

# Plot points, contours, means
ggplot(data = plot.data, aes(x = z1, y = z2, color = type)) +
  geom_density_2d(alpha = 1, linewidth = 0.5, contour_var = "ndensity") +
  geom_point(alpha = 0.5, size = 1) +
  geom_point(data = plot.data2, aes(x = mean_z1, y = mean_z2, color = type),
             size = 10, shape = 10, stroke = 1) +
  xlim(-2, 6) +
  ylim(-4, 4) +
  xlab(TeX("$z_1$")) + ylab(TeX("$z_2$")) +
  theme_bw() +
  theme(text = element_text(size = 15), legend.position = c(0.95, 0.95), legend.justification = c(1, 1)) +
  scale_color_discrete(name = "Points") +
  geom_vline(xintercept = a, linetype = "dashed", color = "grey10") +
  geom_hline(yintercept = b, linetype = "dashed", color = "grey10")
ggsave("plots/VI_ex2.png", width = 8, height = 8)











#### Plot the level curves for r as a function of theta and q

# Make 500 values of theta
theta <- seq(0, 2*pi, length.out = 500)
# Choose 4 values of q
q_vals <- c(1, 2, 3, 6)
q_colours <- c("#46B987", "#46A0B9", "#4665B9", "#8746B9")

# Make dataset for the level curves
# Converted to Cartesian coordinates and shifted to a and b
polar_data <- expand.grid(theta = theta, q = q_vals)
polar_data <- polar_data %>%
  mutate(
    r = q * abs(cos(4*theta)),
    x = r * cos(theta) + a,
    y = r * sin(theta) + b
  )
# Plot levels curves r(theta, q)
ggplot(polar_data, aes(x = x, y = y, color = factor(q))) +
  geom_path(size = 1) +
  coord_equal() +
  theme_minimal() +
  scale_color_manual(values = q_colours) +
  labs(color = "q", x = "", y = "", title = "")
ggsave("plots/level_curves_ex2.png", width = 6, height = 6)


#### Plot the density of q
q_vals <- seq(0, 10, length.out = 500)
density_vals <- 2 * dnorm(q_vals, mean = m, sd = s)
q_df <- data.frame(q = q_vals, density = density_vals)
ggplot(q_df, aes(x = q, y = density)) +
  geom_line(color = "#788FAD", size = 1) +
  theme_minimal() +
  theme(text = element_text(size = 15)) +
  labs(title = "", x = "q", y = "Density")
ggsave("plots/density_q_ex2.png", width = 6, height = 6)
















#---------------------------------------------------- Example 3
# Make synthetic data
# Cylindrical coordinates parametrized by q

# Reproducible (548 for the class number)
set.seed(548)

# Number of points
n <- 10000

# Choose density of q values
m <- 10
s <- 0.5
q <- rgamma(n, shape = m, scale = s)

# Randomly choose an angle
theta <- runif(n, 0, 2*pi)

# Choose the radius function
r <- q * (cos(theta) + sin(theta)) * sin(4 * theta)

# Choose mean of z1
a <- -1

# Choose mean of z2
b <- 2

# Convert back to Cartesian coordinates
z1 <- r * cos(theta) + a
z2 <- r * sin(theta) + b



# Make a Stan dataset
stan_data <- list(
  N = n,
  z_obs = lapply(1:n, function(i) c(z1[i], z2[i]))
)



# Fit a meanfield VI, we've already set sigma = 1
fit_vi <- mod$variational(
  data = stan_data,
  seed = 548,
  algorithm = "meanfield",
  elbo_samples = 100,
  grad_samples = 1,
  iter = 1000,
  output_samples = 1000
)



# Get VI mu draws
vi_draws <- as_draws_df(fit_vi$draws())
z_vi <- data.frame(
  z1 = vi_draws$`mu[1]`,
  z2 = vi_draws$`mu[2]`
)
mu_mean <- colMeans(z_vi)
mu_mean

# Make n points from Normal(mu_mean, 1)
set.seed(548)
pred_points <- data.frame(
  z1 = rnorm(n, mean = mu_mean[1], sd = 1),
  z2 = rnorm(n, mean = mu_mean[2], sd = 1)
)

# Combine synthethic and VI points
plot.data <- data.frame(
  z1 = c(pred_points$z1, z1),
  z2 = c(pred_points$z2, z2),
  type = c(rep("VI", n), rep("Synthetic", n))
)

# Means of synthetic and VI
plot.data2 <- data.frame(
  mean_z1 = c(mu_mean[1], mean(z1)),
  mean_z2 = c(mu_mean[2], mean(z2)),
  type = c("VI", "Synthetic")
)

# Plot points, contours, means
ggplot(data = plot.data, aes(x = z1, y = z2, color = type)) +
  geom_density_2d(alpha = 1, linewidth = 0.75) +
  geom_point(alpha = 0.1, size = 1) +
  geom_point(data = plot.data2, aes(x = mean_z1, y = mean_z2, color = type),
             size = 10, shape = 10, stroke = 1) +
  xlim(-6, 6) +
  ylim(-6, 8) +
  xlab(TeX("$z_1$")) + ylab(TeX("$z_2$")) +
  theme_bw() +
  theme(text = element_text(size = 15), legend.position = c(0.95, 0.95), legend.justification = c(1, 1)) +
  scale_color_discrete(name = "Points") +
  geom_vline(xintercept = a, linetype = "dashed", color = "grey10") +
  geom_hline(yintercept = b, linetype = "dashed", color = "grey10")
ggsave("plots/VI_ex3.png", width = 8, height = 8)











#### Plot the level curves for r as a function of theta and q

# Make 500 values of theta
theta <- seq(0, 2*pi, length.out = 500)
# Choose 4 values of q
q_vals <- c(2, 4, 6, 10)
q_colours <- c("#46B987", "#46A0B9", "#4665B9", "#8746B9")

# Make dataset for the level curves
# Converted to Cartesian coordinates and shifted to a and b
polar_data <- expand.grid(theta = theta, q = q_vals)
polar_data <- polar_data %>%
  mutate(
    r = q * (cos(theta) + sin(theta)) * sin(4 * theta),
    x = r * cos(theta) + a,
    y = r * sin(theta) + b
  )
# Plot levels curves r(theta, q)
ggplot(polar_data, aes(x = x, y = y, color = factor(q))) +
  geom_path(size = 1) +
  coord_equal() +
  theme_minimal() +
  scale_color_manual(values = q_colours) +
  labs(color = "q", x = "", y = "", title = "")
ggsave("plots/level_curves_ex3.png", width = 6, height = 6)


#### Plot the density of q
q_vals <- seq(0, 15, length.out = 500)
density_vals <- dgamma(q_vals, shape = m, scale = s)
q_df <- data.frame(q = q_vals, density = density_vals)
ggplot(q_df, aes(x = q, y = density)) +
  geom_line(color = "#788FAD", size = 1) +
  theme_minimal() +
  theme(text = element_text(size = 15)) +
  labs(title = "", x = "q", y = "Density")
ggsave("plots/density_q_ex3.png", width = 6, height = 6)

