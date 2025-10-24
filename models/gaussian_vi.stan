
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

