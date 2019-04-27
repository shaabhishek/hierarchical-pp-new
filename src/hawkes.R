setwd('/Users/abhisheksharma/Dev/hierarchichal_point_process/data/')
lambda0 <- c(0.1, 0.2, 0.3, 0.4, 0.5)
n_cluster <- length(lambda0)
alpha <- c(0.6, 0.7, 0.9, 0.5, 0.4)
beta <- rep(1.0, n_cluster)
horizon <- rep(100, n_cluster)

hawkesdatatrain <- list()
hawkesdataval <- list()
hawkesdatatest <- list()
for (i in 1:n_cluster) {
  for (j in 1:1000) {
    hawkesdatatrain <- c(hawkesdatatrain, simulateHawkes(lambda0 = lambda0[i], alpha = alpha[i], beta = beta[i], horizon = horizon[i]))
  }
  for (j in 1:100) {
    hawkesdataval <- c(hawkesdataval, simulateHawkes(lambda0 = lambda0[i], alpha = alpha[i], beta = beta[i], horizon = horizon[i]))
    hawkesdatatest <- c(hawkesdatatest, simulateHawkes(lambda0 = lambda0[i], alpha = alpha[i], beta = beta[i], horizon = horizon[i]))
  }
}
# Print
head(hawkesdatatrain)
# Save data
fn <- "hawkes.txt"
#Check its existence
if (file.exists(fn)) 
  #Delete file if it exists
  file.remove(fn)

# Save data
lapply(hawkesdatatrain, write, "hawkesdatatrain.txt", append=TRUE, ncolumns=1000)
lapply(hawkesdataval, write, "hawkesdataval.txt", append=TRUE, ncolumns=1000)
lapply(hawkesdatatest, write, "hawkesdatatest.txt", append=TRUE, ncolumns=1000)


