setwd('/Users/abhisheksharma/Dev/hierarchichal_point_process/data/')
params_data <- read.delim("synthetic_data_params.txt", header = FALSE, col.names = c('lambda0', 'alpha', 'beta', 'horizon'))

# n_cluster <- c(5, 10, 50, 100)
n_cluster <- c(5, 10)
# lambda0 <- c(0.1, 0.2, 0.3, 0.4, 0.5)
# n_cluster <- length(lambda0)
# alpha <- c(0.6, 0.7, 0.9, 0.5, 0.4)
# beta <- rep(1.0, n_cluster)
# horizon <- rep(100, n_cluster)

intensity_nhpp_wrapper <- function(coeff) {
  function(x) sinpi(coeff*x)
}


generate_hawkes <- function(params, splittype) {
  sim_pp <- function(idx, params) {
    if (idx%%2 == 0){
      res <- nhpp.sim(10, 50, intensity_nhpp_wrapper(params[1]), prepend.t0 = F)
    }else{
      attempts = 0
      res <- simulateHawkes(params[1], params[2], params[3], params[4])
      while (length(unlist(res)) < 2)
      {
        attempts <- attempts + 1
        # print(c(attempts, params))
        res <- simulateHawkes(params[1], params[2], params[3], params[4])
      }
    }
    
    return (unlist(res))
  }
  if (splittype =='train'){
    n <- 500
  } else {
    n <- 50
  }
  
  return (sapply(1:n, function(x) sim_pp(x, params)))
}

deletepreviousdata <- function(fn) {
  if (file.exists(fn)){
    #Delete file if it exists
    file.remove(fn)}
}

simulate <- function(n_cluster) {
  for (nclus in n_cluster) {
    print(nclus)
    train_data = apply(params_data[1:nclus,1:4], 1, function(x) generate_hawkes(x, 'train'))
    val_data = apply(params_data[1:nclus,1:4], 1, function(x) generate_hawkes(x, 'val'))
    test_data = apply(params_data[1:nclus,1:4], 1, function(x) generate_hawkes(x, 'test'))
    
    # Flatten clusters into one data set
    train_data <- unlist(train_data, recursive = FALSE, use.names = FALSE)
    val_data <- unlist(val_data, recursive = FALSE, use.names = FALSE)
    test_data <- unlist(test_data, recursive = FALSE, use.names = FALSE)
    
    # File names
    fn_base <- paste0('dump/syntheticdata_nclusters_', nclus) 
    fn_train <- paste0(fn_base, '_train.txt')
    fn_val <- paste0(fn_base, '_val.txt')
    fn_test <- paste0(fn_base, '_test.txt')
    
    # Delete previous files
    sapply(c(fn_train, fn_val, fn_test), deletepreviousdata)
    
    # Save data
    print(lapply(list(train_data, val_data, test_data), function(x) c(min(sapply(x, length)), max(sapply(x, length)), mean(sapply(x, length)))))
    lapply(train_data, function(x) write(x, file = fn_train, append = TRUE, ncolumns = length(x)))
    lapply(val_data, function(x) write(x, file = fn_val, append = TRUE, ncolumns = length(x)))
    lapply(test_data, function(x) write(x, file = fn_test, append = TRUE, ncolumns = length(x)))
  }
}

simulate(n_cluster)
