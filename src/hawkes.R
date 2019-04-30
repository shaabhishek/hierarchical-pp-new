library(hawkes)
library(poisson)
setwd('/Users/abhisheksharma/Dev/hierarchichal_point_process/data/')
params_data <- read.delim("synthetic_data_params.txt", header = FALSE) #col.names = c('lambda0', 'alpha', 'beta', 'horizon'))

# n_cluster <- c(5, 10, 50, 100)
n_cluster <- c(4)
# lambda0 <- c(0.1, 0.2, 0.3, 0.4, 0.5)
# n_cluster <- length(lambda0)
# alpha <- c(0.6, 0.7, 0.9, 0.5, 0.4)
# beta <- rep(1.0, n_cluster)
# horizon <- rep(100, n_cluster)

intensity_nhpp_wrapper <- function(coeff) {
  function(x) sinpi(coeff*x)
}

belowminthres <- function(x) {
  min(diff(x)) < 0.001
}

generate_hawkes <- function(params, splittype) {
  print(params)
  sim_pp <- function(idx, params) {
    ready = FALSE
    attempts = 0
    while (!ready) {
      attempts <- attempts + 1
      
      if (params[1]=='Poisson'){
        res <- nhpp.sim(1, 50, intensity_nhpp_wrapper(as.double(params[2])), prepend.t0 = F)
        if (!belowminthres(res)){
          ready <- TRUE
        }
      }
      else{
        lambda0 = as.double(params[2])
        alpha = as.double(params[3]) #rnorm(1, mean=as.double(params[3]), 0.01)
        beta = as.double(params[4]) #rnorm(1, mean=as.double(params[4]), 0.01)
        beta = min(c(beta, 1.0))
        alpha = min(c(alpha, beta-.01))
        horizon = 25
        res <- simulateHawkes(lambda0, alpha, beta, horizon)
        
        if ((length(unlist(res)) >= 2) && (!belowminthres(unlist(res)))){
          ready <- TRUE
          # print(c(attempts, params))
          # res <- simulateHawkes(lambda0, alpha, beta, horizon)
          # if (attempts == 100) break
        } 
        else{
          horizon <- 2*horizon
        }
        
      }
    }
    
    
    return (unlist(res))
  }
  if (splittype =='train'){
    n <- 500
  } else {
    n <- 50
  }
  
  return (lapply(1:n, function(x) sim_pp(x, params)))
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

# simulate(n_cluster)


intensity_hp <- function(t, P, params){
  P_ <- P[P<t]
  params[1] + params[2]*sum(exp(-1* (t - P_)/params[3]))
}

genHP <- function (T, intensity_hp, params) {
  t = 0
  P = c()
  while(t < T){
    m <- intensity_hp(t, P, params)+1
    e <- rexp(1, rate = m)
    t <- t + e
    u <- runif(1, 0, m)
    if ((t < T) && (u <= intensity_hp(t, P, params))){
      P <- c(P, t)
    }
  }
  return (P)
}

intensity_nhpp <- function(t, params) {
  0.5*sinpi(params[1]*t) + 0.5*cospi(.2*params[1]*t) + 1
}

genNHPP <- function(T, intensity_nhpp, params) {
  t = 0
  P = c()
  while(t < T){
    m <- 5
    e <- rexp(1, rate = m)
    t <- t + e
    u <- runif(1, 0, m)
    if ((t < T) && (u <= intensity_nhpp(t, params))){
      P <- c(P, t)
    }
  }
  return (P)
}

intensity_scp <- function(t, P, params) {
  P_ <- P[P<t]
  # print(sum(table(P_)))
  exp(params[1]*t - params[2]*sum(table(P_)))
}

genSCP <- function(T, intensity_scp, params) {
  t = 0
  P = c()
  while(t < T){
    m <- intensity_scp(t+1, P, params)
    # print(c(t, m))
    e <- rexp(1, rate = m)
    t <- t + e
    u <- runif(1, 0, m)
    if ((t < T) && (u <= intensity_scp(t, P, params))){
      P <- c(P, t)
    }
  }
  return (P)
}


# sigmoid <- function(x) 1/(1+exp(x))

markerSSCT <- function(r, y, params) {
  P <- function(x, r, y, a) exp(x * sum(a*r*y))/(exp(-1 * sum(a*r*y)) + exp(1 * sum(a*r*y)))
  # a <- as.integer(runif(3)*2)*2-1 #+1/-1
  # r <- as.integer(runif(3)*2)
  # y <- as.integer(runif(3)*2)*2-1
  u <- runif(1)
  if (u < P(1,r,y,params)){
    m <- 1
  }
  else{
    m <- -1
  }
  return (m)
}


intensity_ssct <- function(a,r,y) exp(sum(a*r*y))
pointSSCT <- function(r, y, params) {
  
  lambda <- intensity_ssct(params,r,y)#/(exp(-1 * sum(a*r*y)) + exp(1 * sum(a*r*y)))
  print(c(params, r, y, lambda))
  # a <- as.integer(runif(3)*2)*2-1 #+1/-1
  # r <- as.integer(runif(3)*2)
  # y <- as.integer(runif(3)*2)*2-1
  d <- rexp(1, lambda)
  return (d)
}

genSSCT <- function(T, params) {
  r_seq <- as.integer(runif(3)*2)
  y_seq <- as.integer(runif(3)*2)*2-1
  m <- 3
  t <- 0
  t_seq <- c()
  while (t < T) {
    tn <- pointSSCT(tail(r_seq, m), tail(y_seq, m), params)
    t <- t + tn
    # print(t)
    rn <- as.integer(t%%24 < 12)
    print(rn)
    yn <- markerSSCT(tail(r_seq, m), tail(y_seq, m), params)
    t_seq <- c(t_seq, t)
    r_seq <- c(r_seq, rn)
    y_seq <- c(y_seq, yn)
  }
  return(list(t_seq, y_seq))
}

genHPP <- function(T, params) {
  t <- 0
  P <- c()
  while (t < T) {
    e <- rexp(1, rate = params[1])
    t <- t + e
    P <- c(P, t)
  }
  return (P)
}

paramhpp <- c(.7)
datahpp <- genHPP(100, paramhpp)
plot(seq(1,100), rep(paramhpp[1], 100), type='l', ylim=c(0, paramhpp[1]+1))
par()
points(datahpp, rep_len(0.1, length(datahpp)))

paramnhpp <- c(.1)
datanhpp <- genNHPP(100, intensity_nhpp, paramnhpp)
plot(seq(1,100,.5), sapply(seq(1,100,.5), function(x) intensity_nhpp(x, paramnhpp)), type='l')
par()
points(datanhpp, rep_len(0.1, length(datanhpp)))

paramscp <- c(.5, .2)
datascp <- genSCP(100, intensity_scp, paramscp)
plot(seq(1,100,.5), sapply(seq(1,100,.5), function(x) intensity_scp(x, datascp, paramscp)), type='l')
par()
points(datascp, rep_len(1, length(datascp)))

paramshp <- c(.9, .1, 1.0)
datahp <- genHP(100, intensity_hp, paramshp)
plot(seq(1,100,.5), sapply(seq(1,100,.5), function(x) intensity_hp(x, datahp, paramshp)), type='l')
par()
points(datahp, rep_len(1, length(datahp)))

paramsssct <- c(-.2, 0.8, -.8)
datassct <- genSSCT(100, paramsssct)
plot(datassct[[1]][2:length(datassct[[1]])], diff(datassct[[1]]), type='l')
par()
points(datassct[[1]], rep_len(.1, length(datassct[[1]])), col=datassct[[2]]+2)
