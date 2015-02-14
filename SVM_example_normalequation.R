v1 <- c(3,5) # this is a vector of feature 1
v2 <- c(5,9) # vector of feature 2


X = matrix(c(v1, v2), nrow = 2, ncol = 2)
X
Xt = t(X)
X1 = Xt %*% X
X2 = solve(X1)
X3 = X2 %*% Xt
y = matrix(c(-1,1), nrow = 2, ncol = 1)
beta = X3 %*% y
beta
