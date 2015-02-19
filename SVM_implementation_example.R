# created on 2/16/15
# function: step by step tutorial of SVM calculation
# input: 1D vector and corresponding labels
# output: alpha values for SVM classification 

###########################################################

# Step 1: inspect data 
# given five data points and their corresponding classes
# how do you calculate a set of alpha values that can be used for classification

a <- c(-5)
b <- c(-3)
c <- c(1)
d <- c(2)
e <- c(3)
class <- as.factor(c(-1,-1,1,1,-1))

# visualize data

visualize <- as.data.frame(rbind(a,b,c,d,e))
y <- rep(0,5)
visualize <- cbind(visualize, y, class)
visualize
library(ggplot2)
p <- ggplot(visualize, aes(V1, y)) + geom_point(aes(color = class))
p 

# data is not linearly separable!
# in other words, you cannot create a linear boundary that can discriminate

# step 2: map to higher dimension, which may yield discriminative features!

transform <- function(input){
  output <- c(input[1], input[1]^2) # additional dimension is simply x^2
  output
}

# visualize transformed data

data_1 <- list(a,b,c,d,e) 
data_2 <- lapply(data_1, transform)
data_3 <- as.data.frame(matrix(unlist(data_2), nrow = length(data_2), byrow = TRUE))
data_3 <- cbind(data_3, class)
p_2 <- ggplot(data_3, aes(x = V1, y = V2)) + geom_point(aes(color = class))
p_2 

# transforming from 1D to 2D creates linearly separable data!
# notice support vectors are points D and E

# step 3: generate alpha values from support vectors

sv_1 <- list(d,e) 
sv_2 <- lapply(sv_1, transform)
sv_2
class_sv <- c(1, -1)

# step 4: create kernel matrix

n <- length(sv_2)
temp_matrix <- c()
for (i in 1:n){
  for (j in 1:n){
    dot_product <- sv_2[[j]] %*% sv_2[[i]]
    temp_matrix <- c(temp_matrix, dot_product)
  }
}

feature_matrix <- matrix(temp_matrix, nrow=n, ncol = n)
feature_matrix
X <- feature_matrix
Xt <- t(X)
X1 <- Xt %*% X
X2 <- solve(X1)
X3 = X2 %*% Xt
y = matrix(class_sv, nrow = n, ncol = 1)
alpha = X3 %*% y
alpha_values <- as.vector(alpha)
alpha_values # use these alpha values to classify!

# step 5: classify!
# let's classify point Z, which has a value of -5
# we know it should be in the "-1" class

z <- c(-5)
z_transformed <- transform(z) # need to transform Z vector to the right dimension
z_transformed # z went from 1D vector to 2D 

calc <- c()
for (i in 1:n){
  # n is the number of support vectors
  calc <- cbind(calc, alpha_values[i] * sv_2[[i]] %*% z_transformed)  
}
calc

sign(sum(calc)) # success! a value of -5 is classified in the negative class

# lets classify another vector with value 1.5
# visually, we know it should be in the positive class

z <- c(1.5)
z_transformed <- transform(z) # need to transform Z vector to the right dimension
z_transformed # z went from 1D vector to 2D 

calc <- c()
for (i in 1:n){
  # n is the number of support vectors
  calc <- cbind(calc, alpha_values[i] * sv_2[[i]] %*% z_transformed)  
}
calc

sign(sum(calc)) # success! a value of 1.5 is classified in the positive class
