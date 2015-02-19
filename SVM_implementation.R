# 2/14/15
# walks through svm training and classification
# svm finds a set of alpha values, which are used as a reference "image" for classification
# similarity is calculated via dot product (may be in higher dimensions)

###########################################################
# LINEAR SVM IMPLEMENTATION

# step 1: select n support vectors of k-dimensions
# decision boundary will be a hyperplane of k-1 dimensions
# include a bias of 1 for each vector (so (1,0) becomes (1,0,1))
# include corresponding labels as a separate vector

x <- c(1,0,1) # lets assume there are 3 support vectors
y <- c(3,1,1)
z <- c(3,-1,1)
classes <- c(-1,1,1) # these are labels corresponding to support vectors
support_vectors <- list(x, y, z) # create list for downstream looping

# step 2: create feature matrix to solve for alpha values
# feature matrix will be n x n matrix 
# elements are dot products of vectors

n <- length(support_vectors)
temp_matrix <- c()
for (i in 1:n){
  for (j in 1:n){
    dot_product <- support_vectors[[j]] %*% support_vectors[[i]]
    temp_matrix <- c(temp_matrix, dot_product)
  }
}
# review feature matrix

feature_matrix <- matrix(temp_matrix, nrow=n, ncol = n)
feature_matrix # should be nxn of dot products

# step 3: use normal equation to solve for alpha values
# since you have feature matrix and corresponding labels

X <- feature_matrix
Xt <- t(X)
X1 <- Xt %*% X
X2 <- solve(X1)
X3 = X2 %*% Xt
y = matrix(classes, nrow = n, ncol = 1)
alpha = X3 %*% y
alpha_values <- as.vector(alpha)
alpha_values # weights for future vectors

# step 4: classify!

a <- c(6,-1,1) # lets classify vector A

# svm finds similarity between unknown vector and support vectors via dot product
# since support vectors must be kept in memory, svm is instance-based learner
# for non linear data, vectors may be transformed into higher dimensions for similarity mapping!
# if vectors are mapped from k-dimension to l-dimensions (where L >> K), use kernel trick

calc <- c()
for (i in 1:n){
  calc <- cbind(calc, alpha_values[i] * support_vectors[[i]] %*% a)  
}

sign(sum(calc)) # vector A belongs in the "1" class

###########################################################
# IMPLEMENTATION OF NON-LINEAR SVM (WITH TRANSFORMATION TO SAME DIMENSION)

# step 1: identify support vectors

x <- c(1,1,1)
y <- c(2,2,1)
support_vectors <- list(x, y) 
classes <- c(-1, 1)

# step 2: transform vectors (same dimensions but transformed)
# transformation functions can be anything
# that said, stick with functions that can exploit "kernel tricks"
# kernel tricks allow you to calculate similarities in higher dimensional space
# without incurring the computational costs to get there!

# the following is a made up transformation function
# no kernel trick is used
transform <- function(input){
  if (sqrt(input[1]^2 + input[2]^2) <= 2){
    output <- input
  }
  else {
    output <- c(4 - input[2] + abs(input[1] - input[2]), 4 - input[1] + abs(input[1] - input[2]), 1)
  }
  output
}

# iteratively apply transformation function to list of SVs
support_vectors <- lapply(support_vectors, transform)
support_vectors

# lets look at higher dimensional transformation
x1 <- c(1,4)
x2 <- c(5,6)
x3 <- c(4,1)
x4 <- c(9,3)
x5 <- c(3,7)
x6 <- c(8,8)
x7 <- c(9,1)
x8 <- c(3,5)
x9 <- c(1,7)
x10 <- c(7,7)
sv1 <- list(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
sv1
sv2 <- lapply(sv1, transform)
sv2
sv3 <- as.data.frame(matrix(unlist(sv2), nrow = length(sv2), byrow = TRUE))
sv3
library(scatterplot3d)
with(sv3, {scatterplot3d(V1, V2, V3)})

# step 3: generate feature matrix with nested loop
n <- length(support_vectors)
temp_matrix <- c()
for (i in 1:n){
  for (j in 1:n){
    dot_product <- support_vectors[[j]] %*% support_vectors[[i]]
    temp_matrix <- c(temp_matrix, dot_product)
  }
}
feature_matrix <- matrix(temp_matrix, nrow=n, ncol = n)
feature_matrix

# step 4: solve for alpha values
X <- feature_matrix
Xt <- t(X)
X1 <- Xt %*% X
X2 <- solve(X1)
X3 = X2 %*% Xt
y = matrix(classes, nrow = n, ncol = 1)
alpha = X3 %*% y
alpha_values <- as.vector(alpha)
alpha_values # weights for future vectors

# step 5: classify!

z <- c(4,5,1) 
z_transformed <- transform(z) # need to transform vector Z first!
z_transformed 

calc <- c()
for (i in 1:n){
  calc <- cbind(calc, alpha_values[i] * support_vectors[[i]] %*% z_transformed)  
}
calc

sign(sum(calc)) # so vector (4,5) belongs in the -1 class


###########################################################
# IMPLEMENTATION OF NON-LINEAR SVM (WITH TRANSFORMATION TO HIGHER DIMENSION)
# step 1: identify SVs and corresponding classes 

a <- c(2, 2)
b <- c(2, -2)
c <- c(-2, -2)
d <- c(-2, 2)
e <- c(1,1)
f <- c(1, -1)
g <- c(-1,-1)
h <- c(-1, 1)
support_vectors <- list(a,b,c,d,e,f,g,h) 
classes <- c(1,1,1,1,-1,-1,-1,-1)

# step 2: map to higher dimension, which may yield discriminative features!
transform <- function(input){
  output <- c(input[1], input[2], ((input[1]^2 + input[2]^2) - 5)/3)
  output
}

# step 3: transform SVs using transformation function
# notice we went from 2D to 3D
support_vectors_t <- lapply(support_vectors, transform)
support_vectors_t

# in this case, the newly constructed dimension is a perfectly correlated discriminative feature
