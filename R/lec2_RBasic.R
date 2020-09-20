setwd("C:/Users/leesh/Desktop/Prog/R") # set working directory
getwd()

x = 1:10 # [1, 2, ..., 10]
mean(x)

help(mean) # getting a manual
?mean

x<- 100; x
k <<- c("Mary", "Jackson"); k

567 %% 2 # remainder

log(3) # natural log
log(1000, 5)
factorial(100)
atan(1) # arctan

plot(USArrests)

options(digits=2)
cor(USArrests) # correlation

head(USArrests) # 위에서부터 일부만 보여준다

# 섭씨를 화씨로 변환
C = seq(0, 100, 10) # 0부터 100까지 10단위로
F <- C * 9/5 + 32
T <- data.frame(Celsius=C, Fahrenheit=F)

ls() # To see names of all objects in my workspace
save.image() # save contents of workspace, into the file ".RData"
save.image(file="myscript.RData")

seq(0, 120, 20)
# 0 20 40 60 80 100 120
rep(1:5, times=2)
# 1 2 3 4 5 1 2 3 4 5
rep(1:2, each=3)
# 1 1 1 2 2 2

x1 <- c(3,1,4,15,92)
rank(x1) 
x2 <- c(3,1,4,6,5,9)
rev(x2) # reverse
x3 <- c(3:5, 11:8, 8+0:5)
unique(x3)

A <- matrix(c(9,8,8, 2,5,10, 9,10,3, 8,8,4), nrow=4, byrow=TRUE)
A
colnames(A) <- c("Plan", "Analysis", "Writing")
rownames(A) <- c("S1", "S2", "S3", "S4")
A

barplot(A, beside=TRUE, legend=TRUE, col=1:4)


# 배열의 차원은 앞에서부터 1,2,3차원이다.
arr1 <- array(1:18, dim=c(3,3,2))
arr1
dim(arr1)

str(iris)
class(iris) # "data.frame"

install.packages("dplyr") # download and install a package from CRAN
library(dplyr) # load a package into the session
dplyr::select # Use a particular function from a package
data(iris) # load a built-in dataset into the env

for(variable in sequence)
{
  Do something
}
while(condition)
{
  Do something
}
function_name <- function(var) {
  Do something
  return (new_variable)
}

lm(x~y, data=df) # linear model