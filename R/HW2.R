# Exercise 1
x=factor(c(1,2,3,4,5,2,4,3,5,1,2,3,4,5,1,2))
y=c("Red", "Green", "Blue", "Magenta")
y[x]
# Sol)
# [1] "Red"     "Green"   "Blue"    "Magenta" NA       
# [6] "Green"   "Magenta" "Blue"    NA        "Red"    
# [11] "Green"   "Blue"    "Magenta" NA        "Red"    
# [16] "Green"


# Exercise 2
A = matrix(c(1,2,3,0,1,4,5,2,4), nrow=3, byrow=TRUE)
B = matrix(c(2,3,0,-1,2,5,3,9,2), nrow=3, byrow=TRUE)
C = A %*% B; C
# Sol)
#     [,1] [,2] [,3]
#[1,]    9   34   16
#[2,]   11   38   13
#[3,]   20   55   18


# Exercise 3
# (1)
head(state.x77)
class(state.x77) # "matrix" "array -> it isn't data frame
is.data.frame(state.x77) # False
df = as.data.frame(state.x77) # convert

# (2)
df_low_income = subset(df, Income < 4000) # filtering with cond
nrow(df_low_income) # 13 states have an income of less than 4000

# (3)
rownames(subset(df, Income==max(df$Income))) # Alaska