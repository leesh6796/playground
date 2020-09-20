############################
# 1.Liner and Scatter plots
############################

#(1)
?faithful
head(faithful)
str(faithful)
#sorting faithful data.frame by waiting
fa <- faithful[order(faithful$waiting),]
head(fa)

order(faithful$waiting)
sort(faithful$waiting)
faithful$waiting[149]

x <- fa[,2]; y <- fa[,1]

fa[1,]
fa[,1]

plot(x,y)
plot(x,y,type='b')
plot(x,y,type='l',col=4)
plot(x, y, type='l', col=4, xlab='Waiting Time (min)', 
     ylab='Eruption Time (min)', main='Old Faithful Eruptions')
points(x,y,pch=20,col=3)
points(x,y,pch=20,col=2)


#(2)
library(ggplot2)
ggplot(fa, aes(x, y), xtitle='Waiting Time (min)') + 
  geom_point(col=3) + geom_line(col=4) +
  xlab('Waiting Time (min)') + 
  ylab('Eruption Time (min)') +
  ggtitle('Old Faithful Eruptions')

ggplot(fa, aes(x, y), xtitle='Waiting Time (min)') + 
  geom_point(col=3) + geom_line(col=4) +
  xlab('Waiting Time (min)') + 
  ylab('Eruption Time (min)') +
  ggtitle('Old Faithful Eruptions') +
  theme_bw()


#(3)
head(diamonds)
str(diamonds)
ggplot(data=diamonds,aes(x=carat,y=price)) +
  geom_point(aes(color=color)) 

ggplot(data=diamonds,aes(x=carat,y=price)) +
  geom_point(aes(color=color)) +
  facet_grid(color ~ .)

ggplot(data=diamonds,aes(x=carat,y=price)) +
  geom_point(aes(color=color)) +
  facet_grid(. ~ color)

ggplot(data=diamonds,aes(x=carat,y=price)) +
  geom_point(aes(color=color)) +
  facet_wrap(.~color)


############################
# 2.Bar Plots
############################

#(1) simple Barplot 
head(cars)
str(cars)
barplot(cars)
barplot(cars[,2])
barplot(cars[,2],col="cornsilk",
        names.arg=cars[,1],
        xlab="Speed",
        ylab="Stopping Distance")

#(2) Grouped Barplot
attach(sleep)
sleep #list data
str(sleep)
y <- rbind(extra[1:10],extra[11:20])
y
barplot(y)
?barplot
barplot(y, beside=T, col=5:6, names.arg=ID[1:10],
        xlab="ID",ylab="Extra Sleep Hour")
abline(h=0)
abline(v=1)
legend('topleft', title='group', legend=1:2, fill=5:6)

#(3) visualization in ggplot2
library(ggplot2)
ggplot(sleep, aes(x=ID, y=extra, fill=group)) + 
  geom_bar(stat="identity",position="dodge") + 
  theme_bw()

#(4) bar plot of matrix/table data
UPE <- USPersonalExpenditure
UPE
win.graph(8,5)
barplot(UPE, beside=T, col=2:6, ylab="Expenditure (B$)",
        xlab='Year', main='United States Personal Expenditures') 
legend('topleft',legend=row.names(UPE), fill=2:6)
legend(1,20,legend=row.names(UPE), fill=2:6)
#legend('topright',inset=c(-0.5,0),legend=row.names(UPE), fill=2:6)


############################
# 3.Pie Chart
############################

#(1) Simple pie chart
#sample data
Age <- c('<=24','25-34','35-44','45-54','55-64','65-74','>=75')
Obese <- c(4.9,7.7,11.6,16.8,23.7,23.3,11.9)
#plot pie chart
pie(Obese, labels=paste(Obese,'%'), main='Obesity Percents by Age Group',
    col=rainbow(length(Age)))
legend("topleft", Age, cex=0.8, fill=rainbow(length(Age)))


#(2) 3D Pie Chart
library(plotrix)
xb <- paste(Age,"\n",Obese,'%',sep="")
pie3D(Obese, labels=xb, explode=0.1,
      col=rainbow(length(Age)), 
      main="3D Pie Chart of Obesity Percents by Age Group")


############################
# 4.Histogram
############################

#sample data: cane 
library(boot)
dim(cane)
str(cane)
head(cane)


#(1) Simple Histogram 
ratio <- cane$r/cane$n
hist(ratio, breaks=20, xlab='Diseased Shoot Ratio', col='aquamarine',
     main='Histogram of Diseased Shoot Ratio')


#(2)Histograms of Diseased Shoot Ratio by block
library(ggplot2)
ggplot(cane, aes(x=r/n, fill=block)) +
  geom_histogram(colour="black") + theme_bw()

ggplot(cane, aes(x=r/n, fill=block)) +
  geom_histogram(colour="black") + theme_bw() +
  facet_grid(block ~ .) + xlab('Diseased Shoot Ratio') +
  theme(legend.position="none")          


############################
# 5.Bubble Plot
############################

attach(USArrests)
head(USArrests)
summary(USArrests)

win.graph(6,6)

#(1)
radius <- Rape/max(Rape) #circle size
N <- nrow(USArrests)
op <- palette(rainbow(N))
op
symbols(Murder, Assault, circles=radius)
%symbols(Murder, Assault, circles=radius,
         inches=F)
symbols(Murder, Assault, circles=radius,
        inches=0.25)
symbols(Murder, Assault, circles=radius,
        inches=0.25, fg='black', bg=1:N,
        xlab='Murder (/100,000)', 
        ylab='Assault (/100,000)',
        main='Circle shows Rape (7.3~46)')
text(Murder, Assault, row.names(USArrests),
     cex=0.8)
palette(op)


#(2)
library(ggplot2)
ggplot(USArrests, aes(Murder,Assault)) +
  geom_point(colour="magenta")

ggplot(USArrests, aes(Murder,Assault,size=Rape,label=row.names(USArrests))) +
  geom_point(colour="magenta")  + geom_text(size=3) + theme_bw() +
  xlab("Murders (/100,000)") + ylab("Assault (/100,000)")


############################
# 6.Box Plot
############################

#sample data: ToothGrowth
head(ToothGrowth)


#(1) Simple boxplot
boxplot(len~supp, data=ToothGrowth, 
        xlab="Supplement Type",ylab="Tooth Length")


#(2) Boxplot of len against dose and supp factors
boxplot(len~supp*dose, data=ToothGrowth)

boxplot(len~supp*dose, data=ToothGrowth, notch=T,
        xlab="Suppliment and Dose",ylab="Tooth Length",
        col=c("cyan","magenta"))

boxplot(len~dose*supp, data=ToothGrowth, notch=T,
        xlab="Suppliment and Dose",ylab="Tooth Length",
        col=c("cyan","magenta"))


#(3) boxplot in ggplot2
library(ggplot2)
ggplot(ToothGrowth, aes(x=factor(dose), y=len)) +
  geom_boxplot(aes(fill=supp)) +
  xlab("dose") + ylab("length") +
  ggtitle("Analyzing ToothGrowth Data") +
  theme_bw()


############################
# 7.3D Plots
############################

# Sample data: iris
win.graph(5.4,4.8)


#(1)  3D Scatter Plot
library(scatterplot3d) 
s3d <- scatterplot3d(iris[,1:3], color=c(2:4)[iris$Species],
                     col.axis="blue", col.grid="lightblue", pch=16, cex.symbols=1)
legend(s3d$xyz.convert(2.5,3.1,8.8), legend=levels(iris$Species),
       col=2:4, pch=16, bty="n", title="Species")


#(2) 3D Wireframe Plot
library(lattice)
x <- seq(-3, 3, .2); y=x
dat <- expand.grid(x,y)
dat$z <- dnorm(dat[,1])*dnorm(dat[,2])
names(dat) <- c('x','y','z')
wireframe(z ~ x*y, data=dat,  scales=list(arrows=FALSE),
          aspect=c(1,.6), drape=TRUE,
          par.settings=list(axis.line=list(col='transparent')))


############################
# 8.3D Scatter Plots
############################

# Example 1
# 3D Scatterplot with Coloring and Vertical Drop Lines
library(scatterplot3d) 
attach(mtcars) 
scatterplot3d(wt,disp,mpg, pch=16, highlight.3d=TRUE,
              col.grid="lightblue", type="h", main="3D Scatterplot")


# Example 2
# Prepare the data
data(iris); head(iris,4)
library(scatterplot3d) 
s3d <- scatterplot3d(iris[,1:3],color=c("red","green","blue")[iris$Species],
                     col.axis="blue",col.grid="lightblue",
                     main="3D Scatterplot of Iris Data",pch=16,cex.symbols=1)
legend(s3d$xyz.convert(2.3,3.1,9.3), legend=levels(iris$Species),
       col=c("red","green","blue"), pch=16, bty="n")


# lattice wireframe plot
library(lattice)
x <- seq(-3, 3, .2); y=x
dat <- expand.grid(x,y)
dat$z <- dnorm(dat[,1]) * dnorm(dat[,2])
names(data) <- c("x","y","z")

wireframe(z ~ x*y, data,  scales=list(arrows=FALSE),
          aspect=c(1,.6), drape=TRUE,
          par.settings=list(axis.line=list(col='transparent')))


library(rgl)
open3d(windowRect=c(50,50,600,600))

x <- y <- seq(-4, 4, length=50)
z <- outer(x,y, function(x,y) sin(sqrt(x^2+y^2)) )

palette <- colorRampPalette(c("blue","cyan","green","yellow","orange","red")) 
col.table <- palette(256)
col.ind <- cut(z, 256)
persp3d(x, y, z, col=col.table[col.ind])



## https://fas-web.sunderland.ac.uk/~cs0her/Statistics/UsingLatticeGraphicsInR.htm
# volcano  ## 87 x 61 matrix
library(lattice)
head(volcano)
wireframe(volcano, shade=TRUE, 
          aspect=c(61/87,0.4),
          light.source=c(10,0,10))


## http://www.r-bloggers.com/r-fun-with-surf3d-function/
# 3D Plot of Half of a Torus
library(plot3D)
R=3; r=2
M <- mesh(x=seq(0,2*pi,length.out=50),y=seq(0,2*pi,length.out=50))
alpha <- M$x; beta <- M$y
surf3D(x = (R + r*cos(alpha))*cos(beta),
       y = (R + r*cos(alpha))*sin(beta),
       z = r*sin(alpha), colkey=TRUE, bty="b2", main="Half of a Torus",
       phi=40, theta=30, shade=0.2)



z <- outer(x,y, function(x,y) sin(sqrt(x^2+y^2)))
persp(x,y,z)

jet.colors <- colorRampPalette( c("blue", "green") ) 
pal <- jet.colors(100)
col.ind <- cut(z,100) # colour indices of each point
persp3d(x,y,z,col=pal[col.ind])


scatterplo3D(x, y, z, theta=30, phi=20, pch=20, cex=2, ticktype="detailed", 
             bty="b2", main="Iris Data", xlab="Sepal.Length",
             ylab="Petal.Length", zlab="Sepal.Width")


## http://www.sthda.com/english/wiki/impressive-package-for-3d-and-4d-graph-r-software-and-data-visualization
# x, y and z coordinates
x <- iris$Sepal.Length
y <- iris$Petal.Length
z <- iris$Sepal.Width



library(animation)

saveGIF({
  par(mai = c(0.1,0.1,0.1,0.1))
  for(i in 1:100){
    X <- seq(0, 2*pi, length.out = 100)
    Y <- seq(-15, 6, length.out = 100)
    M <- mesh(X, Y)
    u <- M$x
    v <- M$y
    
    # x, y and z grids
    x <- (1.16 ^ v) * cos(v) * (1 + cos(u))
    y <- (-1.16 ^ v) * sin(v) * (1 + cos(u))
    z <- (-2 * 1.16 ^ v) * (1 + sin(u))
    
    # full colored image
    surf3D(x, y, z, colvar = z, 
           col = ramp.col(col = c("red", "red", "orange"), n = 100),
           colkey = FALSE, shade = 0.5, expand = 1.2, box = FALSE, 
           phi = 35, theta = i, lighting = TRUE, ltheta = 560,
           lphi = -i)
  }
}, interval = 0.1, ani.width = 550, ani.height = 350)


############################
# 9.Ridge line plot
############################

#Example 1
library(ggridges)
library(ggplot2)


#sample data : diamonds{ggplot2}
head(diamonds)


#ridgelineplot
ggplot(diamonds, aes(x=price,y=cut,fill=cut)) +
  geom_density_ridges() +
  theme_ridges() +
  theme(legend.position="none")


#Example 2
library(ggridges)
library(ggplot2)
ggplot(data = iris, aes(y=Species, x=Sepal.Width, fill=Species)) +
  geom_density_ridges(alpha=0.7, scale=1.2, stat = "binline",  bins=25) +
  scale_y_discrete(expand = c(0.01, 0)) +
  scale_x_continuous(expand = c(0.01, 0)) +
  theme_ridges(grid = F) +
  theme(legend.position = "none")


#Example 3
library(ggplot2)
library(ggplot2movies)
head(movies,2)
ggplot(data = movies, aes(y=year, x=length, group=year, fill=year)) + 
  geom_density_ridges(alpha=0.5, scale=10) +
  scale_x_log10(limits=c(1,500),expand = c(0.01, 0)) + 
  scale_y_reverse(breaks=(seq(1900,2000,20))) +
  theme_ridges() +
  theme(legend.position = "none") + 
  scale_fill_distiller(palette = "Spectral") +
  labs(title="Movies Veriseti, 2017")

# The End of Data Visualization.

