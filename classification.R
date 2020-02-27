

## Classification
## Random Forest classification
### ARtvin Coruh University Engineering Faculty
### Geomatics Department

# Installing libs ---------------------------------------------------------
## Installing libs
lapply(names(sessionInfo()$otherPkgs), function(pkgs)
  detach(
    paste0('package:', pkgs),
    character.only = T,
    unload = T,
    force = T
  ))

library(RANN)
library(rgl)
library(matlab)
library(lidR)
library(pracma)
library(beepr)


tic()
#change path
setwd("E:/hazirlik/R/")


# Read Data ---------------------------------------------------------------

#change path
data1 <- readLAS(files = "/TRDizin/hazirlik/TBC_classification/classification_1m.las")
sprintf("Las verisi okundu", toc())

plot(data1, color= "Classification")
data <- cbind(data1@data$X,data1@data$Y,data1@data$Z)

# Neighboorhood
k <- 10
nearest <- nn2(data = data,k=k+1)
nn <- nearest[["nn.idx"]]
nn <- nn[,-1]
nn.dist <- nearest[["nn.dists"]]
nn.dist <- nn.dist[,-1]
pp <- as.matrix(data)

r_mat <- kronecker(matrix(1,k,1),pp[,1:3])
p_mat <- pp[nn,1:3]

p <-  p_mat - r_mat;
p <- matlab::reshape(p, size(pp,1), k, 3);


C <-  matrix(0,dim(data),6)
C[,1] <- rowSums(p[,,1]*p[,,1])
C[,2] <- rowSums(p[,,1]*p[,,2])
C[,3] <- rowSums(p[,,1]*p[,,3])
C[,4] <- rowSums(p[,,2]*p[,,2])
C[,5] <- rowSums(p[,,2]*p[,,3])
C[,6] <- rowSums(p[,,3]*p[,,3])
C <- C/k


# Compute features --------------------------------------------------------


###### normals and curvature calculation
normals <-  matrix(0,dim(pp),3)
curvature <-  matrix(0,dim(pp))
omnivariance <-  matrix(0,dim(pp))
planarity <-  matrix(0,dim(pp))
linearity <-  matrix(0,dim(pp))
surf_var <-  matrix(0,dim(pp))
anisotropy <-  matrix(0,dim(pp))

for (i in 1:nrow(pp)) {
  #Covariance
  Cmat <- matrix( c(C[i,1], C[i,2], C[i,3],
                    C[i,2], C[i,4], C[i,5],
                    C[i,3], C[i,5], C[i,6]), byrow = TRUE,ncol = 3)  
  
  #Eigen values and vectors
  
  d <- eigen(Cmat)$values
  v <- eigen(Cmat)$vectors
  lambda  <- min(d);
  ind <- which(d==min(d))
  
  #store normals
  normals[i,] = t(v[,ind])
  
  #store curvature
  curvature[i] = lambda / sum(d);
  
  #store omni
  omnivariance[i] = (d[1]*d[2]*d[3])^(1/3)
  #store planarity
  planarity[i] = (d[2]-d[3])/d[1]
  #store linearity
  
  linearity[i] = (d[1]-d[2])/d[1] 
  #store surf_var
  
  surf_var[i] = d[3]/(d[1]+d[2]+d[3])
  #store anisotropy
  
  anisotropy[i] = (d[1]-d[3])/d[1]
}

sprintf("Eigenvalues hesaplandi", toc())

###### Flipping normals


#%% flipping normals
#%ensure normals point towards viewPoint
#viewPoint <- matrix(c(0,0,0),ncol = 3)
#pp[,1:3] = pp[,1:3] - kronecker(matrix(1,size(pp,1),1),viewPoint)
dist_nn <- nn.dist[,1]
range(data1@data$R)
bit8 <- function(x){
  y <- round((x/255)-1)
  return(y)
}
data1@data$R2 <- bit8(data1@data$R)
data1@data$G2 <- bit8(data1@data$G)
data1@data$B2 <- bit8(data1@data$B)

write.csv(cbind(pp,normals,data1@data$R2, data1@data$G2, data1@data$B2, 
                curvature,omnivariance,planarity,linearity,surf_var,anisotropy),file = "pcl_normals.txt")
sprintf("Eigenvalues dosyasi export edildi", toc())



#Delete all variables
#rm(list = ls())

#change path
#data1 <- readLAS(files = "/hazirlik/TBC_classification/classification/classification.las")
#sprintf("Las verisi okundu", toc())
dtm <- grid_terrain(data1,  algorithm = knnidw(k = 6L, p = 2),res = 1)
las <- lasnormalize(data1, dtm)
plot(las)
writeLAS(las, file = "normalized.las")


# Machine Learning --------------------------------------------------------


###### Machine Learning Step ##########
library(caret)
library(ellipse)
library(e1071)
library(kernlab)
library(randomForest)
library(rgl)
library(spatstat)
library(parallel)
library(doParallel)

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
#veri2 <- read.csv(file = "pcl_normals.txt")
#veri2 <- cbind(pp,normals,curvature,omnivariance,planarity,linearity,surf_var,anisotropy,dist_nn)
veri2 <- cbind(data1@data$X,data1@data$Y,data1@data$Z,normals[,1],normals[,2],normals[,3],
               data1@data$R2,data1@data$G2,data1@data$B2,curvature, omnivariance, planarity,
               linearity, surf_var, anisotropy, las@data$Z, data1@data$Classification)
#veri2 <- cbind(veri2, las@data$Z, las@data$Classification)
colnames(veri2) <- c("X","Y","Z", "nx","ny", "nz","R","G","B", "Curvature",
                     "Omnivariance","Planarity", "Linearity", "Surface_Variance","Anisotropy","AGL","Class")

veri2 <- as.data.frame(veri2)
#SAmple data
veri2<- veri2[sample(nrow(veri2), 20000), ]

#veri2 <- as.data.frame(veri2)
#### Validation ###
validation_index <- createDataPartition(veri2$Class, p=0.7,list = FALSE)
validation <- veri2[-validation_index,]
validation$Class <- factor(validation$Class)
dataset <- veri2[validation_index,]

####### Dimensions of the dataset #####
dataset$Class <- factor(dataset$Class)
dim(dataset)
sapply(dataset, class)
head(dataset)
levels(dataset$Class)

####### Building Models #######

control <- trainControl(method = "repeatedcv",number = 10,repeats=3,search='grid', allowParallel = TRUE)
metric <- "Accuracy"
tunegrid <- expand.grid(.mtry = (1:13)) 
set.seed(7)
fit.rf <- train(Class~ nx+ny+nz+R+G+B+Curvature+Omnivariance+Planarity+Linearity+Surface_Variance+Anisotropy+AGL,data = na.omit(dataset), method = "rf",
               metric = metric, trControl = control,tuneGrid = tunegrid)
#fit.rf <- train(Class~ nz+Omnivariance + Planarity, data = na.omit(dataset), method = "rf",
#                metric = metric, trControl = control)
stopCluster(cluster)
registerDoSEQ()
plot(fit.rf$finalModel)
summary(fit.rf)
print(fit.rf)
sprintf("Train islemi bitti", toc())

##### Make Predictions #######
predictions <- predict(fit.rf, validation)
result <- confusionMatrix(predictions, validation$Class, mode = "everything")
result
saveRDS(fit.rf, "./finalModel.rds")


##### Predict New Data ######
superModel <- readRDS("./finalModel.rds")
print(superModel)

# New Data Import ---------------------------------------------------------

veri2 <-  read.csv(file = "pcl_normals.txt")
veri2 <- cbind(veri2,las@data$Z)

colnames(veri2) <- c("ID","X","Y","Z", "nx","ny", "nz", "R","G","B","Curvature",
                     "Omnivariance","Planarity", "Linearity", "Surface_Variance","Anisotropy", "AGL")
veri2$Class <- 0

finalPredictions <- predict(superModel, veri2)
#confusionMatrix(finalPredictions, veri2$Class)
export<-veri2
export$Class <- finalPredictions
export$Class <- as.numeric(as.character(export$Class))
export$ID <- NULL
write.csv(export, "./finalPCL.txt")

beep()
sprintf("Program bitti", toc())


# Accuracy Assessments ----------------------------------------------------


#### Variable importance in RBF

library(ggplot2)

superModel <- readRDS("finalModel.rds")
print(superModel)

var <- varImp(superModel, scale = TRUE)
print(var)
aa <- var[["importance"]]
aa
x <- rownames(aa)
y <- aa$Overall
par(mar=c(5,4,4,2)+1)
p<-ggplot(data=aa, aes(x=x, y=y)) +
  geom_bar(stat="identity")+
  theme(text = element_text(size = 28))+
  xlab("Features")+ ylab("Variable Importance (%)")
p

plot(var, top = 13, xlab =" Önem Derecesi (%)", ylab= "Özellik")


