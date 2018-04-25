# Fichier pour entreposer le code hors du markdown. C'est plus pratique pour la partie algo
# On scinde en autant de partie que de méthodes à mettre en oeuvre

#Initialisation des packages ----
rm(list =ls())
install.packages("neuralnet")
library(neuralnet)
library(car)
library(caret)
library(nnet)
library(e1071)
library("xts")
library("sp")
library("zoo")
library(MASS)
library(magrittr)
library(rpart)
library(rpart.plot)

install.packages("CASdatasets", repos = "http://dutangc.free.fr/pub/RRepos/", type="source")
library("CASdatasets")


data("freMTPL2freq")
data("freMTPL2sev")

frequence <- freMTPL2freq
severite <- freMTPL2sev

rm(freMTPL2freq, freMTPL2sev)

#Mise en forme des donnéees ----


#On formate les données de la base de frequence
#frequence$ClaimNb <- frequence$ClaimNb %>% unname() %>% as.numeric()

frequence$VehPower <- as.factor(frequence$VehPower)
frequence$Exposure <- as.double(frequence$Exposure)
frequence$Area <- as.factor(frequence$Area)
frequence$VehAge <- as.integer(frequence$VehAge)
frequence$DrivAge <- as.integer(frequence$DrivAge)
frequence$BonusMalus <- as.integer(frequence$BonusMalus)
frequence$VehBrand <- as.factor(frequence$VehBrand)
frequence$VehGas <- as.factor(frequence$VehGas)
frequence$Region <- as.factor(frequence$Region)
frequence$ClaimNb<- as.numeric(frequence$ClaimNb)
frequence$Density=as.numeric(frequence$Density)

nrow(frequence)#678 010

#On formate la base de cout
severite$IDpol <- as.integer(severite$IDpol)
severite$ClaimAmount <- as.numeric(severite$ClaimAmount)

#On prend que les exposures inférieures à 1 ?
frequence <- frequence[frequence$Exposure <= 1,]
nrow(frequence)# 676 789
#Nos bases
head(frequence)
head(severite)


severite.mean <- aggregate(ClaimAmount ~ IDpol, data = severite, mean)
names(severite.mean) <- c("IDpol", "MeanClaimAmount")
base.mean <- merge(x = frequence, y = severite.mean, by = "IDpol", all.x = T)
base.mean$MeanClaimAmount <- replace(base.mean$MeanClaimAmount, is.na(base.mean$MeanClaimAmount), 0)


head(base.mean,5)



#On enlève les polices sinistrées avec un montant moyen de sinistres nul
base.mean <- base.mean[-which(base.mean$ClaimNb > 0 & base.mean$MeanClaimAmount ==0),]

nrow(base.mean)#667 673

summary(base.mean)

# Create DAta partition: Base apprentissage et Base Test ( Caret package) un problème avec le package! 
#j'ai réparti la base avec la méthode traditionnelle
#75% de data en apprentissage et 25% test

#trainIndex <- createDataPartition(base.mean, p=.75, list=F)
#train <- base.mean[trainIndex, ]
#test <- base.mean[-trainIndex, ]

#size <- floor(0.75 * nrow(base.mean))
size <- floor(0.01 * nrow(base.mean)) #pour faire plus rapide 
set.seed(100)
train_ind <- sample(seq_len(nrow(base.mean)), size =size)
train <- base.mean[train_ind, ]
test <- base.mean[-train_ind, ]

summary(train)
summary(test)

nrow(test)+nrow(train);nrow(base.mean)

                                         # Avec décomposition 

# Dans cet algo, on cherchera à détermine le paramètre  le plus important:le nombre de neurones
#sur la couche cachée parallèlement aux conditions d’apprentissage (temps ou nombre de boucles) 

# A noter que l'alternative pour déterminer le nombre de neurones est celle du decay: paramètre de régularisation  
 

                                        #Mod�le de fréquence
# R�gression

# On Fitte un neural network avec la base d'apprentissage: La meilleure méhotde pour déterminer le nombre de layers 
#et le nombre de neurones
n <- names(train)
mygrid <- expand.grid(size=c(1,2,3,4,5,6,7,8),decay=seq(1,5),KEEP.OUT.ATTRS = TRUE, stringsAsFactors = TRUE)
#as.formula nous permet de voir au plus clair sur les variables prise en compte pour les fit

varb <- as.formula(paste(" ClaimNb", paste(c("DrivAge","VehAge","VehPower","VehBrand","VehGas","BonusMalus","Area","Region","Density"), collapse = " + "),sep=" ~ "))

# Trainning net avec les poids

#train$ClaimNb = as.factor(train$ClaimNb) sur stackoverflow.com, on recommande que le y doit �tre un factor mais n'emp�che

str(train) # v�rific des types de variables avant le fit

ctrl    <- trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        verboseIter = T,
                        returnResamp = "all")
train.fit = train(varb ,data=train,method = "nnet",tuneGrid = mygrid ,trace=F, trControl =ctrl, weights=train$Exposure)
summary(train.fit)
plot(train.fit) # Figure 1
train.fit$resample
#RMSE was used to select the optimal model using the smallest value.
#The final values used for the model were size = 7 and decay = 1.

nnet=tune.nnet(varb,data = train, weights =train$Exposure, size=seq(1,7), decay=seq(1,5), maxit=1000,linout=TRUE)
plot(nnet)
