# Fichier pour entreposer le code hors du markdown. C'est plus pratique pour la partie algo
# On scinde en autant de partie que de m√©thodes √† mettre en oeuvre

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
library(metrics)
library(devtools)
install.packages("CASdatasets", repos = "http://dutangc.free.fr/pub/RRepos/", type="source")
library("CASdatasets")


data("freMTPL2freq")
data("freMTPL2sev")

frequence <- freMTPL2freq
severite <- freMTPL2sev

rm(freMTPL2freq, freMTPL2sev)

#Mise en forme des donn√©ees 

#On formate les donn√©es de la base de frequence



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

nrow(frequence)#678 013

#  La base sÈvÈritÈ

severite$IDpol <- as.integer(severite$IDpol)
severite$ClaimAmount <- as.numeric(severite$ClaimAmount)

nrow(severite)

#On prend que les exposures inf√©rieures √ 1 ?

frequence <- frequence[frequence$Exposure <= 1,]

nrow(frequence)# 676 789

#Nos bases
head(frequence)
head(severite)

#On merge les deux tables + traitements

severite<- aggregate(severite$ClaimAmount,by = list(severite$IDpol),sum)
colnames(severite) <- c("IDpol","ClaimAmount")
base <- merge(x=frequence,y=severite,by="IDpol",all.x=TRUE)
nrow(base)
base$ClaimAmount <- replace(base$ClaimAmount, which(is.na(base$ClaimAmount)), 0)
base_sans_zero=base
base_sans_zero<- base[-which( base$ClaimAmount ==0),]



#Nous dÈcidons de supprimer les donnÈes manquantes au lieu de les estimer

#base=base[-which(is.na(base$ClaimNb)), ]

nrow(base)
#nous obtenons 676789 obs

##Separation  des attris et des graves 

Seuil_sep<-20000

base_attris <- base[which(base$ClaimAmount < Seuil_sep),]
base_graves <- base[which(base$ClaimAmount >=Seuil_sep),]

nrow(base_attris)+nrow(base_graves)==nrow(base)#TRUE

Am_grave <-sum(base_graves$ClaimAmount)
Am_attri <-sum(base_attris$ClaimAmount)
Am_tot <-sum(base$ClaimAmount)
Am_tot
Am_attri+Am_grave

 # SANS DECOMPOS

#???RÈseaux de neurones appliquÈs au co˚t

set.seed(11)
base_attris2=base_attris
base_attris2<-base_attris2[-which(base_attris2$ClaimAmount==0),]
p=0.7

alea1 = sample(1:dim(base_attris2)[1], ceiling(p*dim(base_attris2)[1]), replace = FALSE)
alea2 = seq(1,dim(base_attris2)[1])
(ligne=6000/length(alea1))
alea2 = alea2[-alea1]

# Echantillon apprentissage attritionnel:
base_app = base_attris2[alea1,]
set.seed(2)
alea3=sample(1:dim(base_app)[1], ceiling(ligne*dim(base_app)[1]), replace = FALSE)
# Echantillon rÈduit pour un gain de temps
echantillon=base_app[alea3,]

# Echantillon test attritionnel:
base_test = base_attris2[alea2,]

#Validation de l echantillon

summary(base_attris2$ClaimNb);summary(base_app$ClaimNb);summary(echantillon$ClaimNb)


                                         # SANS d√©composition 

# Dans cet algo, on cherchera √† d√©termine le param√®tre  le plus important:le nombre de neurones
#sur la couche cach√©e parall√®lement aux conditions d‚Äôapprentissage (temps ou nombre de boucles) 

# A noter que l'alternative pour d√©terminer le nombre de neurones est celle du decay: param√®tre de r√©gularisation  
 

                                       
# RÈgression

#  Pour rÈduire le temps de calcul, On va  Fitter un neural network sur la base rÈduite ( 7000 rows): La meilleure m√©hotde pour d√©terminer le nombre de layers 
#et le nombre de neurones

n <- names(echantillon)

mygrid <- expand.grid(size=c(1,2,3,4,5,6,8),decay=seq(1,6),KEEP.OUT.ATTRS = TRUE, stringsAsFactors = TRUE)

#as.formula nous permet de voir au plus clair sur les variables prise en compte pour les fit

varb <- as.formula(paste(" ClaimAmount", paste(c("DrivAge","VehAge","VehPower","VehBrand","VehGas","BonusMalus","Area","Region","Density"), collapse = " + "),sep=" ~ "))

# Trainning net avec les poids


str(echantillon) # vÈrific des types de variables avant le fit

ctrl    <- trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        verboseIter = T,
                        returnResamp = "all")

train.fit = train(varb ,data=echantillon,method = "nnet",tuneGrid = mygrid ,trace=F, trControl =ctrl)


plot(train.fit) 
train.fit$resample
train.fit$bestTune # ???Size 5 et Decay 2 ( Le modËle choisi a le plus petit RMSE)

#Nous nous contentons de 100 itÈrations
nnet=tune.nnet(varb,data = echantillon, size=seq(1,8), decay=seq(1,5), maxit=100,linout=TRUE)

plot(nnet)


cout_rn = nnet(varb,data = base_app, size=5, decay=2, maxit=100, linout=TRUE)

summary(cout_rn)


plot.nnet(cout_rn)

summary(cout_rn$fitted.values)

summary(base_app$ClaimAmount)

# PrÈdiction cout_rn

prd_rn=predict(cout_rn,base_test)

mse_c=mse(as.numeric(prd_rn), base_test$ClaimAmount)
rmse_c=rmse(as.numeric(prd_rn), base_test$ClaimAmount)

