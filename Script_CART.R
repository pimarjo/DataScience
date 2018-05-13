#_______________________________________________________________________________________________
#                              				CART                                                |
#_______________________________________________________________________________________________|

# ------------------------------- INITIALISATION DES PACKAGES --------------------------------- #
library(CASdatasets)
library(magrittr)
library(rpart)
library(rpart.plot)
library(sp)
library(xts)

# -------------------------------   TRAITEMENT DES DONNEES   --------------------------------- #

train$VehPower <- as.integer(train$VehPower)
train$Exposure <- as.double(train$Exposure)
train$Area <- as.factor(train$Area)
train$VehAge <- as.integer(train$VehAge)
train$DrivAge <- as.integer(train$DrivAge)
train$BonusMalus <- as.integer(train$BonusMalus)
train$VehBrand <- as.factor(train$VehBrand)
train$VehGas <- as.factor(train$VehGas)
train$Region <- as.factor(train$Region)
train$MeanClaimAmount <- as.numeric(train$MeanClaimAmount)
train$ClaimNb <- as.numeric(train$ClaimNb)
train_NonNulle = train[-which(train$MeanClaimAmount == 0),]

# ------------------------------- APPLICATION DE LA METHODE CART ------------------------------ #

### ----------------- MODELE SEVERITE
## Premier essais arbre CART 
ad.Claim = rpart(MeanClaimAmount ~ Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Region + Density
                 , weights = Exposure
                 , data = train_NonNulle
                 , method = "anova"
                 , control = rpart.control(cp = 0))
prp(ad.Claim,extra = 1, type = 2, branch = 1, cex = 0.5, Margin = 0 ) 
# Arbre illisible, ajout d'une contraite sur le nb d'ind min par feuille avec minbucket

## ARBRE COMPLET SEVERITE
ad.Claim2 = rpart(MeanClaimAmount ~ Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Region + Density
                  , weights = Exposure
                  , data = train_NonNulle
                  , method = "anova"
                  , control = rpart.control( minbucket = 500, cp = 0))
summary((adOpt.Claim2))
## SORTIES GRAPHIQUES
names(ad.Claim2)
plotcp(ad.Claim2); 
printcp(ad.Claim2)
prp(ad.Claim2, uniform = TRUE, extra = 0, type = 2, branch = 0, cex = 0.7, Margin = 0, compress = TRUE)

# Complexite
plot(ad.Claim2$cptable[1:16,2], ad.Claim2$cptable[1:16,4],xlab='Complexite',ylab='CV erreur',main='Evo du CP')

## SIMPLIFICATION DE L'ARBRE
# Recuperation du cp minimisant 
adOpt.Claim2 = prune(ad.Claim2, cp = ad.Claim2$cptable[which.min(ad.Claim2$cptable[,4]),1]) 

# Representation de l'arbre obtenu
par(mfrow = c(1, 1), mar = c(1, 1, 1, 1))
plot(adOpt.Claim2, uniform = T, compress = T, margin = 0.1, branch = 0.3)
text(adOpt.Claim2, use.n = T, digits = 3, cex = 0.6)
prp(adOpt.Claim2, uniform = TRUE, extra=1,type=1, branch = 0, cex = 0.7, Margin = 0, compress=TRUE)

## PREVISIONS
# Calcul des prévisions sur l'echantillon de TRAIN
preds = predict(adOpt.Claim2)
mc = table(train_NonNulle$MeanClaimAmount,preds)
mc1=as.data.frame(cbind(train_NonNulle$MeanClaimAmount,preds))
View(mc1)
mc

summary(train_NonNulle$MeanClaimAmount)
summary(preds)
sum((preds-train_NonNulle$MeanClaimAmount)^2)/nrow(train_NonNulle)# Erreur quadratique moyenne de prévision
#3006876

# Calcul des prévisions sur l'echantillon TEST
test_NonNulle = test[-which(test$MeanClaimAmount == 0),]
preds2 = predict(adOpt.Claim2, newdata = test_NonNulle)
mc_1 = table(test_NonNulle$MeanClaimAmount,preds2)
mc1_1 = as.data.frame(cbind(test_NonNulle$MeanClaimAmount,preds2))
View(mc1_1)
mc_1

summary(test_NonNulle$MeanClaimAmount)
summary(preds)
# Erreur quadratique moyenne de prévision
sum((preds2-test_NonNulle$MeanClaimAmount)^2)/nrow(test_NonNulle) #3160889


### ----------------- MODELE FREQUENCE

ad.freq = rpart(ClaimNb ~ Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Region + Density
                , data = train
                , weights = Exposure
                , control = rpart.control(cp=0))# trop long sans minbucket
prp(ad.freq,extra=1,type=2, branch = 1,cex=0.5, Margin=0 ) 

ad.freq2 = rpart(ClaimNb ~ Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Region + Density
                 , data = train
                 , weights = Exposure
                 , control = rpart.control( minbucket = 2000, cp = 0)) 
summary(ad.freq2)

## SORTIES GRAPHIQUES
plotcp(ad.freq2)	
printcp(ad.freq2)
prp(ad.freq2, uniform = TRUE, extra = 0, type = 2, branch = 0, cex = 0.7, Margin = 0, compress = TRUE, ycompress = TRUE)

## SIMPLIFICATION DE L'ARBRE
# Recuperation du cp minimisant 
adOpt.freq2 = prune(ad.freq2,cp=ad.freq2$cptable[which.min(ad.freq2$cptable[,4]),1]) 
# cp = 4.569855e-05

# Representation de l'arbre obtenu
par(mfrow = c(1, 1), mar = c(1, 1, 1, 1))
plot(adOpt.freq2, uniform = T, compress = T, margin = 0.1, branch = 0.3)
text(adOpt.freq2, use.n = T, digits = 3, cex = 0.6)
prp(adOpt.freq2, uniform = TRUE, extra = 0, type = 1, branch = 0, cex = 0.7, Margin = 0, compress = TRUE)

## PREVISIONS
# Sur l'echantillon de TRAIN
preds_freq = predict(adOpt.freq2)
mc_freq = table(train$ClaimNb,preds_freq)
mc1_freq = as.data.frame(cbind(train$ClaimNb,preds_freq))
mc_freq
View(mc1_freq)

summary(train$ClaimNb)
summary(preds_freq)
# Erreur quadratique moyenne de prévision
sum((preds_freq - train$ClaimNb)^2)/nrow(train)


# Calcul des prévisions sur 
# Sur l'echantillon TEST
preds2_freq = predict(adOpt.freq2, newdata = test)
mc_1_freq = table(test$ClaimNb,preds2_freq)
mc1_1_freq = as.data.frame(cbind(test$ClaimNb, preds2_freq))
mc_1_freq
View(mc1_1_freq)

summary(test$ClaimNb)
summary(preds2_freq)
# Erreur quadratique moyenne de prévision
sum((preds2_freq - test$ClaimNb)^2)/nrow(test)

