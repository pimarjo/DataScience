# Fichier pour entreposer le code hors du markdown. C'est plus pratique pour la partie algo
# On scinde en autant de partie que de méthodes à mettre en oeuvre

#Initialisation des packages ----

rm(list =ls())

library(magrittr)
library(rpart)
library(rpart.plot)


#Initialisation des donnees ----

load("data/freMTPL2freq.rda")
load("data/freMTPL2sev.rda")

frequence <- freMTPL2freq
severite <- freMTPL2sev
rm(freMTPL2freq, freMTPL2sev)

#Mise en forme des donnéees ----

frequence <- frequence[, ! colnames(frequence) %in% "Density"]

frequence$ClaimNb <- frequence$ClaimNb %>% names %>% as.numeric()
frequence$VehPower <- as.integer(frequence$VehPower)
frequence$Exposure <- as.double(frequence$Exposure)
frequence$Area <- as.factor(frequence$Area)
frequence$VehAge <- as.integer(frequence$VehAge)
frequence$DrivAge <- as.integer(frequence$DrivAge)
frequence$BonusMalus <- as.integer(frequence$BonusMalus)
frequence$VehBrand <- as.factor(frequence$VehBrand)
frequence$VehGas <- as.factor(frequence$VehGas)
frequence$Region <- as.factor(frequence$Region)

severite$IDpol <- as.integer(severite$IDpol)
severite$ClaimAmount <- as.numeric(severite$ClaimAmount)

frequence <- frequence[frequence$Exposure <= 1,]

head(frequence)
head(severite)

base <- merge(x = frequence, y = severite, by = "IDpol")
head(base)

rpart::rpart(formula = base$ClaimAmount ~ base$IDpol + base$ClaimNb + base$Area + base$VehPower + base$VehAge + base$DrivAge + base$BonusMalus + base$VehBrand + base$VehGas + base$Region
             , weights = base$Exposure
             , method = "anova"
             , control = list(cp = 0)) -> arbre

arbre %>% rpart.plot::rpart.plot()

arbre %>% rpart::plotcp()

arbre %>% rpart::printcp()

rpart(formula = ClaimAmount ~ ClaimNb + Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Region
             , weights = Exposure
             , method = "anova"
             , data = base
             , control = list(cp = 0)) -> arbre

arbre %>% plotcp()

arbre %>% printcp() %>% data.frame() -> CP


0.008097

rpart(formula = ClaimAmount ~ ClaimNb + Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Region
      , weights = Exposure
      , method = "poisson"
      , data = base
      , control = list(cp = 0, minsplit = 1)) -> arbre

arbre %>% plotcp()

prune(arbre, cp = 0.0128637) %>% rpart.plot()