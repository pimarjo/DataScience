# Fichier pour entreposer le code hors du markdown. C'est plus pratique pour la partie algo
# On scinde en autant de partie que de méthodes à mettre en oeuvre

#Initialisation des packages ----
rm(list =ls())

library(magrittr)
library(rpart)
library(rpart.plot)

setwd("~/ISFA/3A/Data Science/ProjetDataScience/DataScience/data")
#Initialisation des donnees ----

load("data/freMTPL2freq.rda")
load("data/freMTPL2sev.rda")

frequence <- freMTPL2freq
severite <- freMTPL2sev

rm(freMTPL2freq, freMTPL2sev)

#Mise en forme des donnéees ----

#Enlèvons Density?
frequence <-frequence[, ! colnames(frequence) %in% "Density"]

#On formate les données de la base de frequence
frequence$ClaimNb <- frequence$ClaimNb %>% unname() %>% as.numeric()
frequence$VehPower <- as.integer(frequence$VehPower)
frequence$Exposure <- as.double(frequence$Exposure)
frequence$Area <- as.factor(frequence$Area)
frequence$VehAge <- as.integer(frequence$VehAge)
frequence$DrivAge <- as.integer(frequence$DrivAge)
frequence$BonusMalus <- as.integer(frequence$BonusMalus)
frequence$VehBrand <- as.factor(frequence$VehBrand)
frequence$VehGas <- as.factor(frequence$VehGas)
frequence$Region <- as.factor(frequence$Region)

#On formate la base de cout
severite$IDpol <- as.integer(severite$IDpol)
severite$ClaimAmount <- as.numeric(severite$ClaimAmount)

#On prends que les exposures inférieures à 1 ?
frequence <- frequence[frequence$Exposure <= 1,]

#Nos bases
head(frequence)
head(severite)


#On fusionne les bases
base <- merge(x = frequence, y = severite, by = "IDpol")
head(base)

summary(base)

#######################################################################################################
######################_________________________   CART
#######################################################################################################


#Premier arbre CART max methode anova
rpart::rpart(formula = base$ClaimAmount ~ base$IDpol + base$ClaimNb + base$Area + base$VehPower + base$VehAge + base$DrivAge + base$BonusMalus + base$VehBrand + base$VehGas + base$Region
             , weights = base$Exposure
             , method = "anova"
             , control = list(cp = 0)) -> arbre

arbre %>% rpart.plot::rpart.plot()

arbre %>% rpart::plotcp()

#Je vois pas quoi en faire...
arbre %>% rpart::printcp()

#Arbre avec poisson
rpart(formula = ClaimAmount ~ ClaimNb + Area + VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Region
      , weights = Exposure
      , method = "poisson"
      , data = base
      , control = list(cp = 0, minsplit = 1)) -> arbre

#Je plot les cp (parametre de complexité)
arbre %>% plotcp()

arbre$cptable %>% View()


#j'élague l'arbre avec le paramètre de complexité qui m'amène la plus petite xstd
prune(arbre, cp = 0.0128637) %>% rpart.plot()


#######################################################################################################
######################_________________________   Random forest
#######################################################################################################












#######################################################################################################
######################_________________________   Neural
#######################################################################################################












#######################################################################################################
######################_________________________   Gradient boosting 
#######################################################################################################

# paramètre : 
set.seed(seed=100)
.Proportion.Wanted = 0.70 # pour des question de rapiditée d'exection, j'ai déscendu la proportion a 0.01, il faut la remonter a 0.8 avent de rendre le code.

# application : 

#Je fais une liste d'éléments pris au hazard dans les indices de notre BDD de fréquence
.index_entrainement <- (1:nrow(base.mean)) %>% sample(.,size = .Proportion.Wanted * nrow(base.mean))

test <- base.mean[.index_entrainement,]
train <- base.mean[! seq(from = 1, to = nrow(base.mean)) %in% .index_entrainement, ]

# retour : 
.Proportion.Achieved = round(100* nrow(train) / nrow(base.mean), 2)







#fonction pour le calcul du taux d’erreur
err_rate <- function(D,prediction){
  #matrice de confusion
  mc <- table(D$chiffre,prediction)
  #taux d’erreur
  #1- somme(individus classés correctement) / somme totale individus
  err <- 1 - sum(diag(mc))/sum(mc)
  print(paste("Error rate :",round(100*err,2),"%"))
}

severite.mean <- aggregate(ClaimAmount ~ IDpol, data = severite, mean)
names(severite.mean) <- c("IDpol", "MeanClaimAmount")
base.mean <- merge(x = frequence, y = severite.mean, by = "IDpol", all.x = T)
base.mean$MeanClaimAmount <- replace(base.mean$MeanClaimAmount, is.na(base.mean$MeanClaimAmount), 0)


head(base.mean,5)


library(gbm)

#On met en oeuvre une décomposition fréquence coût

m.gbm.defaut <- gbm(MeanClaimAmount ~ Area + VehGas + VehBrand + VehAge + VehPower + DrivAge + Region + BonusMalus
                    , data = base.mean
                    , distribution="gaussian"
                    , shrinkage = 0.01
                    , interaction.depth = 7)
print(m.gbm.defaut)
print(head(summary(m.gbm.defaut),10))
View(head(severite.mean,30))

