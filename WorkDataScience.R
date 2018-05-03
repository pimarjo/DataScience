# Fichier pour entreposer le code hors du markdown. C'est plus pratique pour la partie algo
# On scinde en autant de partie que de méthodes à mettre en oeuvre

#Initialisation des packages ----
rm(list =ls())


library(magrittr)
library(rpart)
library(rpart.plot)

<<<<<<< HEAD
setwd("~/ISFA/3A/Data Science/ProjetDataScience/DataScience")
=======
#setwd("~/ISFA/3A/Data Science/ProjetDataScience/DataScience/data")
>>>>>>> 523eb97fe40e5c8cb32700f2da6f21ad8975a781
#Initialisation des donnees ----

data("freMTPL2freq")
data("freMTPL2sev")


# On récupère les deux datasets : 
data(freMTPL2freq)
data(freMTPL2sev)

data("freMTPL2freq")
data("freMTPL2sev")

frequence <- freMTPL2freq
severite <- freMTPL2sev

rm(freMTPL2freq, freMTPL2sev)

#Mise en forme des donnéees ----


#Enlèvons Density?
frequence <-frequence[, ! colnames(frequence) %in% "Density"]

#On formate les données de la base de frequence
#frequence$ClaimNb <- frequence$ClaimNb %>% unname() %>% as.numeric()
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


<<<<<<< HEAD

#INUTILE DE MERGE LA BASE SANS AVOIR ENLEVER LES SINISTRES GRAVES

# att <- data.frame(severite[which(severite$ClaimAmount<=.seuil.grv),],rep(1,nrow(severite[which(severite$ClaimAmount<=.seuil.grv),])))
# names(att) <- c("IDpol","ClaimAmount","AttClaimNb")
# att.mean <- aggregate(ClaimAmount ~ IDpol, data = severite, mean)
# names(att.mean) <- c("IDpol", "MeanClaimAmount")
# base <- merge(x = frequence, y = att.mean, by = "IDpol", all.x = T)
# base$MeanClaimAmount <- replace(base$MeanClaimAmount, is.na(base$MeanClaimAmount), 0)

head(base,5)
summary(base)
=======
severite.mean <- aggregate(ClaimAmount ~ IDpol, data = severite, mean)
names(severite.mean) <- c("IDpol", "MeanClaimAmount")
base.mean <- merge(x = frequence, y = severite.mean, by = "IDpol", all.x = T)
base.mean$MeanClaimAmount <- replace(base.mean$MeanClaimAmount, is.na(base.mean$MeanClaimAmount), 0)


head(base.mean,5)
summary(base.mean)


#On enlève les polices sinistrées avec un montant moyen de sinistres nul
base.mean <- base.mean[-which(base.mean$ClaimNb > 0 & base.mean$MeanClaimAmount ==0),]
>>>>>>> 523eb97fe40e5c8cb32700f2da6f21ad8975a781


#On enlève les polices sinistrées avec un montant moyen de sinistres nul
base <- base[-which(base$ClaimNb>0 & base$MeanClaimAmount == 0),]



##____ Création base de train et base de test
# Pour le Neural il sera très probable nécessaire de réduire le nombre de données pour des question de temps d'exécution

# paramètre :
set.seed(seed=100)
.Proportion.Wanted = 0.30 # pour des question de rapiditée d'exection, j'ai déscendu la proportion a 0.01, il faut la remonter a 0.8 avent de rendre le code.

# application :

#Je fais une liste d'éléments pris au hazard dans les indices de notre BDD de fréquence
.index_entrainement <- (1:nrow(base.mean)) %>% sample(.,size = .Proportion.Wanted * nrow(base.mean))

test <- base.mean[.index_entrainement,]
train <- base.mean[! seq(from = 1, to = nrow(base.mean)) %in% .index_entrainement, ]

# retour :
.Proportion.Achieved = round(100* nrow(train) / nrow(base.mean), 2)

.Proportion.Achieved



#Il faut maintenant érifier que les propriété statistiques de la base de données sont respectées

summary(base.mean)
summary(train)
summary(test)







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

<<<<<<< HEAD
library(caret)
library(xgboost)
library(fExtremes)
library(gbm)
library(onehot)

###########################
#########_____    Fonctions utiles
###########################

#fonction pour le calcul du taux d’erreur
err_rate <- function(D,prediction){
  #matrice de confusion
  mc <- table(D$chiffre,prediction)
  #taux d’erreur
  #1- somme(individus classés correctement) / somme totale individus
  err <- 1 - sum(diag(mc))/sum(mc)
  print(paste("Error rate :",round(100*err,2),"%"))
}


=======
# # paramètre : 
# set.seed(seed=100)
# .Proportion.Wanted = 0.70 # pour des question de rapiditée d'exection, j'ai déscendu la proportion a 0.01, il faut la remonter a 0.8 avent de rendre le code.
# 
# # application : 
# 
# #Je fais une liste d'éléments pris au hazard dans les indices de notre BDD de fréquence
# .index_entrainement <- (1:nrow(base.mean)) %>% sample(.,size = .Proportion.Wanted * nrow(base.mean))
# 
# test <- base.mean[.index_entrainement,]
# train <- base.mean[! seq(from = 1, to = nrow(base.mean)) %in% .index_entrainement, ]
# 
# # retour : 
# .Proportion.Achieved = round(100* nrow(train) / nrow(base.mean), 2)
>>>>>>> 523eb97fe40e5c8cb32700f2da6f21ad8975a781




###########################
#########_____    Mise en forme, Ecrètement et tarification des sinistres graves
###########################


mePlot(severite$ClaimAmount)

.seuil.grv <- 20000
nrow(severite[which(severite$ClaimAmount>.seuil.grv),])/nrow(severite)*100

att <- data.frame(severite[which(severite$ClaimAmount<=.seuil.grv),]
                  ,rep(1,nrow(severite[which(severite$ClaimAmount<=.seuil.grv),]))
                  )
names(att) <- c("IDpol","ClaimAmount","AttClaimNb")


att.mean <- aggregate(ClaimAmount ~ IDpol, data = att, mean)

#On renomme
names(att.mean) <- c("IDpol", "MeanClaimAmount")


base <- merge(x = frequence, y = att.mean, by = "IDpol", all.x = T)
base$MeanClaimAmount <- replace(base$MeanClaimAmount, is.na(base$MeanClaimAmount), 0)

head(base,5)
summary(base)
base <- base[-which(base$ClaimNb >0 & base$MeanClaimAmount ==0),]

grave <- data.frame(severite[which(severite$ClaimAmount>.seuil.grv),],rep(1,nrow(severite[which(severite$ClaimAmount>.seuil.grv),])))
names(grave) <- c("IDpol","GrvClaimAmount","GrvClaimNb")

surprime.grv <- sum(grave$GrvClaimAmount)/sum(base$Exposure)

#La surprime est de ...
surprime.grv



##############################################
################_____ Modèle de cout
##############################################

#On ne garde que les polices sinistrées
base.cout <- base[which(base$ClaimNb>0),]

###########################
#########_____    Séparation des données en 3 jeux: train, validation et test
###########################



 base.cout.oh <- predict(onehot(base.cout, stringsAsFactors = T, addNA = FALSE, max_levels = 100)
                      , base.cout)


#Les proportions que l'on veut pour nos différentes bases
proportion.train <- 0.5
porportion.valid <- 0.25
proportion.test <- 0.25

#On stocke toute les bases dans une liste de base, c'est plus simple à utiliser

#Les indices pour la base de train
sample.train <- sample.int(n = nrow(base.cout)
                           , size = floor(proportion.train*nrow(base.cout))
                           , replace = F)

#Une base temporaire qui est le reste de la base initiale après la division pour la de train
base.temp <- base.cout[-sample.train,]

sample.test <- base.temp %>% nrow() %>% sample.int(n=.
                                                   , size = floor(porportion.valid/proportion.train*.)
                                                   , replace = F)


#On crée les bases et la liste
base.cout <- list(full = base.cout
                  , train = base.cout[sample.train,]
                  , test = base.temp[sample.test,]
                  , valid = base.temp[-sample.test,])


###########################
#########_____    Modélisation
###########################
#On crée une liste des bases au format xgbMatrix
base.cout.xgbM <- list(train = xgb.DMatrix(data = base.cout$train[,-c(1,3,length(base.cout$full[1,]))]
                                           , label = base.cout$train[,'MeanClaimAmount'])
                       ,test = xgb.DMatrix(data = base.cout$test[,-c(1,3,length(base.cout$full[1,]))]
                                          , label = base.cout$test[,'MeanClaimAmount'])
                       ,valid = xgb.DMatrix(data = base.cout$valid[,-c(1,3,length(base.cout$full[1,]))]
                                            , label = base.cout$valid[,'MeanClaimAmount'])
                       )



#################
####____  Premier modèle (paramètres par défaut)
################


watchlist <- list(train = base.cout.xgbM$train, valid = base.cout.xgbM$valid)


bst_slow = xgb.train(data = base.cout.xgbM$train 
                     , max.depth = 2
                     , alpha = 0
                     , lambda = 1
                     , eta = 0.0001 
                     , nthread = 2 
                     , nround = 1000
                     , watchlist = watchlist
                     , objective = "reg:linear" 
                     , print_every_n = 500)

res <- rep(0,100)
for (i in seq(from= 1, to = 10001, by = 100)){
  bst_slow = xgb.train(data = base.cout.xgbM$train 
                       , max.depth = 10
                       , alpha = 0
                       , lambda = 1
                       , eta = 0.0001 
                       , nthread = 2 
                       , nround = i
                       , watchlist = watchlist
                       , objective = "reg:linear"
                       , verbose = 0
                       )
  print(i)
  
  res[i]<- sqrt(mean(((predict(bst_slow, base.cout.xgbM$test) - base.cout$test[,'MeanClaimAmount'])^2)))
}

<<<<<<< HEAD

y_hat_valid = predict(bst_slow, base.cout.xgbM$test)
test_mse = mean(((y_hat_valid - base.cout$test[,'MeanClaimAmount'])^2))
test_rmse = sqrt(test_mse)
test_rmse


#################
####____  Optimisation paramètres de tunning
################



# on regarde les paramètres du modèle
modelLookup("xgbLinear")

# on set up la grille de paramètres

xgb_grid_1 = expand.grid(nrounds = c(1000,2000,3000,4000) ,
                         eta = c(0.01, 0.001, 0.0001),
                         lambda = 1,
                         alpha = 0)
xgb_grid_1


#on utilise une cross validation
xgb_trcontrol_1 = trainControl(method = "cv",
                               number = 5,
                               verboseIter = TRUE,
                               returnData = FALSE,
                               returnResamp = "all", 
                               allowParallel = TRUE)

#on train
xgb_train_1 = train(x = base.cout.xgbM$train,
                    y = base.cout$train[,'MeanClaimAmount'],
                    trControl = xgb_trcontrol_1,
                    tuneGrid = xgb_grid_1,
                    method = "xgbLinear",
                    max.depth = 5)





m.gbm.defaut <- gbm(data = base.cout$train
                    ,formula = MeanClaimAmount ~ VehGas + VehBrand + VehAge + VehPower + DrivAge + Area +  BonusMalus + Region
                    ,distribution = "gaussian"
                    ,n.trees = 1000
                    ,shrinkage = 0.01
                    ,interaction.depth = 6
                    )
=======
# severite.mean <- aggregate(ClaimAmount ~ IDpol, data = severite, mean)
# names(severite.mean) <- c("IDpol", "MeanClaimAmount")
# base.mean <- merge(x = frequence, y = severite.mean, by = "IDpol", all.x = T)
# base.mean$MeanClaimAmount <- replace(base.mean$MeanClaimAmount, is.na(base.mean$MeanClaimAmount), 0)
# 
# 
# head(base.mean,5)
>>>>>>> 523eb97fe40e5c8cb32700f2da6f21ad8975a781




print(m.gbm.defaut)
print(head(summary(m.gbm.defaut),10))
pred <- predict(m.gbm.defaut, base.cout$test, n.trees = 100)


(pred - base.cout$test$MeanClaimAmount)^2 %>% mean() %>% sqrt()


length(pred[which(pred<0)])
#typeof(pred)
mean(pred[which(pred>0)])
mean(base.mean.sev$MeanClaimAmount)

gbm.perf(m.gbm.defaut,oobag.curve = T,method = "test")




