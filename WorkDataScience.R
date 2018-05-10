# Fichier pour entreposer le code hors du markdown. C'est plus pratique pour la partie algo
# On scinde en autant de partie que de méthodes à mettre en oeuvre

#Initialisation des packages ----
rm(list =ls())

library(dplyr)
library(magrittr)
library(rpart)
library(rpart.plot)
library(caret)
library(xgboost)
library(fExtremes)
library(gbm)
library(Matrix)

#LOADING DES JEUX DENTRAINEMENT ET DE TESTS
load("data/trainandtest.rda")
head(train)
head(test)



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




#INUTILE DE MERGE LA BASE SANS AVOIR ENLEVER LES SINISTRES GRAVES

# att <- data.frame(severite[which(severite$ClaimAmount<=.seuil.grv),],rep(1,nrow(severite[which(severite$ClaimAmount<=.seuil.grv),])))
# names(att) <- c("IDpol","ClaimAmount","AttClaimNb")
# att.mean <- aggregate(ClaimAmount ~ IDpol, data = severite, mean)
# names(att.mean) <- c("IDpol", "MeanClaimAmount")
# base <- merge(x = frequence, y = att.mean, by = "IDpol", all.x = T)
# base$MeanClaimAmount <- replace(base$MeanClaimAmount, is.na(base$MeanClaimAmount), 0)

head(base,5)
summary(base)

severite.mean <- aggregate(ClaimAmount ~ IDpol, data = severite, mean)
names(severite.mean) <- c("IDpol", "MeanClaimAmount")
base.mean <- merge(x = frequence, y = severite.mean, by = "IDpol", all.x = T)
base.mean$MeanClaimAmount <- replace(base.mean$MeanClaimAmount, is.na(base.mean$MeanClaimAmount), 0)


head(base.mean,5)
summary(base.mean)


#On enlève les polices sinistrées avec un montant moyen de sinistres nul
base.mean <- base.mean[-which(base.mean$ClaimNb > 0 & base.mean$MeanClaimAmount ==0),]



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

train$ClaimNb <- unname(train$ClaimNb)
test$ClaimNb <- unname(test$ClaimNb)
train <- train[which(train$MeanClaimAmount<20000),]
test <- test[which(test$MeanClaimAmount<20000),]

train <- train[-which(train$ClaimNb>0 & train$MeanClaimAmount == 0),]
test <- test[-which(test$ClaimNb>0 & test$MeanClaimAmount == 0),]

train <- cbind(train, ClaimAnnualNb = train$ClaimNb/train$Exposure)
test <- cbind(test, ClaimAnnualNb = test$ClaimNb/test$Exposure)


#####################################
###########_________   Modèle de cout
#####################################

train.cout <- list(data = train[which(train$ClaimNb>0),]
                   , label = train$MeanClaimAmount[which(train$ClaimNb>0)]
)

test.cout <- list(data = test[which(test$ClaimNb>0),]
                  , label = test$MeanClaimAmount[which(test$ClaimNb>0)]
)


#On crée maintenant les flag


train.cout.xgb = xgb.DMatrix(data = cbind(predict(dummyVars(data=train.cout$data,formula = "~Area"), newdata = train.cout$data)
                                          , predict(dummyVars(data=train.cout$data,formula = "~VehPower"), newdata = train.cout$data)
                                          , VehAge = train.cout$data$VehAge
                                          , DrivAge = train.cout$data$DrivAge
                                          , BonusMalus = train.cout$data$BonusMalus
                                          , predict(dummyVars(data=train.cout$data,formula = "~VehBrand"), newdata = train.cout$data)
                                          , predict(dummyVars(data=train.cout$data,formula = "~VehGas"), newdata = train.cout$data)
                                          , Density = train.cout$data$Density
                                          , predict(dummyVars(data=train.cout$data,formula = "~Region"), newdata = train.cout$data)
)
, label = train.cout$label)



test.cout.xgb = xgb.DMatrix(data = cbind(predict(dummyVars(data=test.cout$data,formula = "~Area"), newdata = test.cout$data)
                                         , predict(dummyVars(data=test.cout$data,formula = "~VehPower"), newdata = test.cout$data)
                                         , VehAge = test.cout$data$VehAge
                                         , DrivAge = test.cout$data$DrivAge
                                         , BonusMalus = test.cout$data$BonusMalus
                                         , predict(dummyVars(data=test.cout$data,formula = "~VehBrand"), newdata = test.cout$data)
                                         , predict(dummyVars(data=test.cout$data,formula = "~VehGas"), newdata = test.cout$data)
                                         , Density = test.cout$data$Density
                                         , predict(dummyVars(data=test.cout$data,formula = "~Region"), newdata = test.cout$data)
)
, label = test.cout$label)


###########________   Un 1er modèle

#On train un premier modèle à l'aide la fonction xgb.train
watchlist = list(train = train.cout.xgb
                 ,test = test.cout.xgb)

cout.fit.1 = xgb.train(data = train.cout.xgb 
                       , max.depth = 4
                       , eta = 0.3
                       , gamma = 1
                       , colsample_bytree = 0.8
                       , subsample = 0.8
                       , nround = 100
                       , watchlist = watchlist
                       , print_every_n = 500
                       , early_stopping_rounds = 50)

pred.cout.1 <- predict(cout.fit.1, test.cout.xgb)
View(cbind(test.cout$data, test.cout$label, pred.cout.1))

#Nous avons un biais de 48
pred.cout.1 %>% mean()
test.cout$label %>% mean()

###########________   Optimisation des paramètres de tuning

#On cherche maintenant à optimiser les paramètres de tuning
#Nous fixons le nombres d'arbres à 10 et nous cherchons les autres paramètres qui minimisent le RMSE

# on complète notre grille de paramètres

xgb.grid.1 = expand.grid(nrounds = 50
                         , max_depth = c(3,6,9)
                         , eta = c(0.01, 0.001, 0.0001)
                         , gamma = c(0.2,0.4,0.6,0.8,1)
                         , colsample_bytree = c(0.2,0.4,0.6,0.8,1)
                         , min_child_weight = c(0.2,0.4,0.6,0.8,1)
                         , subsample = 0.8
)
xgb.grid.1


#On fait une cross validation
xgb.trcontrol.1 = trainControl(method = "cv",
                               number = 7,
                               verboseIter = TRUE,
                               returnData = FALSE,
                               returnResamp = "all", 
                               allowParallel = TRUE)

#On train, ça prend du temps, beaucoup de temps
xgb.train.1 = train(x = train.cout.xgb,
                    y = train.cout$label,
                    trControl = xgb.trcontrol.1,
                    tuneGrid = xgb.grid.1,
                    method = "xgbTree")


#Nous allons maintenant chercher le nombre d'arbres optimal


xgb.grid.2 = expand.grid(nrounds = seq(from = 1, to = 1001, by = 5)
                         , max_depth = xgb.train.1$bestTune$max_depth
                         , eta = xgb.train.1$bestTune$eta
                         , gamma = xgb.train.1$bestTune$gamma
                         , colsample_bytree = xgb.train.1$bestTune$colsample_bytree
                         , min_child_weight = xgb.train.1$bestTune$min_child_weight
                         , subsample = xgb.train.1$bestTune$subsample
)


xgb.trcontrol.2 = trainControl(method = "cv",
                               number = 7,
                               verboseIter = TRUE,
                               returnData = FALSE,
                               returnResamp = "all", 
                               allowParallel = TRUE)


xgb.train.2 = train(x = train.cout.xgb
                    , y = train.cout$label
                    , trControl = xgb.trcontrol.2
                    , tuneGrid = xgb.grid.2
                    , method = "xgbTree")


#On affiche la progression du RMSE en fonction de nombre d'arbres
plot(x = xgb.train.2$results$nrounds[which(xgb.train.2$results$RMSE<=1800)]
     , y = xgb.train.2$results$RMSE[which(xgb.train.2$results$RMSE<=1800)]
     , type = "l"
     , xlab = "Nombre d'arbres"
     , ylab = "RMSE"
     , main = "RMSE en fonction du nombre d'arbres"
)

#On affiche à partir de quand on commence l'overfitting
abline(h = xgb.train.2$results$RMSE[which(xgb.train.2$results$nrounds == xgb.train.2$bestTune$nrounds)], col = "red")
abline(v=xgb.train.2$bestTune$nrounds, col = "red")


#####################################
###########_________   Modèle de fréquence
#####################################

train <- train[which(train$ClaimAnnualNb<=10),]
test <- test[which(test$ClaimAnnualNb<=10),]

train.freq <- list(data = train[which(train$ClaimAnnualNb<=10),]
                   , label = train$ClaimAnnualNb[which(train$ClaimAnnualNb<=10)]
)

test.freq <- list(data = test[which(test$ClaimAnnualNb<=10),]
                  , label = test$ClaimAnnualNb[which(test$ClaimAnnualNb<=10)]
)

train.freq.sparse = Matrix(sparse = F
                           , data = cbind(predict(dummyVars(data=train.freq$data,formula = "~Area"), newdata = train.freq$data)
                                          , predict(dummyVars(data=train.freq$data,formula = "~VehPower"), newdata = train.freq$data)
                                          , VehAge = train.freq$data$VehAge
                                          , DrivAge = train.freq$data$DrivAge
                                          , BonusMalus = train.freq$data$BonusMalus
                                          , predict(dummyVars(data=train.freq$data,formula = "~VehBrand"), newdata = train.freq$data)
                                          , predict(dummyVars(data=train.freq$data,formula = "~VehGas"), newdata = train.freq$data)
                                          , Density = train.freq$data$Density
                                          , predict(dummyVars(data=train.freq$data,formula = "~Region"), newdata = train.freq$data)
                           ))

train.freq.xgb = xgb.DMatrix(data = train.freq.sparse
                             , label = train.freq$label)


watchlist = list(train = train.freq.xgb
                 #,test = test.cout.xgb
)

cout.fit.1 = xgboost(data = train.freq.sparse
                     , label = train.freq$label
                     , obj = NULL
                     , feval = NULL
                     , max.depth = 4
                     , eta = 0.3
                     , gamma = 1
                     , colsample_bytree = 0.8
                     , subsample = 0.8
                     , nround = 1000
                     , watchlist = watchlist
                     , print_every_n = 10
                     , early_stopping_rounds = 50)

m.gbm.defaut <- gbm(data = train
                    ,formula = ClaimNb ~  VehBrand   + DrivAge  + Region + BonusMalus + Density + VehGas + VehAge
                    ,distribution = "gaussian"
                    , weights = train[,'Exposure']
                    ,n.trees  = 200
                    ,shrinkage = 0.1
                    ,interaction.depth = 5
                    ,n.minobsinnode = 500
                    , train.fraction = 0.75
                    
)
print(m.gbm.defaut)
print(head(summary(m.gbm.defaut),200))
pred <- predict(m.gbm.defaut,test, n.trees = 200)


(pred - test$ClaimNb)^2 %>% mean() %>% sqrt()
head(pred,100)

pred %>% mean()
test$ClaimNb %>% mean()
train.freq$data$ClaimAnnualNb[which(train.freq$data$ClaimAnnualNb<10)]%>%summary()