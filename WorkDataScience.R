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
load("./data/train&valid&test.RData")
head(test)
head(valid)
head(train)


nrow(merge(test, valid, by = "IDpol"))


#Mise en forme des donnéees ----


# #Enlèvons Density?
# frequence <-frequence[, ! colnames(frequence) %in% "Density"]
# 
# #On formate les données de la base de frequence
# #frequence$ClaimNb <- frequence$ClaimNb %>% unname() %>% as.numeric()
# frequence$VehPower <- as.integer(frequence$VehPower)
# frequence$Exposure <- as.double(frequence$Exposure)
# frequence$Area <- as.factor(frequence$Area)
# frequence$VehAge <- as.integer(frequence$VehAge)
# frequence$DrivAge <- as.integer(frequence$DrivAge)
# frequence$BonusMalus <- as.integer(frequence$BonusMalus)
# frequence$VehBrand <- as.factor(frequence$VehBrand)
# frequence$VehGas <- as.factor(frequence$VehGas)
# frequence$Region <- as.factor(frequence$Region)
# 
# #On formate la base de cout
# severite$IDpol <- as.integer(severite$IDpol)
# severite$ClaimAmount <- as.numeric(severite$ClaimAmount)
# 
# #On prends que les exposures inférieures à 1 ?
# frequence <- frequence[frequence$Exposure <= 1,]
# 
# #Nos bases
# head(frequence)
# head(severite)
# 
# 
# 
# 
# #INUTILE DE MERGE LA BASE SANS AVOIR ENLEVER LES SINISTRES GRAVES
# 
# # att <- data.frame(severite[which(severite$ClaimAmount<=.seuil.grv),],rep(1,nrow(severite[which(severite$ClaimAmount<=.seuil.grv),])))
# # names(att) <- c("IDpol","ClaimAmount","AttClaimNb")
# # att.mean <- aggregate(ClaimAmount ~ IDpol, data = severite, mean)
# # names(att.mean) <- c("IDpol", "MeanClaimAmount")
# # base <- merge(x = frequence, y = att.mean, by = "IDpol", all.x = T)
# # base$MeanClaimAmount <- replace(base$MeanClaimAmount, is.na(base$MeanClaimAmount), 0)
# 
# head(base,5)
# summary(base)
# 
# severite.mean <- aggregate(ClaimAmount ~ IDpol, data = severite, mean)
# names(severite.mean) <- c("IDpol", "MeanClaimAmount")
# base.mean <- merge(x = frequence, y = severite.mean, by = "IDpol", all.x = T)
# base.mean$MeanClaimAmount <- replace(base.mean$MeanClaimAmount, is.na(base.mean$MeanClaimAmount), 0)
# 
# 
# head(base.mean,5)
# summary(base.mean)
# 
# 
# #On enlève les polices sinistrées avec un montant moyen de sinistres nul
# base.mean <- base.mean[-which(base.mean$ClaimNb > 0 & base.mean$MeanClaimAmount ==0),]
# 
# 
# 
# #On enlève les polices sinistrées avec un montant moyen de sinistres nul
# base <- base[-which(base$ClaimNb>0 & base$MeanClaimAmount == 0),]
# 
# 
# 
# ##____ Création base de train et base de test
# # Pour le Neural il sera très probable nécessaire de réduire le nombre de données pour des question de temps d'exécution
# 
# # paramètre :
# set.seed(seed=100)
# .Proportion.Wanted = 0.30 # pour des question de rapiditée d'exection, j'ai déscendu la proportion a 0.01, il faut la remonter a 0.8 avent de rendre le code.
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
# 
# .Proportion.Achieved
# 


#Il faut maintenant érifier que les propriété statistiques de la base de données sont respectées


train %>% summary()
valid %>% summary()
test %>% summary()

#OK nous sommes biens. Les propriétés de la base de données sont toujour biens vérifiées





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


sample3=sample(1:dim(train)[1], ceiling(0.4*dim(train)[1]), replace = FALSE)

# Echantillon réduit pour un gain de temps

echantillon=train[sample3,]

nrow(echantillon)

################################# Réseau de Neurones########################################

#1  Coût

train.cout <- train[-which(train$MeanClaimAmount == 0),]
test.cout <- test[-which(test$MeanClaimAmount == 0),]
echantillon.cout<-echantillon[-which(echantillon$MeanClaimAmount == 0),] #6839
summary(echantillon.cout$MeanClaimAmount)
summary(train.cout$MeanClaimAmount)

# Dans cet algo, on cherchera  à déterminer l'élément  le plus important:le nombre de neurones sur la couche cachée parallèlement aux conditions d'apprentissage (temps ou nombre de boucles) 

# A noter que l'alternative pour déterminer le nombre de neurones est celle du decay: paramètre de régularisation  

#1-1 Régression

#  Pour réduire le temps de calcul, on va  Fitter un neural network sur la base réduite ( 6839 rows): La meilleure mÃ©hotde pour dÃ©terminer le nombre de layers 
#et le nombre de neurones

n <- names(echantillon.cout)
mygrid <- expand.grid(size=c(1,2,3,4,5,6,7),decay=seq(1,5),KEEP.OUT.ATTRS = TRUE, stringsAsFactors = TRUE)

#as.formula nous permet de voir au plus clair sur les variables prise en compte pour les fit

varb <- as.formula(paste(" MeanClaimAmount", paste(c("DrivAge","VehAge","VehPower","VehBrand","VehGas","BonusMalus","Area","Region","Density"), collapse = " + "),sep=" ~ "))

# 1-2 Trainning net 

str(echantillon.cout) # vérification des types de variables avant  la régression train
ctrl    <- trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        verboseIter = T,
                        returnResamp = "all")

train.fit = train(varb ,data=echantillon.cout,method = "nnet",tuneGrid = mygrid ,trace=F, trControl =ctrl)

train.fit = train(MeanClaimAmount~DrivAge+VehAge+VehPower+VehBrand+VehGas+BonusMalus+Area+Region+Density ,data=echantillon.cout,method = "nnet",tuneGrid = mygrid ,trace=F, trControl =ctrl)

plot(train.fit) 

train.fit$resample

train.fit$bestTune # Size 4 et Decay 2 ( Le modèle choisi a le plus petit RMSE)

#Nous nous contentons de 500 itérations ( maxit= 500)

#Optimiser les paramètres nécessite la validation croisée. La fonction tune.nnet() de la librairie e1071 est adaptée:

nnet=tune.nnet(MeanClaimAmount~DrivAge+VehAge+VehPower+VehBrand+VehGas+BonusMalus+Area+Region+Density,data = echantillon.cout, size=seq(1,7), decay=seq(1,5), maxit=500,linout=TRUE)

nnet=tune.nnet(varb,data = echantillon.cout, size=seq(1,7), decay=seq(1,5), maxit=500,linout=TRUE)

plot(nnet)

cout_rn = nnet(MeanClaimAmount~DrivAge+VehAge+VehPower+VehBrand+VehGas+BonusMalus+Area+Region+Density,data =train.cout, size=4, decay=2, maxit=500, linout=TRUE)

summary(cout_rn)

plot.nnet(cout_rn)

summary(cout_rn$fitted.values); summary(train.cout$MeanClaimAmount)

# 1-3 Prédiction cout_rn

prd_rn=predict(cout_rn,test.cout)

#  Nous comparons la qualité des prédictions en observant les montants de sinistres de la predictiona avec la base  test :

summary(prd_rn)

summary(test.cout$MeanClaimAmount)
 
#Nous avons un bon  ajustement pour la moyenne  mais nous constatons une sous-estimation du nombre maximal de sinistres  dans notre prédiction 1527 contre 19 810


mse_c=mse(as.numeric(prd_rn), test.cout$MeanClaimAmount)

rmse_c=rmse(as.numeric(prd_rn), test.cout$MeanClaimAmount)

# RMSE, Erreur du modèle  ( vérification par une 2 ème méthode de calcul)

RMSE_C <- sqrt(sum((prd_rn-test.cout$MeanClaimAmount)^2)/nrow(test.cout));rmse_c


#2- Fréquence

x=0.01444018

sample4=sample(1:dim(train)[1], ceiling(x*dim(train)[1]), replace = FALSE)

# Echantillon réduit pour un gain de temps

echantillon.freq=train[sample4,]

nrow(echantillon.freq)

#Validation de l'échantillon

summary(echantillon.freq$ClaimNb)
summary(train$ClaimNb) #Pour la fréquence nous gardons toute la base d'apprentissage

#2-1 Régression

#as.formula nous permet de voir au plus clair sur les variables prise en compte pour les fit

varb_f <- as.formula(paste(" ClaimNb", paste(c("DrivAge","VehAge","VehPower","VehBrand","VehGas","BonusMalus","Area","Region","Density"), collapse = " + "),sep=" ~ "))


str(echantillon.freq) # vérification des types de variables avant  la régression train
 
ctrl    <- trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        verboseIter = T,
                        returnResamp = "all")

echantillon.freq$ClaimNb<- as.numeric(echantillon.freq$ClaimNb)

Fit_freq = train(ClaimNb ~ DrivAge+VehAge+VehPower+VehBrand+VehGas+BonusMalus+Area+Region+Density,data=echantillon.freq, method = "nnet", tuneGrid = mygrid ,trace=F, trControl =ctrl, weights=echantillon.freq$Exposure)

Fit_freq = train(varb_f, method = "nnet", tuneGrid = param , trControl = ctrl, weights=echantillon.freq$Exposure)


# Pramètres optimaux size = 2 et decay = 1 

summary(Fit_freq)

plot(Fit_freq)

Fit_freq$bestTune

# 2-2 Trainning net ( avec  l'exposition comme  pondération )

freq_optim = nnet(ClaimNb ~ DrivAge+VehAge+VehPower+VehBrand+VehGas+BonusMalus+Area+Density + Region,
               data = train, weights = train$Exposure, size=2, decay=1, maxit=100,linout=TRUE)

freq_optim = nnet(varb_f, data = train, weights = train$Exposure, size=2, decay=1, maxit=500,linout=TRUE)

summary(freq_optim)


summary(freq_optim$fitted.values) ; summary(train$ClaimNb) 

plot.nnet(freq_optim)

# 2-3 Prédiction 

freq_pred = predict(freq_optim,test,weights = test$Exposure,type="raw")

summary(freq_pred) ; summary(test$ClaimNb)

min(freq_pred)
max(freq_pred)

# un bon ajustement de la moyenne contre une sous-estimation du Max ( 0.23142 contre 16)!

# RMSE & MSE

mse_f=mse(as.numeric(freq_pred), test$ClaimNb)

rmse_f=rmse(as.numeric(freq_pred), test$ClaimNb)

PP_NN=mean(prd_rn)* mean(freq_pred)



#######################################################################################################
######################_________________________   Gradient boosting 
#######################################################################################################

# train$ClaimNb <- unname(train$ClaimNb)
# test$ClaimNb <- unname(test$ClaimNb)
# train <- train[which(train$MeanClaimAmount<=20000),]
# test <- test[which(test$MeanClaimAmount<=20000),]
# 
# train <- train[-which(train$ClaimNb>0 & train$MeanClaimAmount == 0),]
# test <- test[-which(test$ClaimNb>0 & test$MeanClaimAmount == 0),]
# 
# train <- cbind(train, AnnualClaimNb = train$ClaimNb/train$Exposure)
# test <- cbind(test, AnnualClaimNb = test$ClaimNb/test$Exposure)
# 
# proportion.valid <- 0.5
# sample.valid <- sample.int(nrow(test), size = nrow(test)*proportion.valid)
# 
# valid <- test[sample.valid,]
# test <- test[-sample.valid,]
# 
# save(list = c("train", "valid", "test")
#      , file = "./data/train&valid&test.RData")


#####################################
###########_________   Modèle de cout
#####################################

train.cout <- list(data = train[which(train$ClaimNb>0),]
                   , label = train$MeanClaimAmount[which(train$ClaimNb>0)]
)

valid.cout <- list(data = valid[which(valid$ClaimNb>0),]
                   , label = valid$MeanClaimAmount[which(valid$ClaimNb>0)]
)

test.cout <- list(data = test[which(test$ClaimNb>0),]
                  , label = test$MeanClaimAmount[which(test$ClaimNb>0)]
)

train.cout$data%>%summary()
valid.cout$data%>%summary()
test.cout$data%>%summary()


#On crée maintenant les flag et on enregistre a format xgb.DMatrix


train.cout.xgb <<- xgb.DMatrix(data = cbind(predict(dummyVars(data=train.cout$data,formula = "~Area"), newdata = train.cout$data)
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

valid.cout.xgb <<- xgb.DMatrix(data = cbind(predict(dummyVars(data=valid.cout$data,formula = "~Area"), newdata = valid.cout$data)
                                            , predict(dummyVars(data=valid.cout$data,formula = "~VehPower"), newdata = valid.cout$data)
                                            , VehAge = valid.cout$data$VehAge
                                            , DrivAge = valid.cout$data$DrivAge
                                            , BonusMalus = valid.cout$data$BonusMalus
                                            , predict(dummyVars(data=valid.cout$data,formula = "~VehBrand"), newdata = valid.cout$data)
                                            , predict(dummyVars(data=valid.cout$data,formula = "~VehGas"), newdata = valid.cout$data)
                                            , Density = valid.cout$data$Density
                                            , predict(dummyVars(data=valid.cout$data,formula = "~Region"), newdata = valid.cout$data)
)
, label = valid.cout$label)


test.cout.xgb <<- xgb.DMatrix(data = cbind(predict(dummyVars(data=test.cout$data,formula = "~Area"), newdata = test.cout$data)
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
                 ,valid = valid.cout.xgb)

cout.fit.1 = xgb.train(data = train.cout.xgb 
                       , max.depth = 4
                       , eta = 0.3
                       , gamma = 1
                       , colsample_bytree = 0.8
                       , subsample = 1
                       , nround = 100
                       , watchlist = watchlist
                       , print_every_n = 500
                       , early_stopping_rounds = 50)

pred.cout.1 <- predict(cout.fit.1, test.cout.xgb)
#View(cbind(test.cout$data, test.cout$label, pred.cout.1))

#Nous avons un biais de 48
pred.cout.1 %>% mean()
test.cout$label %>% mean()

###########________   Optimisation des paramètres de tuning

#On cherche maintenant à optimiser les paramètres de tuning
#Nous fixons le nombres d'arbres à 10 et nous cherchons les autres paramètres qui minimisent le RMSE

# # on complète notre grille de paramètres
# 
# xgb.grid.1 = expand.grid(nrounds = 50
#                          , max_depth = c(3,6,9)
#                          , eta = c(0.01, 0.001, 0.0001)
#                          , gamma = c(0.2,0.4,0.6,0.8,1)
#                          , colsample_bytree = c(0.2,0.4,0.6,0.8,1)
#                          , min_child_weight = c(0.2,0.4,0.6,0.8,1)
#                          , subsample = 0.8
# )
# xgb.grid.1
# 
# 
# #On fait une cross validation
# xgb.trcontrol.1 = trainControl(method = "cv",
#                                number = 7,
#                                verboseIter = TRUE,
#                                returnData = FALSE,
#                                returnResamp = "all", 
#                                allowParallel = TRUE)
# 
# #On train, ça prend du temps, beaucoup de temps
# xgb.train.1 = train(x = train.cout.xgb,
#                     y = train.cout$label,
#                     trControl = xgb.trcontrol.1,
#                     tuneGrid = xgb.grid.1,
#                     method = "xgbTree")




# Fitting nrounds = 50, max_depth = 3, eta = 0.01, gamma = 0.8, colsample_bytree = 1, min_child_weight = 0.8, subsample = 0.8 on full training set


#Nous allons maintenant chercher le nombre d'arbres optimal


mod.cout.xgb <- xgb.train(data = train.cout.xgb
                      , nrounds = 1000 
                      , max_depth = 3
                      , eta = 0.01
                      , gamma = 0.8
                      , colsample_bytree = 1
                      , min_child_weight = 0.8
                      , subsample = 1
                      , watchlist = watchlist
                      , early_stopping_rounds = 10
                      )
#nrounds = 386
# On l'enregistre
xgb.save(mod.cout.xgb, fname = "./xgboost/mode.cout.xgb.xgboost")

mod.graphe <- xgb.train(data = train.cout.xgb
                        , nrounds = 1000 #à renseigner
                        , max_depth = 3
                        , eta = 0.01
                        , gamma = 0.8
                        , colsample_bytree = 1
                        , min_child_weight = 0.8
                        , subsample = 0.8
                        , watchlist = watchlist
                        , early_stopping_rounds = 10
)

#On affiche l'évolution du RMSE en fonction du nombre d'arbres

plot(x = mod.cout$evaluation_log$iter
     , y = mod.cout$evaluation_log$valid_rmse
     , type = "l"
     , col = "red"
     , xlab = "Nombres d'arbres"
     , ylab = "RMSE"
     , ylim = c(1650,2300))

lines(x = mod.cout$evaluation_log$iter
      , y = mod.cout$evaluation_log$train_rmse
      , col = "blue")

#On affiche à partir de quand on commence l'overfitting
# abline(h = xgb.train.2$results$RMSE[which(xgb.train.2$results$nrounds == xgb.train.2$bestTune$nrounds)], col = "red")
abline(v=xgb.train.2$bestTune$nrounds, col = "red")


# #on peut s'amuser à grapher d'autres sensibilités
# 
# xgb.grid.1
# sensib <- function(tune.grid){
#   xgb.fit <- xgb.train(data = train.cout.xgb
#                        ,nrounds = tune.grid[1]
#                        , max_depth = tune.grid[2]
#                        , eta = tune.grid[3]
#                        , gamma = tune.grid[4]
#                        , colsample_bytree = tune.grid[5]
#                        , min_child_weight = tune.grid[6]
#                        , subsample = tune.grid[7]
#                        , watchlist = watchlist
#                        , print_every_n = 1000)
#   return(xgb.fit$evaluation_log$train_rmse %>% tail(.,1))
# }
# 
# xgb.grid.2 <- expand.grid(nrounds = seq(from = 1, to = 1001, by = 5)
#                            , max_depth = 3
#                            , eta = 0.01
#                            , gamma = 0.8
#                            , colsample_bytree = 1
#                            , min_child_weight = 0.8
#                            , subsample = 1
# )
# 
# sensib_nrounds <- apply(xgb.grid.2
#                         , MARGIN = 1
#                         , FUN = sensib)
# 
# 
# 
# 
# 
# xgb.grid.cout.max_depth <- expand.grid(nrounds = xgb.train.2$bestTune$nrounds 
#                          , max_depth = seq(from = 1, to = 10, by = 1)
#                          , eta = 0.01
#                          , gamma = 0.8
#                          , colsample_bytree = 1
#                          , min_child_weight = 0.8
#                          , subsample = 1
# )
# 
# sensib_max_depth <- apply(xgb.grid.cout.max_depth
#                         , MARGIN = 1
#                         , FUN = sensib)
# 
# 
# plot(x = xgb.grid.cout.max_depth$max_depth
#      , y = sensib_nrounds
#      , xlab = "Profondeur maximale"
#      , ylab = "RMSE"
#      , type = "l"
#      )
# 
# 
# 
# #le rmse décroit et atteint un minimum en 2
# plot(x = xgb.train.cout.max_depth$results$max_depth
#      , y = xgb.train.cout.max_depth$results$RMSE
#      , type = "l"
#      , xlab = "Max_depth"
#      , ylab = "RMSE"
#      , main = "RMSE en fonction de la profondeur maximale des arbres"
# )
# 
# 
# 



#On affiche la prédiction
 
pred.cout <- predict(mod.cout.xgb, test.cout.xgb)
#View(cbind(test.cout$data, test.cout$label, pred.cout))

#Nous avons un biais de 48
pred.cout %>% mean()
test.cout$label %>% mean()

(pred.cout - test.cout$label)^2 %>% mean() %>% sqrt() -> xgb.test.cout.RMSE 


#####################################
###########_________   Modèle de fréquence
#####################################


# train <- train[sample.int(n = nrow(train), size = nrow(train)*0.2),]

train.freq <- list(data = train[which(train$AnnualClaimNb<=10),]
                   , label = train$AnnualClaimNb[which(train$AnnualClaimNb<=10)]
)

valid.freq <- list(data = valid[which(valid$AnnualClaimNb<=10),]
                   , label = valid$AnnualClaimNb[which(valid$AnnualClaimNb<=10)]
)

test.freq <- list(data = test[which(test$AnnualClaimNb<=10),]
                  , label = test$AnnualClaimNb[which(test$AnnualClaimNb<=10)]
)



train.freq.xgb = xgb.DMatrix(data = as.matrix(cbind(predict(dummyVars(data=train.freq$data,formula = "~Area"), newdata = train.freq$data)
                                                 , predict(dummyVars(data=train.freq$data,formula = "~VehPower"), newdata = train.freq$data)
                                                 , VehAge = train.freq$data$VehAge
                                                 , DrivAge = train.freq$data$DrivAge
                                                 , BonusMalus = train.freq$data$BonusMalus
                                                 , predict(dummyVars(data=train.freq$data,formula = "~VehBrand"), newdata = train.freq$data)
                                                 , predict(dummyVars(data=train.freq$data,formula = "~VehGas"), newdata = train.freq$data)
                                                 , Density = train.freq$data$Density
                                                 , predict(dummyVars(data=train.freq$data,formula = "~Region"), newdata = train.freq$data)))
                             , label = train.freq$label)


valid.freq.xgb = xgb.DMatrix(data = as.matrix(cbind(predict(dummyVars(data=valid.freq$data,formula = "~Area"), newdata = valid.freq$data)
                                                    , predict(dummyVars(data=valid.freq$data,formula = "~VehPower"), newdata = valid.freq$data)
                                                    , VehAge = valid.freq$data$VehAge
                                                    , DrivAge = valid.freq$data$DrivAge
                                                    , BonusMalus = valid.freq$data$BonusMalus
                                                    , predict(dummyVars(data=valid.freq$data,formula = "~VehBrand"), newdata = valid.freq$data)
                                                    , predict(dummyVars(data=valid.freq$data,formula = "~VehGas"), newdata = valid.freq$data)
                                                    , Density = valid.freq$data$Density
                                                    , predict(dummyVars(data=valid.freq$data,formula = "~Region"), newdata = valid.freq$data)))
                             , label = valid.freq$label)

test.freq.xgb = xgb.DMatrix(data = as.matrix(cbind(predict(dummyVars(data=test.freq$data,formula = "~Area"), newdata = test.freq$data)
                                                    , predict(dummyVars(data=test.freq$data,formula = "~VehPower"), newdata = test.freq$data)
                                                    , VehAge = test.freq$data$VehAge
                                                    , DrivAge = test.freq$data$DrivAge
                                                    , BonusMalus = test.freq$data$BonusMalus
                                                    , predict(dummyVars(data=test.freq$data,formula = "~VehBrand"), newdata = test.freq$data)
                                                    , predict(dummyVars(data=test.freq$data,formula = "~VehGas"), newdata = test.freq$data)
                                                    , Density = test.freq$data$Density
                                                    , predict(dummyVars(data=test.freq$data,formula = "~Region"), newdata = test.freq$data)))
                             , label = test.freq$label)



###########________   Un 1er modèle

watchlist = list(train = train.freq.xgb
                 , valid = valid.freq.xgb
)

freq.fit.1 = xgb.train(data = train.freq.xgb
                     , max.depth = 9
                     , eta = 0.01
                     , gamma = 4
                     , colsample_bytree = 1
                     , subsample = 0.2
                     , nround = 30
                     , watchlist = watchlist
                     , print_every_n = 10
                     , weights = train.freq$data$Exposure
                     )


#On affiche la pred
pred <- predict(freq.fit.1, test.freq.xgb)
View(cbind(test.freq$data, test.freq$label, pred))


pred %>% mean()

test.freq$label %>% mean()

(pred - test.freq$label)^2 %>% mean() %>% sqrt()


#on charge la grille de candidats
load("./xgboost/candidats.freq.xgb")

###_ Code pour la recherche des paramètres, les résultats sont enregistrés dans le fichier "candidats.freq.xgb"

# # On se fait une petite optimisation des paramètres avec la même méthode que pour le coût à la main, caret ne tourne pas pour ça
# 
# xgb.grid.freq.1 = expand.grid(nrounds = 10
#                          , max_depth = c(3,6,9)
#                          , eta = c(0.01, 0.001, 0.0001)
#                          , gamma = c(0.2,0.4,0.6,0.8,1)
#                          , colsample_bytree = c(0.2,0.4,0.6,0.8,1)
#                          , min_child_weight = c(0.2,0.4,0.6,0.8,1)
#                          , subsample = 1
# )
# xgb.grid.freq.1
# 
# 
# #Une fonction qui prend en paramètre un jeu de paramètres et renvoie le meilleur RMSE du modèle ajusté pour ces paramètres
# test <- function(tune.grid){
#     xgb.fit <- xgb.train(data = train.freq.xgb
#                          ,nrounds = tune.grid[1]
#                          , max_depth = tune.grid[2]
#                          , eta = tune.grid[3]
#                          , gamma = tune.grid[4]
#                          , colsample_bytree = tune.grid[5]
#                          , min_child_weight = tune.grid[6]
#                          , subsample = 1
#                          , watchlist = watchlist
#                          , print_every_n = 1000
#                          , weights = train.freq$data$Exposure
#                          , verbose = T)
#     return(tail(xgb.fit$evaluation_log$valid_rmse,1))
#   }
# 
# 
# #On applique cette fonction à notre grille de recherche des paramètres et on enregistre les résultats dans une grilles avec en dernière colonne le RMSE
# train.freq.grid <- cbind(xgb.grid.freq.1, apply(xgb.grid.freq.1, FUN = test, MARGIN = 1))
# 
# 
# #on renomme les colonnes
# names(train.freq.grid) <- c("nrounds", "max_depth", "eta", "gamma", "colsample_bytree", "min_child_weight", "subsample","RMSE")
# 
# 
# #Nos candidats sont ceux avec le RMSE minimal
# # on remarque que seuls les paramètres gamma et min_child_weight
# candidats.freq.xgb <- train.freq.grid[which(train.freq.grid$RMSE == min(train.freq.grid$RMSE)),]
# 
# 
# #on les enregistre
# 

# #On ajoute une colonne avec le nombre d'arbres estimé
# candidats.freq.xgb <- cbind(candidats.freq.xgb
#                    , nrounds.opti = rep(0, length(candidats.freq.xgb$nrounds))
#                    , test.rmse.annual = rep(0, length(candidats.freq.xgb$nrounds))
#                    , test.rmse= rep(0, length(candidats.freq.xgb$nrounds)))
# 
# 
# 
# for (i in 1:10){
#   cat(i)
#   mod <- xgb.train(data = train.freq.xgb
#                    , nrounds = 10000
#                    , max_depth = candidats.freq.xgb$max_depth[i]
#                    , eta = candidats.freq.xgb$eta[i]
#                    , gamma = candidats.freq.xgb$gamma[i]
#                    , colsample_bytree = candidats.freq.xgb$colsample_bytree[i]
#                    , min_child_weight = candidats.freq.xgb$min_child_weight[i]
#                    , subsample = 1
#                    , watchlist = watchlist
#                    , print_every_n = 1000
#                    , early_stopping_rounds = 20
#                    , weights = train.freq$data$Exposure
#                    , verbose = F)
#   predict <- predict(mod, test.freq.xgb)
#   candidats.freq.xgb$nrounds.opt[i] <- mod$best_iteration
#   candidats.freq.xgb$test.rmse.annual[i] <- sqrt(mean((predict - test.freq$label)^2))
#   candidats.freq.xgb$test.rmse[i] <- sqrt(mean((test.freq$data$Exposure * predict(mod, test.freq.xgb) - test.freq$data$ClaimNb)^2))
# }




#Maintenant on va tester sur la base de test et on calcule le RMSE
# 
# 
# candidats.freq.xgb <- list(nrounds = candidats.freq.xgb$nrounds
#                   , max_depth = candidats.freq.xgb$max_depth
#                   , eta = candidats.freq.xgb$eta
#                   , gamma = candidats.freq.xgb$gamma
#                   , colsample_bytree = candidats.freq.xgb$colsample_bytree
#                   , min_child_weight = candidats.freq.xgb$min_child_weight
#                   , subsample = candidats.freq.xgb$subsample
#                   , RMSE = candidats.freq.xgb$RMSE
#                   , nrounds.opt = candidats.freq.xgb$nrounds.opt
#                   , test.RMSE = rep(0,length(candidats.freq.xgb$nrounds)))
# 
# 
# for (i in 1:10){
#   cat(i)
#   pred <- test.freq$data$Exposure * predict(xgb.train(data = train.freq.xgb
#                                                        , nrounds = candidats.freq.xgb$nrounds.opt[i]
#                                                        , max_depth = candidats.freq.xgb$max_depth[i]
#                                                        , eta = candidats.freq.xgb$eta[i]
#                                                        , gamma = candidats.freq.xgb$gamma[i]
#                                                        , colsample_bytree = candidats.freq.xgb$colsample_bytree[i]
#                                                        , min_child_weight = candidats.freq.xgb$min_child_weight[i]
#                                                        , subsample = 1
#                                                        , watchlist = watchlist
#                                                        , print_every_n = 1000
#                                                        , early_stopping_rounds = 20
#                                                        , weights = train.freq$data$Exposure
#                                                        , verbose = F)
#                                              , test.freq.xgb)
#   test.RMSE[i] <- sqrt(mean(pred - test.freq$label)^2)
# }

# #on les enregistre
# save(candidats.freq.xgb, file = "./candidats.freq.xgb")

#Bon peut choisir n'importe lequelle d'entre eux

#On prend arbitrairement le 8e
i<-8
#Onle train et on l'enregistre

mod.freq.xgb <- xgb.train(data = train.freq.xgb
                          , nrounds = candidats.freq.xgb$nrounds.opt[i]
                          , max_depth = candidats.freq.xgb$max_depth[i]
                          , eta = candidats.freq.xgb$eta[i]
                          , gamma = candidats.freq.xgb$gamma[i]
                          , colsample_bytree = 1
                          , min_child_weight = candidats.freq.xgb$min_child_weight[i]
                          , subsample = 1
                          , watchlist = watchlist
                          , weights = train.freq$data$Exposure
                          , verbose = T)

mod.freq.xgb <- xgb.load("./xgboost/mod.freq.xgb.xgboost")

#On affiche la prédiction

xgb.pred.freq <- predict(mod.freq.xgb, test.freq.xgb)
#View(cbind(test.freq$data, test.freq$label, pred.freq))

#Nous avons un biais de 48
xgb.pred.freq %>% mean()
test.freq$label %>% mean()

(xgb.pred.freq - test.freq$label)^2 %>% mean() %>% sqrt() -> xgb.test.freq.RMSE


xgb.save(mod.freq.xgb, fname = "./xgboost/mod.freq.xgb.xgboost")

test$ClaimNb %>% mean()
train.freq$data$ClaimAnnualNb[which(train.freq$data$ClaimAnnualNb<10)]%>%summary()

