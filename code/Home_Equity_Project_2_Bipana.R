library(mice)
library(ROSE)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(pROC)
library(ggplot2)

# SECTION 1: LOAD DATA
hmeq <- read.csv("./hmeq.csv", stringsAsFactors = FALSE)

# Basic structure and summary of all variables
dim(hmeq)
str(hmeq)
summary(hmeq)

# SECTION 2: DATA CLEANING
# Converting blank strings to NA
hmeq$JOB[hmeq$JOB == ""]       <- NA
hmeq$REASON[hmeq$REASON == ""] <- NA

# Converting numeric columns
num_cols <- c("LOAN","MORTDUE","VALUE","YOJ","DEROG","DELINQ","CLAGE","NINQ","CLNO","DEBTINC")
for (col in num_cols) {hmeq[[col]] <- as.numeric(hmeq[[col]])}

# BAD must be a factor for classification
hmeq$BAD <- factor(hmeq$BAD, levels = c(0, 1), labels = c("Good", "Bad"))
print(colSums(is.na(hmeq)))


# Creating NMAR Missing Indicator Flags BEFORE any imputation
hmeq$DEBTINC_MISSING <- ifelse(is.na(hmeq$DEBTINC), 1, 0)
hmeq$VALUE_MISSING   <- ifelse(is.na(hmeq$VALUE),   1, 0)
hmeq$DEROG_MISSING   <- ifelse(is.na(hmeq$DEROG),   1, 0)

# Recoding categorical missing as 'Unknown'
hmeq$JOB[is.na(hmeq$JOB)]       <- "Unknown"
hmeq$REASON[is.na(hmeq$REASON)] <- "Unknown"

# Converting to factors
hmeq$JOB    <- as.factor(hmeq$JOB)
hmeq$REASON <- as.factor(hmeq$REASON)
print(table(hmeq$JOB))
print(table(hmeq$REASON))


# Removing logically impossible records
# DEBTINC > 100 means total debt exceeds total income - not a real financial state
# CLAGE = 0 means oldest credit line is 0 months old - impossible for a home equity borrower
before <- nrow(hmeq)
hmeq <- hmeq[!(!is.na(hmeq$DEBTINC) & hmeq$DEBTINC > 100), ]
hmeq <- hmeq[!(!is.na(hmeq$CLAGE)   & hmeq$CLAGE == 0),    ]
after <- nrow(hmeq)

# Cap extreme values instead of deleting at 99th percentile, preserving real data
winsorize <- function(x, p = 0.99) {cap <- quantile(x, p, na.rm = TRUE)
x[!is.na(x) & x > cap] <- cap
return(x)}
hmeq$CLAGE   <- winsorize(hmeq$CLAGE)
hmeq$VALUE   <- winsorize(hmeq$VALUE)
hmeq$MORTDUE <- winsorize(hmeq$MORTDUE)
hmeq$LOAN    <- winsorize(hmeq$LOAN)

# EDA VISUALIZATIONS 
clean_theme <- theme_minimal(base_size = 13) + theme(
    plot.title       = element_text(face = "bold", size = 14, hjust = 0.5),
    plot.subtitle    = element_text(size = 11, hjust = 0.5, color = "gray40"),
    axis.title       = element_text(size = 11),
    axis.text        = element_text(size = 10),
    panel.grid.minor = element_blank(),
    plot.margin      = ggplot2::margin(15, 15, 15, 15) )

# PLOT 1: Class Imbalance
bad_counts <- as.data.frame(table(hmeq$BAD))
colnames(bad_counts) <- c("Status", "Count")
bad_counts$Label     <- c("Good (Repaid)", "Bad (Defaulted)")
bad_counts$Pct       <- round(bad_counts$Count / sum(bad_counts$Count) * 100, 1)
bad_counts$PctLabel  <- paste0(bad_counts$Count, "\n(", bad_counts$Pct, "%)")

ggplot(bad_counts, aes(x = Label, y = Count, fill = Label)) +
  geom_bar(stat = "identity", width = 0.45, color = "white") +
  geom_text(aes(label = PctLabel), vjust = -0.4,
            size = 4.5, fontface = "bold", color = "gray20") +
  scale_fill_manual(values = c("Good (Repaid)"   = "#4A90D9",
                               "Bad (Defaulted)" = "#E05C5C")) +
  scale_y_continuous(limits = c(0, 5800), expand = c(0, 0)) +
  labs(
    title    = "Loan Outcome Distribution",
    subtitle = "Dataset is imbalanced — only 1 in 5 borrowers defaulted",
    x = NULL, y = "Number of Loans") +
  clean_theme +
  theme(legend.position = "none")

# PLOT 2: Missing Value Rate by Variable
miss_vals <- colSums(is.na(hmeq))
miss_df   <- data.frame(Variable = names(miss_vals),Missing  = as.numeric(miss_vals))
miss_df <- miss_df[miss_df$Missing > 0, ]
miss_df$Rate <- round(miss_df$Missing / nrow(hmeq) * 100, 1)
miss_df <- miss_df[order(miss_df$Rate), ]
miss_df$Variable <- factor(miss_df$Variable, levels = miss_df$Variable)

ggplot(miss_df, aes(x = Variable, y = Rate, fill = Rate)) + geom_bar(stat = "identity", width = 0.6, color = "white") +
geom_text(aes(label = paste0(Rate, "%")),hjust = -0.15, size = 3.8, color = "gray20") + coord_flip() +
scale_fill_gradient(low = "#FFD580", high = "#E05C5C") + scale_y_continuous(limits = c(0, 28), expand = c(0, 0)) +
     labs(title = "Missing Value Rate by Variable",
     subtitle = "DEBTINC missing in 21% of applications — too high for simple imputation",
     x = NULL, y = "Missing Rate (%)") + clean_theme + theme(legend.position = "none")

# PLOT 3: NMAR Evidence - creating the DEBTINC_MISSING flag
nmar_df <- data.frame( Outcome = c("Good Borrowers", "Defaulters"),Missing_Rate = c(10.1, 66.1))
nmar_df$Outcome <- factor(nmar_df$Outcome,levels = c("Good Borrowers", "Defaulters"))

ggplot(nmar_df, aes(x = Outcome, y = Missing_Rate, fill = Outcome)) +geom_bar(stat = "identity", width = 0.45, color = "white") +
geom_text(aes(label = paste0(Missing_Rate, "%")),vjust = -0.5, size = 6, fontface = "bold", color = "gray20") +
scale_fill_manual(values = c("Good Borrowers" = "#4A90D9","Defaulters"  = "#E05C5C")) +
scale_y_continuous(limits = c(0, 80), expand = c(0, 0)) + 
  labs(title = "DEBTINC Missing Rate by Loan Outcome",
  subtitle = "Defaulters withheld their debt-to-income ratio 6x more often\nMissingness itself is a risk signal — not random",
  x = NULL, y = "Missing Rate (%)") + clean_theme +theme(legend.position = "none")

# PLOT 4: Default Rate by DELINQ Count
delinq_df <- data.frame(DELINQ = factor(c("0","1","2","3","4","5","6+"),levels = c("0","1","2","3","4","5","6+")),
Default_Rate = c(14.0, 33.9, 44.8, 55.0, 59.0, 81.6, 100.0), N = c(4179, 654, 250, 129, 78, 38, 52))

ggplot(delinq_df, aes(x = DELINQ, y = Default_Rate, fill = Default_Rate)) +geom_bar(stat = "identity", width = 0.6, color = "white") +
geom_text(aes(label = paste0(Default_Rate, "%")),vjust = -0.5, size = 4.2, fontface = "bold", color = "gray20") +
scale_fill_gradient(low = "#FFD580", high = "#8B0000") + scale_y_continuous(limits = c(0, 115), expand = c(0, 0)) +
   labs(title = "Default Rate by Number of Delinquent Credit Lines",
   subtitle = "Every borrower with 6 or more delinquencies defaulted — a perfect risk signal",
   x = "Number of Delinquent Credit Lines",y = "Default Rate (%)") +clean_theme +theme(legend.position = "none")

# PLOT 5: Boxplots — DEBTINC and CLAGE by Loan Outcome
# Using par() to put two boxplots side by side in base R
par(mfrow = c(1, 2), mar = c(5, 4, 4, 2))

# DEBTINC by outcome
boxplot(DEBTINC ~ BAD,data = hmeq[!is.na(hmeq$DEBTINC), ],col = c("#4A90D9", "#E05C5C"),names = c("Good", "Bad"),
main = "Debt-to-Income by Outcome",xlab = "Loan Outcome",ylab = "Debt-to-Income Ratio",outline = FALSE,frame = FALSE)

# CLAGE by outcome
boxplot(CLAGE ~ BAD,data = hmeq[!is.na(hmeq$CLAGE), ],col = c("#4A90D9", "#E05C5C"),names = c("Good", "Bad"),
main  = "Credit Age by Outcome",xlab = "Loan Outcome",ylab = "Age of Oldest Credit Line (Months)",outline = FALSE,frame = FALSE)
par(mfrow = c(1, 1))  

# SECTION 3: DATA PARTITIONING
set.seed(2026)
train_index <- createDataPartition(hmeq$BAD, p = 0.70, list = FALSE)
train_raw <- hmeq[ train_index, ]
test_raw  <- hmeq[-train_index, ]

cat("\nBAD distribution in training set:\n")
print(prop.table(table(train_raw$BAD)))
cat("\nBAD distribution in test set (should match training):\n")
print(prop.table(table(test_raw$BAD)))

# SECTION 4: MICE IMPUTATION
imp_train <- mice(train_raw, m = 5, maxit = 5, seed = 2026, printFlag = FALSE)
train_imputed <- complete(imp_train, 1)

imp_test <- mice(test_raw, m = 5, maxit = 5, seed = 2026, printFlag = FALSE)
test_imputed <- complete(imp_test, 1)


# Verifying no missing values remain
cat("\nMissing values after imputation:\n")
cat("Training set:", sum(is.na(train_imputed)), "\n")
cat("Test set:    ", sum(is.na(test_imputed)),  "\n")

# SECTION 5: HANDLE CLASS IMBALANCE WITH ROSE
cat("Training set before ROSE:\n")
print(table(train_imputed$BAD))
set.seed(2026)
train_balanced <- ROSE(BAD ~ ., data = train_imputed, seed = 2026)$data
cat("\nTraining set after ROSE:\n")
print(table(train_balanced$BAD))
cat("Test set stays untouched (real-world distribution):\n")
print(table(test_imputed$BAD))

# SECTION 6: CLASSIFICATION MODELS
ctrl <- trainControl(method = "repeatedcv",number = 10,repeats = 3,
classProbs = TRUE, summaryFunction = twoClassSummary,savePredictions = TRUE)

# Model 1: Logistic Regression
set.seed(2026)
logit_fit <- train(BAD ~ .,data = train_balanced, method = "glm",family = "binomial",trControl = ctrl,metric = "ROC")

# Model 2: Decision Tree
set.seed(2026)
tree_fit <- train(BAD ~ .,data  = train_balanced,method = "rpart",trControl = ctrl,metric = "ROC",tuneLength = 10)

# Model 3: Random Forest 
set.seed(2026)
rf_fit <- train(BAD ~ .,data  = train_balanced,method = "rf",trControl = ctrl,metric = "ROC",ntree = 500)

# 6.1 Cross-Validation Comparison
class_results <- resamples(list(LogisticRegression = logit_fit,DecisionTree = tree_fit,RandomForest = rf_fit))
print(summary(class_results))

# Boxplot comparison
bwplot(class_results,main = "Classification Model Comparison - Cross Validation Performance")

# Dot plot
dotplot(class_results, main = "Classification Model Comparison - Mean Performance")

# 6.2 Test Set Evaluation 
# Predictions on test set
logit_pred <- predict(logit_fit, newdata = test_imputed)
tree_pred  <- predict(tree_fit,  newdata = test_imputed)
rf_pred    <- predict(rf_fit,    newdata = test_imputed)

# Predicted probabilities for ROC curves
logit_prob <- predict(logit_fit, newdata = test_imputed, type = "prob")[, "Bad"]
tree_prob  <- predict(tree_fit,  newdata = test_imputed, type = "prob")[, "Bad"]
rf_prob    <- predict(rf_fit,    newdata = test_imputed, type = "prob")[, "Bad"]

# Confusion matrices
cat("\nLogistic Regression - Confusion Matrix:\n")
cm_logit <- confusionMatrix(logit_pred, test_imputed$BAD, positive = "Bad")
print(cm_logit)

cat("\nDecision Tree - Confusion Matrix:\n")
cm_tree <- confusionMatrix(tree_pred, test_imputed$BAD, positive = "Bad")
print(cm_tree)

cat("\nRandom Forest - Confusion Matrix:\n")
cm_rf <- confusionMatrix(rf_pred, test_imputed$BAD, positive = "Bad")
print(cm_rf)


# Side-by-Side Performance Comparison Table 
# ROC AUC for each model
roc_logit <- roc(test_imputed$BAD, logit_prob, levels = c("Good","Bad"))
roc_tree  <- roc(test_imputed$BAD, tree_prob,  levels = c("Good","Bad"))
roc_rf    <- roc(test_imputed$BAD, rf_prob,    levels = c("Good","Bad"))

class_summary <- data.frame( Model       = c("Logistic Regression", "Decision Tree", "Random Forest"),
  Accuracy    = round(c(cm_logit$overall["Accuracy"],cm_tree$overall["Accuracy"],cm_rf$overall["Accuracy"]), 4),
  Kappa       = round(c(cm_logit$overall["Kappa"],cm_tree$overall["Kappa"],cm_rf$overall["Kappa"]), 4),
  Sensitivity = round(c(cm_logit$byClass["Sensitivity"],cm_tree$byClass["Sensitivity"],cm_rf$byClass["Sensitivity"]), 4),
  Specificity = round(c(cm_logit$byClass["Specificity"],cm_tree$byClass["Specificity"],cm_rf$byClass["Specificity"]), 4),
  Precision   = round(c(cm_logit$byClass["Precision"],cm_tree$byClass["Precision"],cm_rf$byClass["Precision"]), 4),
  F1_Score    = round(c(cm_logit$byClass["F1"],cm_tree$byClass["F1"],cm_rf$byClass["F1"]), 4),
  AUC_ROC     = round(c(auc(roc_logit),auc(roc_tree),auc(roc_rf)), 4))

print(class_summary)


# 6.4 ROC Curves - All Three on One Plot
plot(1 - roc_logit$specificities, roc_logit$sensitivities,type = "l",col = "blue",lwd = 2,xlim = c(0,1),
ylim = c(0,1),xlab = "False Positive Rate",ylab = "True Positive Rate",main = "ROC Curves - Classification Models")
lines(1 - roc_tree$specificities, roc_tree$sensitivities,col = "red", lwd = 2)
lines(1 - roc_rf$specificities, roc_rf$sensitivities,col = "green", lwd = 2)
abline(0,1,lty=2,col="gray60")
legend("bottomright",legend=c(paste("Logistic Regression AUC =", round(auc(roc_logit),3)),
paste("Decision Tree AUC =", round(auc(roc_tree),3)),paste("Random Forest AUC =", round(auc(roc_rf),3)) ),
col=c("blue","red","green"),lwd=2)

# 6.5 Decision Tree Visualization

# Plot the final decision tree
final_tree <- rpart(BAD ~ ., data = train_balanced, cp = 0.01)
rpart.plot(final_tree,main = "Decision Tree - Loan Default Prediction",type = 4,extra = 104,fallen.leaves = TRUE)


# 6.6 Random Forest Variable Importance
var_imp <- varImp(rf_fit)
print(var_imp)
plot(var_imp, main = "Random Forest - Variable Importance for Default Prediction")

# SECTION 7: REGRESSION MODELS
# Removing BAD (outcome, not a predictor of DEBTINC)
# Removing DEBTINC_MISSING flag (it would perfectly predict when DEBTINC was missing)
# Keepping all other financial variables as predictors

reg_exclude <- c("BAD", "DEBTINC_MISSING")
train_reg <- train_imputed[, !(names(train_imputed) %in% reg_exclude)]
test_reg  <- test_imputed[,  !(names(test_imputed)  %in% reg_exclude)]

cat("Regression training set:", nrow(train_reg), "rows\n")
cat("Predictors:", ncol(train_reg) - 1, "(all variables except BAD and DEBTINC_MISSING)\n\n")

# Cross-validation control for regression
ctrl_reg <- trainControl(method  = "repeatedcv",number = 10,repeats = 3)

# Model 1: Linear Regression
set.seed(2026)
lm_fit <- train(DEBTINC ~ .,data = train_reg,method = "lm",trControl = ctrl_reg)

# Model 2: Regression Tree
set.seed(2026)
tree_reg_fit <- train(DEBTINC ~ .,data = train_reg,method = "rpart",trControl = ctrl_reg,tuneLength = 10)

# Model 3: Gradient Boosting
set.seed(2026)
gbm_fit <- train(DEBTINC ~ .,data = train_reg,method = "gbm",trControl = ctrl_reg,verbose = FALSE)


# 7.1 Cross-Validation Comparison
reg_results <- resamples(list(LinearRegression = lm_fit,RegressionTree = tree_reg_fit,GradientBoosting = gbm_fit))
print(summary(reg_results))

bwplot(reg_results,main = "Regression Model Comparison - Cross Validation Performance")
dotplot(reg_results,main = "Regression Model Comparison - Mean Performance")


# 7.2 Test Set Evaluation
lm_pred  <- predict(lm_fit,newdata = test_reg)
tree_reg_pred <- predict(tree_reg_fit, newdata = test_reg)
gbm_pred <- predict(gbm_fit,newdata = test_reg)

# Calculating MAE, RMSE, R-squared for each model
calc_metrics <- function(actual, predicted, model_name) 
{mae  <- mean(abs(actual - predicted))
rmse <- sqrt(mean((actual - predicted)^2)) 
ss_res <- sum((actual - predicted)^2)
ss_tot <- sum((actual - mean(actual))^2)
r2 <- 1 - ss_res / ss_tot
cat(model_name, "- MAE:", round(mae, 4),
      "| RMSE:", round(rmse, 4),
      "| R-squared:", round(r2, 4), "\n")
  return(c(MAE = mae, RMSE = rmse, R2 = r2))}

m1 <- calc_metrics(test_reg$DEBTINC, lm_pred,       "Linear Regression ")
m2 <- calc_metrics(test_reg$DEBTINC, tree_reg_pred, "Regression Tree   ")
m3 <- calc_metrics(test_reg$DEBTINC, gbm_pred,      "Gradient Boosting ")


# 7.3 Side-by-Side Regression Performance Table
reg_summary <- data.frame(Model = c("Linear Regression", "Regression Tree", "Gradient Boosting"),
MAE = round(c(m1["MAE"],  m2["MAE"],  m3["MAE"]),4),
RMSE = round(c(m1["RMSE"], m2["RMSE"], m3["RMSE"]), 4), 
R_Squared = round(c(m1["R2"], m2["R2"],   m3["R2"]),4))
print(reg_summary)


# 7.4 Regression Tree Visualization 

final_tree_reg <- rpart(DEBTINC ~ ., data = train_reg, cp = 0.01)
rpart.plot(final_tree_reg,main = "Regression Tree - Debt-to-Income Ratio Estimation",type = 4,extra = 101)

# 7.5 Actual vs Predicted Plot
plot(test_reg$DEBTINC, gbm_pred, xlab = "Actual DEBTINC",
     ylab = "Predicted DEBTINC", main = "Gradient Boosting - Actual vs Predicted DEBTINC",
     col = "steelblue", pch = 16)
abline(0, 1, col = "red", lwd = 2)
legend("topleft", legend = "Perfect Prediction Line", col = "red", lwd = 2)

# 7.6 Gradient Boosting Variable Importance 
gbm_imp <- varImp(gbm_fit)
print(gbm_imp)
plot(gbm_imp,main = "Gradient Boosting - Variable Importance for DEBTINC Estimation")

# SECTION 8: CLUSTERING
# Preparing clustering dataset
# Use imputed training data
# Exclude BAD (unsupervised - no outcome)
# Exclude categorical variables that k-means cannot handle directly
# Keep all numeric financial variables

clust_vars <- c("LOAN","MORTDUE","VALUE","YOJ","DEROG","DELINQ","CLAGE","NINQ","CLNO","DEBTINC","DEBTINC_MISSING","VALUE_MISSING","DEROG_MISSING")
clust_data <- train_imputed[, clust_vars]

# Scaling all variables 
clust_scaled <- scale(clust_data)
cat("Clustering variables:", length(clust_vars), "\n")
cat("Clustering records:", nrow(clust_scaled), "\n")


# 8.1 Elbow Method: Finding optimal number of clusters
set.seed(2026)
wss <- sapply(1:10, function(k) {kmeans(clust_scaled, centers = k, nstart = 25)$tot.withinss})
plot(1:10, wss,type = "b", pch = 19, col = "steelblue", lwd = 2,xlab = "Number of Clusters (k)",
ylab = "Total Within-Cluster Sum of Squares", main = "Elbow Method - Optimal Number of Borrower Segments")
abline(v = 4, col = "red", lty = 2, lwd = 2)


# 8.2 Silhouette Analysis - Confirm k 
library(cluster)
sil_scores <- sapply(2:8, function(k) {
  km  <- kmeans(clust_scaled, centers = k, nstart = 25, iter.max = 100)
  sil <- silhouette(km$cluster, dist(clust_scaled))
  mean(sil[, 3]) })

cat("\nSilhouette scores by k:\n")
for (i in seq_along(sil_scores)) {cat("k =", i + 1, ": silhouette =", round(sil_scores[i], 4), "\n")}
best_k <- which.max(sil_scores) + 1
cat("\nBest k based on silhouette:", best_k, "\n")

# 8.3 Final K-means Clustering
set.seed(2026)
km_final <- kmeans(clust_scaled, centers = best_k, nstart = 25)
cat("\nCluster sizes:\n")
print(table(km_final$cluster))

# Add cluster labels back to training data for profiling
train_clustered <- train_imputed
train_clustered$Cluster <- factor(km_final$cluster)


# 8.4 Cluster Profiling
# Now bring BAD back in to understand each cluster's risk level
# Default rate per cluster
cat("\nDefault rate by cluster:\n")
default_by_cluster <- tapply(train_clustered$BAD == "Bad",train_clustered$Cluster,mean)
print(round(default_by_cluster, 4))

# Average financial profile per cluster
profile_vars <- c("LOAN","MORTDUE","VALUE","DEBTINC","DELINQ","DEROG","CLAGE","YOJ","NINQ")
cat("\nAverage financial profile by cluster:\n")
cluster_profile <- aggregate(train_clustered[, profile_vars],by = list(Cluster = train_clustered$Cluster),FUN  = mean,na.rm = TRUE)
cluster_profile[, -1] <- round(cluster_profile[, -1], 2)
print(cluster_profile)


# 8.5 Cluster Visualization
# PLOT 1: Default Rate by Cluster
default_df <- data.frame(Cluster = paste("Cluster", names(default_by_cluster)),
                         Default_Rate = round(as.numeric(default_by_cluster) * 100, 1))

ggplot(default_df, aes(x = Cluster, y = Default_Rate,fill = ifelse(Default_Rate == max(Default_Rate), "red", "blue"))) +
geom_bar(stat = "identity", width = 0.5, color = "white") + geom_text(aes(label = paste0(Default_Rate, "%")),
vjust = -0.5, size = 5, fontface = "bold") + scale_fill_identity() + scale_y_continuous(limits = c(0, 105),
labels = function(x) paste0(x, "%")) + 
labs(title = "Default Rate by Borrower Segment", 
     subtitle = "Cluster 1 is a near-pure default segment",
     x = "Borrower Segment", y = "Default Rate (%)") +theme_minimal(base_size = 13) +
     theme(plot.title = element_text(face = "bold", hjust = 0.5),
plot.subtitle = element_text(hjust = 0.5, color = "gray40"), 
legend.position = "none",panel.grid.major.x = element_blank())

# PLOT 2: Cluster Profile - Bar Chart per Variable
# Simple grouped bar chart showing average value of key variables per cluster
profile_df <- data.frame(Cluster  = rep(paste("Cluster", 1:4), each = 4),
Variable = rep(c("Delinquencies","Derog Marks","Debt-to-Income","Credit Age (÷10)"), 4),
Value = c(
    # Cluster 1
    2.38, 1.09, 31.77, 17.19,
    # Cluster 2
    0.33, 0.24, 33.18, 16.62,
    # Cluster 3
    0.27, 0.16, 36.38, 20.97,
    # Cluster 4
    1.93, 0.47, 34.52, 18.28))

ggplot(profile_df, aes(x = Variable, y = Value, fill = Cluster)) +geom_bar(stat = "identity", position = "dodge", width = 0.7, color = "white") +
scale_fill_manual(values = c("#E05C5C","#4A90D9","#5BAD6F","#F0A500")) +
  labs(title = "Borrower Segment Financial Profiles",
  subtitle = "Average values per cluster across key risk variables",x = NULL, y = "Average Value",fill = "Segment") +
  theme_minimal(base_size = 13) +theme(plot.title = element_text(face = "bold", hjust = 0.5),
plot.subtitle = element_text(hjust = 0.5, color = "gray40"),axis.text.x = element_text(size = 10),panel.grid.major.x = element_blank())

# SECTION 9: FINAL SUMMARY

cat("--- CLASSIFICATION: Who will default? ---\n")
print(class_summary)

cat("\n--- REGRESSION: Estimating debt-to-income ratio ---\n")
print(reg_summary)

cat("\n--- CLUSTERING: Borrower segments ---\n")
cat("Number of segments identified:", best_k, "\n")
cat("Default rate per segment:\n")
print(round(default_by_cluster * 100, 2))
