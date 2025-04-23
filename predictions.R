# Chargement des bibliothèques nécessaires
library(readxl)    # Pour lire les fichiers Excel
library(ggplot2)   # Pour la visualisation des données
library(dplyr)     # Pour la manipulation des données
library(caret)     # Pour la division des données et l'évaluation des modèles
library(scales)    # Pour la mise en forme des graphiques
library(car)       # Pour les tests statistiques supplémentaires
library(broom)    # Pour nettoyer les sorties des modèles

# 1. Importation des données
tryCatch({
  data <- read_excel("data.xlsx")
}, error = function(e) {
  stop("Erreur lors de la lecture du fichier. Vérifiez le chemin et le format du fichier.")
})

# Vérification des données importées
if(nrow(data) == 0) {
  stop("Le fichier de données est vide ou n'a pas été correctement lu.")
}

# 2. Nettoyage des données
cat("Nombre initial d'observations:", nrow(data), "\n")

# Suppression des NA et des doublons
data_clean <- data %>% 
  na.omit() %>% 
  distinct()

cat("Nombre d'observations après nettoyage:", nrow(data_clean), "\n")

# Vérification de la perte de données
if(nrow(data_clean) < nrow(data)*0.7) {
  warning("Plus de 30% des données ont été supprimées lors du nettoyage. Vérifiez la qualité des données.")
}

# 3. Statistiques descriptives
summary_stats <- data_clean %>% 
  summarise_all(list(
    mean = mean, 
    sd = sd, 
    min = min, 
    max = max,
    median = median,
    q25 = ~quantile(., 0.25),
    q75 = ~quantile(., 0.75),
    missing = ~sum(is.na(.))
  ))

print("Statistiques descriptives:")
print(summary_stats)

# 4. Exploration visuelle des données
# Boxplot
boxplot(data_clean$Close, main = "Distribution des prix de clôture", ylab = "Prix")

# Histogramme avec densité
ggplot(data_clean, aes(x = Close)) + 
  geom_histogram(aes(y = ..density..), bins = 30, fill = "lightblue", color = "black") +
  geom_density(color = "blue", size = 1) +
  ggtitle("Distribution des prix de clôture") +
  theme_minimal()

# Matrice de corrélation (si plusieurs variables)
if(ncol(data_clean) > 2) {
  numeric_data <- data_clean %>% select(where(is.numeric))
  cor_matrix <- cor(numeric_data)
  print("Matrice de corrélation:")
  print(cor_matrix)
  
  # Visualisation des corrélations
  corrplot::corrplot(cor_matrix, method = "circle")
}

# 5. Préparation des données
# Normalisation des données
norm_data <- data_clean %>% 
  mutate(across(where(is.numeric), ~ scale(.) %>% as.vector))

# Vérification de la normalisation
cat("Moyennes après normalisation:\n")
print(colMeans(norm_data %>% select(where(is.numeric))))

cat("\nÉcarts-types après normalisation:\n")
print(apply(norm_data %>% select(where(is.numeric)), 2, sd))

# 6. Division des données
set.seed(123) # Pour la reproductibilité
train_index <- createDataPartition(norm_data$Close, p = 0.8, list = FALSE)
train_data <- norm_data[train_index, ]
test_data <- norm_data[-train_index, ]

cat("Taille de l'ensemble d'entraînement:", nrow(train_data), "\n")
cat("Taille de l'ensemble de test:", nrow(test_data), "\n")

# 7. Construction du modèle
model <- lm(Close ~ ., data = train_data)

# Résumé détaillé du modèle
model_summary <- summary(model)
print(model_summary)

# Diagnostic du modèle
par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))

# Test de normalité des résidus
shapiro_test <- shapiro.test(residuals(model))
cat("\nTest de Shapiro-Wilk pour la normalité des résidus:\n")
print(shapiro_test)

# Test d'hétéroscédasticité
ncv_test <- car::ncvTest(model)
cat("\nTest d'hétéroscédasticité:\n")
print(ncv_test)

# 8. Évaluation du modèle
predictions <- predict(model, newdata = test_data)

# Métriques d'évaluation
r2 <- R2(predictions, test_data$Close)
rmse <- RMSE(predictions, test_data$Close)
mae <- MAE(predictions, test_data$Close)

cat("\nMétriques d'évaluation:\n")
cat("R²:", round(r2, 3), "\n")
cat("RMSE:", round(rmse, 3), "\n")
cat("MAE:", round(mae, 3), "\n")

# Visualisation des résultats
results <- data.frame(
  Actual = test_data$Close,
  Predicted = predictions,
  Residual = test_data$Close - predictions
)

# Graphique des prédictions vs valeurs réelles
ggplot(results, aes(x = Actual, y = Predicted)) +
  geom_point(color = "darkgreen", alpha = 0.6, size = 2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red", size = 1) +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(title = "Prédictions vs Valeurs Réelles",
       subtitle = paste("Modèle linéaire - R² =", round(r2, 3), "RMSE =", round(rmse, 3)),
       x = "Valeurs Réelles (normalisées)",
       y = "Prédictions (normalisées)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Graphique des résidus
ggplot(results, aes(x = Predicted, y = Residual)) +
  geom_point(color = "darkred", alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "blue") +
  labs(title = "Analyse des résidus", 
       x = "Prédictions", 
       y = "Résidus") +
  theme_minimal()

# 9. Prédiction sur de nouvelles données (exemple)
# Remplacez ces valeurs par vos propres données
new_data_example <- data.frame(
  Variable1 = mean(data_clean$Variable1, na.rm = TRUE),
  Variable2 = median(data_clean$Variable2, na.rm = TRUE),
  Variable3 = min(data_clean$Variable3, na.rm = TRUE)
)

# Assurez-vous que les noms des colonnes correspondent à votre jeu de données
names(new_data_example) <- names(data_clean)[1:3] # Ajustez selon vos variables

# Normalisation des nouvelles données (en utilisant les paramètres d'origine)
new_data_norm <- new_data_example %>% 
  mutate(across(everything(), ~ (.-mean(data_clean[[cur_column()]]))/sd(data_clean[[cur_column()]])))

prediction_new <- predict(model, newdata = new_data_norm)

# Conversion de la prédiction normalisée à l'échelle originale
prediction_original_scale <- prediction_new * sd(data_clean$Close) + mean(data_clean$Close)

cat("\nPrédiction pour les nouvelles données (échelle normalisée):", prediction_new, "\n")
cat("Prédiction pour les nouvelles données (échelle originale):", prediction_original_scale, "\n")

# 10. Sauvegarde des résultats
write.csv(results, "predictions_resultats.csv", row.names = FALSE)

# Sauvegarde du modèle
saveRDS(model, "modele_regression_lineaire.rds")

# Sauvegarde des statistiques descriptives
write.csv(summary_stats, "statistiques_descriptives.csv", row.names = FALSE)

# 11. Rapport automatique des résultats
sink("rapport_analyse.txt")
cat("Rapport d'analyse - Modèle de régression linéaire\n")
cat("===============================================\n\n")
cat("Date de l'analyse:", as.character(Sys.time()), "\n\n")

cat("1. Statistiques descriptives\n")
cat("----------------------------\n")
print(summary_stats)

cat("\n2. Performances du modèle\n")
cat("-------------------------\n")
cat("R²:", r2, "\n")
cat("RMSE:", rmse, "\n")
cat("MAE:", mae, "\n")

cat("\n3. Coefficients du modèle\n")
cat("-------------------------\n")
print(model_summary$coefficients)

cat("\n4. Tests de diagnostic\n")
cat("-----------------------\n")
cat("Test de normalité des résidus (Shapiro-Wilk): p-value =", shapiro_test$p.value, "\n")
cat("Test d'hétéroscédasticité (NCV): p-value =", ncv_test$p, "\n")
sink()

# Message de fin
cat("\nAnalyse terminée. Les résultats ont été sauvegardés dans:\n")
cat("- predictions_resultats.csv\n")
cat("- modele_regression_lineaire.rds\n")
cat("- statistiques_descriptives.csv\n")
cat("- rapport_analyse.txt\n")