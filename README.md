# Description
Data taken from competition
https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview. Solved the Binary
classification problem for predicting the probability of loan default (target = 1). 
- An example of baseline model training is shown.
- 3 models selected: CatBoost, XGBoost, LightGBM, hyperparameters have been found for them.
- Ensemble models created: Voting, Blending, Stacking classifiers.
- Feature importance using SHAP researched.

___
## Notebooks
- `EDA` - Exploratory data analysis
- `Baseline` - Training baseline models
- `Tuning` - Selection of hyperparameters
- `Ensembling` - Creating Ensemble Models
- `Post analysis` - SHAP values, finance
