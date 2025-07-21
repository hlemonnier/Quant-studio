import joblib
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# === 1. Chargement du modèle original ===
print("🔄 Chargement du modèle original...")
model = joblib.load('strategies/Signal/Machine_Learning/stacking_model.joblib')

# === 2. Nettoyage du XGBClassifier ===
print("🧹 Nettoyage du XGBClassifier...")
xgb = model.named_estimators_['xgb']
xgb_params = xgb.get_params()

# Suppression du paramètre obsolète
xgb_params.pop('use_label_encoder', None)

# Reconstruire un XGBClassifier propre
xgb_clean = XGBClassifier(**xgb_params)

# === 3. Reconstruction du modèle Stacking ===
print("🛠 Reconstruction du modèle StackingClassifier...")
estimators = [
    ('xgb', xgb_clean),
    ('et', model.named_estimators_['et']),
    ('gbc', model.named_estimators_['gbc']),
    ('knn', model.named_estimators_['knn']),
]

final_estimator = model.final_estimator_

model_clean = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    passthrough=model.passthrough,
    cv=model.cv,
    stack_method=model.stack_method
)

# === 4. Sauvegarde ===
print("💾 Sauvegarde du modèle nettoyé...")
joblib.dump(model_clean, 'strategies/Signal/Machine_Learning/stacking_model_clean.joblib')

print("✅ Modèle nettoyé sauvegardé avec succès dans stacking_model_clean.joblib")
