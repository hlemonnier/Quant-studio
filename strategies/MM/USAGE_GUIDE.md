# 🚀 Market Making V1 – Guide d’Utilisation

---

## 🎯 Comment lancer le programme

### ✅ RÉSOLUTION DU BUG « Spread 0.0 bps »
Le spread nul provenait d’un horizon `T` trop long et d’une mauvaise formule ; le correctif applique :
1. `T = 120/86400` (2 minutes)  
2. Formule fermée : `δ* = (1/γ) × ln(1 + γ/k)`  
3. Ré-intégration du centre OFI (±1 tick)

---

### 🚀 Lancement du système

#### 1️⃣ Paper Trading (recommandé pour débuter)
```bash
# Trading papier illimité
python strategies/MM/main.py --mode=paper-trading --symbol=BTCUSDT

# Trading papier limité à 2 h
python strategies/MM/main.py --mode=paper-trading --duration=2

# Verbosité maximale
python strategies/MM/main.py --mode=paper-trading --log-level=DEBUG
```

#### 2️⃣ Backtesting & Validation
```bash
# Validation complète
python strategies/MM/main.py --mode=backtest

# Exécution directe du script de backtest
python strategies/MM/backtesting_v1.py
```

#### 3️⃣ Calibration automatique des paramètres
```bash
python strategies/MM/main.py --mode=calibration
```

#### 4️⃣ Trading live (⚠️ argent réel)
```bash
export BINANCE_API_KEY="votre_api_key"
export BINANCE_API_SECRET="votre_api_secret"
python strategies/MM/main.py --mode=live --symbol=BTCUSDT
```

---

## 📊 Modes disponibles

| Mode             | Description                | Sécurité | Prérequis |
|------------------|----------------------------|----------|-----------|
| paper-trading    | Simulation temps réel      | ✅       | Aucun     |
| backtest         | Test historique            | ✅       | Aucun     |
| calibration      | Optimisation paramètres    | ✅       | Aucun     |
| live             | Trading réel               | ⚠️       | API Keys  |

---

## 🔧 Tests unitaires & rapides

```bash
# Suite math / algo
pytest strategies/MM/tests/test_v1_algo.py -v

# Vérification rapide du spread
python strategies/MM/test_spread_fix.py

# Tableau KPI interactif
python strategies/MM/kpi_tracker.py
```

---

## 📈 Surveillance en temps réel

Le système affiche automatiquement un tableau récapitulatif :

```
🤖 Market Making V1 Strategy
📅 2025-07-22 14:30:00
🎯 Mode: paper-trading
💰 Symbol: BTCUSDT

🔧 Configuration Market Making V1
========================================
Symboles: BTCUSDT
Risk Aversion (γ): 0.1
Inventory Max: 1.0
Spread range: 5–200 bps
OFI β: 0.3 | Window: 1.0 s
========================================

🚀 Starting Paper Trading for BTCUSDT
💡 Press Ctrl+C to stop gracefully

📊 BTCUSDT Performance Report (Real-time)
============================================================
Current Mid: $50 123.45
Inventory: 0.1234 BTC
Total Quotes: 1 250
Total Fills: 89  (7.1 % fill ratio ✅)
Spread: 12.5 bps
PnL: +$45.67

KPI Status: 5/6 targets met ✅
============================================================
```

---

## ⚡ Dépannage rapide

### ❌ Spread toujours à 0 bps ?
```bash
python - <<'PY'
from strategies.MM.config import mm_config
print(f"γ={mm_config.gamma}, k={mm_config.k}, T={mm_config.T}")
PY

python strategies/MM/test_spread_fix.py
```

### 🔧 Ajuster les paramètres
Éditer `strategies/MM/config.py` :
```python
# Spread plus large (plus conservateur)
self.gamma = 0.2

# Spread plus serré (plus agressif)
self.gamma = 0.05

# Taille de quote réduite
self.base_quote_size = 0.005
```

### 📊 Monitoring avancé
```bash
# Logs en continu
tail -f logs/mm_v1_$(date +%Y%m%d).log

# KPI toutes les 5 s
watch -n 5 "python - <<'PY'
from strategies.MM.kpi_tracker import KPITracker
print(KPITracker('BTCUSDT').get_summary())
PY"
```

---

## 🎯 KPI suivis automatiquement

| Métrique              | Cible          |
|-----------------------|---------------|
| Spread capturé        | ≥ 70 %        |
| RMS inventaire        | ≤ 0.4         |
| Fill ratio            | ≥ 5 %         |
| Cancel ratio          | ≤ 70 %        |
| Latence P99           | ≤ 300 ms      |
| PnL total             | Positif       |

---

## 🚦 Workflow recommandé

1. **Test rapide** : `python strategies/MM/test_spread_fix.py`  
2. **Backtest** : `python strategies/MM/main.py --mode=backtest`  
3. **Calibration** : `python strategies/MM/main.py --mode=calibration`  
4. **Paper trading** : `python strategies/MM/main.py --mode=paper-trading --duration=1`  
5. **Live** : seulement après validation complète

---

## 💡 Support & ressources

- **Logs** : `logs/mm_v1_<date>.log`  
- **Configuration** : `strategies/MM/config.py`  
- **Tests** : `pytest`  
- **Arrêt d’urgence** : `Ctrl+C` (shutdown propre)

**Le système V1 est maintenant prêt pour le déploiement !** 🎉
