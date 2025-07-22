# 🚀 Market Making V1 Strategy – Guide Final

## 📁 Structure du projet
```
strategies/MM/
├── config.py                 # ⚙️ Configuration centralisée
├── main.py                   # 🚀 Point d’entrée principal
├── avellaneda_stoikov.py     # 📊 Calculs A&S + OFI
├── trading_engine.py         # 🔄 Moteur de trading temps-réel
├── kpi_tracker.py            # 📈 Suivi des performances
├── backtesting_v1.py         # 🧪 Tests historiques
├── parameter_calibration.py  # 🔧 Optimisation paramètres
├── ofi.py                    # 📊 Order-Flow Imbalance
├── local_book.py             # 📖 Reconstruction carnet
├── inventory_control.py      # ⚖️ Gestion inventaire
└── tests/                    # 🧪 Tests unitaires & intégration
    ├── test_v1_algo.py
    ├── test_v1_complete.py
    └── test_spread_fix.py
```

---

## ⚙️ Configuration

**Tout se règle dans `config.py`**.

### Symboles
```python
self.symbols = ['BTCUSDT', 'ETHUSDT']
```

### Paramètres Avellaneda-Stoikov
```python
self.gamma = 0.1          # Aversion au risque (↑ = spread ↑)
self.sigma = 0.02         # Volatilité initiale
self.T     = 120/86400    # Horizon (2 min)
self.k     = 1.5          # Impact de marché
```

### Limites & Risques
```python
self.max_inventory  = 1.0   # Position max
self.min_spread_bps = 5     # Spread mini (5 bps)
self.max_spread_bps = 200   # Spread maxi (200 bps)
self.base_quote_size = 0.01 # Taille d’ordre
```

### Paramètres OFI
```python
self.beta_ofi           = 0.3  # Sensibilité OFI
self.ofi_window_seconds = 1.0  # Fenêtre de calcul
```

---

## 🚀 Utilisation

### Paper Trading (recommandé)
```bash
python strategies/MM/main.py --mode=paper-trading
python strategies/MM/main.py --mode=paper-trading --duration=2          # 2 h
python strategies/MM/main.py --mode=paper-trading --symbol=ETHUSDT
```

### Backtesting
```bash
python strategies/MM/main.py --mode=backtest
```

### Calibration
```bash
python strategies/MM/main.py --mode=calibration
```

### Live Trading ⚠️
```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
python strategies/MM/main.py --mode=live --symbol=BTCUSDT
```

---

## 🧪 Tests

```bash
# Test rapide du spread
pytest strategies/MM/tests/test_spread_fix.py -v

# Suite unitaire
pytest strategies/MM/tests/test_v1_algo.py -v

# Intégration complète
pytest strategies/MM/tests/test_v1_complete.py -v

# Tous les tests
pytest strategies/MM/tests -v
```

---

## 📊 Monitoring temps réel

```bash
python strategies/MM/kpi_tracker.py                    # Dashboard KPI
tail -f logs/mm_v1_$(date +%Y%m%d).log                 # Logs
```

### KPI suivis
• Spread capturé ≥ 70 %  
• RMS inventaire ≤ 0.4  
• Fill ratio ≥ 5 %  
• Cancel ratio ≤ 70 %  
• Latence P99 ≤ 300 ms  
• PnL total positif  

---

## ⚙️ Personnalisation rapide

```python
# config.py – profil conservateur
self.gamma          = 0.2
self.max_inventory  = 0.5
self.min_spread_bps = 10

# profil agressif
self.gamma          = 0.05
self.max_inventory  = 2.0
self.min_spread_bps = 3
```

### Ajouter un nouveau symbole
```python
# config.py
self.symbols.append('ADAUSDT')
```
```bash
python strategies/MM/main.py --mode=paper-trading --symbol=ADAUSDT
```

---

## 🛑 Arrêt & Contrôles

• **Ctrl +C** : arrêt gracieux + résumé  
• Pause auto si :  
  – inventaire > limite  
  – volatilité > 2× baseline  
  – latence > 300 ms  
  – stop-loss déclenché  

---

## 📈 Workflow conseillé

1. `pytest strategies/MM/tests -v`  
2. `python strategies/MM/main.py --mode=backtest`  
3. `python strategies/MM/main.py --mode=calibration`  
4. `python strategies/MM/main.py --mode=paper-trading --duration=1`  
5. `python strategies/MM/main.py --mode=live` *(après validation)*  

---

## ✅ Prêt pour la production
Le Market Maker V1 est **production-ready** :  
• Algorithme A&S + OFI validé  
• Moteur temps réel robuste  
• Contrôles de risque intégrés  
• KPI tracking en live  
• Tests unitaires & intégration  
• Configuration unique dans `config.py`

**Bon trading ! 🎯**
