# Market Making V1 - Rule-based Strategy

## 🎯 Objectif

Stratégie de Market Making **rule-based** (sans ML/RL) basée sur le modèle **Avellaneda-Stoikov** avec contrôle d'inventaire et skew automatique.

## 📋 Roadmap V1

| Étape | Objectif | Livrable | Status |
|-------|----------|----------|---------|
| 1️⃣ | **WS → parquet** | Capturer depth20@100ms pour chaque symbole | ✅ |
| 2️⃣ | **Book local** | Rejouer snapshot REST + diff stream pour L2 propre | ✅ |
| 3️⃣ | **Quoting "statique"** | Prix = Avellaneda-Stoikov (spread optimal) | ✅ |
| 4️⃣ | **Contrôle d'inventaire** | Skew dès que l'inventaire dépasse les seuils | ✅ |
| 5️⃣ | **Back-test latence 0** | Rejouer une journée, log PnL, Δinventory | ✅ |

## 🏗️ Architecture

```


## 🚀 Usage

### Configuration

1. **Variables d'environnement** (optionnel pour live trading):
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

2. **Paramètres** dans `config.py`:
```python
# À définir avec le boss
gamma = 0.1           # Risk aversion (γ)
max_inventory = 1.0   # N_max
inventory_threshold = 0.5  # N★
symbols = ['BTCUSDT', 'ETHUSDT']
```

### Tests des composants

```bash
# Test complet
python -m strategies.MM.main test

# Test spécifique
python -m strategies.MM.main test --component ws
python -m strategies.MM.main test --component quotes
```

### Backtesting

```bash
# Backtest avec données automatiques
python -m strategies.MM.main backtest --symbol BTCUSDT --date 2024-01-01

# Backtest avec fichier spécifique
python -m strategies.MM.main backtest --symbol BTCUSDT --data path/to/data.parquet
```

### Stratégie Live

```bash
python -m strategies.MM.main live
```

## 📊 Modèle Avellaneda-Stoikov

### Formules clés

**Prix de réservation:**
```
r = S - q × γ × σ² × (T - t)
```

**Spread optimal:**
```
δ = γ × σ² × (T - t) + (2/γ) × ln(1 + γ/k)
```

**Quotes finaux:**
```
bid = r - δ/2
ask = r + δ/2
```

### Paramètres

- **S**: Prix mid actuel
- **q**: Inventaire actuel
- **γ**: Aversion au risque (0.1 par défaut)
- **σ**: Volatilité estimée
- **T-t**: Temps restant jusqu'à l'horizon
- **k**: Paramètre d'impact de marché (1.5)

## 🎛️ Contrôle d'inventaire

### Seuils

- **N★** (inventory_threshold): Seuil de déclenchement du skew
- **N_max** (max_inventory): Inventaire maximum autorisé

### Skew automatique

Quand `|inventaire| > N★`:
- **Long**: Favorise la vente (décale les prix vers le bas)
- **Short**: Favorise l'achat (décale les prix vers le haut)

## 📈 Métriques de performance

### Backtesting

- **PnL total**: Profit/Loss réalisé + non réalisé
- **Sharpe Ratio**: Rendement ajusté du risque
- **Max Drawdown**: Perte maximale depuis un pic
- **Taux de réussite**: % de trades gagnants
- **Distribution inventaire**: Vérification de la centrage

### Live trading

- **PnL temps réel**: Suivi continu
- **Inventaire**: Contrôle des limites
- **Spreads**: Monitoring des conditions de marché

## 🔧 Étapes immédiates

### 1. Tests de validation

- [ ] **Connexion WS**: Vérifier ping/pong & latence
- [ ] **Flux données**: Sauver 15-30 min, contrôler les trous de séquence
- [ ] **LocalBook**: Coder + snapshot complet
- [ ] **Avellaneda-Stoikov**: Implémenter et valider

### 2. Paramétrage

À définir avec le boss:
- **γ** (risk aversion)
- **N★** et **N_max** (seuils inventaire)
- **Symboles live** (BTCUSDT, ETHUSDT ?)

### 3. Validation backtesting

Vérifier sur une journée:
- [ ] **PnL moyen positif** ?
- [ ] **Variance du PnL raisonnable** ?
- [ ] **Distribution inventaire centrée** ?

## 🔗 Références

- **Paper**: "High-frequency trading in a limit order book" - Avellaneda & Stoikov (2008)
- **GitHub**: [fedecaccia/avellaneda-stoikov](https://github.com/fedecaccia/avellaneda-stoikov)
- **API Binance**: [Documentation WebSocket](https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams)

## 📝 Notes

- **V1**: Rule-based uniquement, pas de ML/RL
- **Latence**: 0ms pour le backtesting
- **Phase 2**: RL sera ajouté après validation V1
- **Data**: Parquet quotidien pour replay
- **Risk Management**: Stop-loss et limites d'inventaire

---

**🚨 Important**: Cette stratégie est en développement. Tests approfondis requis avant utilisation en production. 