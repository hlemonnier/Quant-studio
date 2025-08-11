# Structure du Projet Market Making

## Organisation des Modules

```
strategies/MM/
├── main.py                    # Point d'entrée principal
├── trading_engine.py          # Moteur de trading principal
├── simple_backtest.py         # Backtesting simple
│
├── core/                      # Signaux de trading fondamentaux
│   ├── __init__.py
│   ├── depth_imbalance.py     # Calcul du Depth Imbalance (DI)
│   └── ofi.py                 # Order Flow Imbalance (OFI)
│
├── data_capture/              # Capture et gestion des données
│   ├── __init__.py
│   ├── local_book.py          # Order book local (REST + WebSocket)
│   ├── ws_integration.py      # Intégration WebSocket ↔ LocalBook
│   └── ws_data_capture.py     # Capture WebSocket standalone
│
├── market_making/             # Stratégies de market making
│   ├── __init__.py
│   ├── avellaneda_stoikov.py  # Stratégie A&S classique
│   ├── avellaneda_stoikov_v15.py  # Version V1.5 améliorée
│   └── quote_manager.py       # Gestion des quotes
│
├── utils/                     # Utilitaires et configuration
│   ├── __init__.py
│   ├── config.py              # Configuration globale
│   ├── kpi_tracker.py         # Suivi des KPIs
│   ├── inventory_control.py   # Contrôle d'inventaire
│   ├── parameter_calibration.py  # Calibration des paramètres
│   └── performance_validator.py   # Validation des performances
│
└── tests/                     # Tests unitaires et d'intégration
    ├── unit/
    └── integration/
```

## Flux de Données

### 1. Capture des Données
```
Binance WebSocket → BinanceDepthStreamCapture → WSLocalBookIntegration → LocalBook
```

### 2. Signaux de Trading
```
LocalBook → DepthImbalanceCalculator → DI Signal
LocalBook → OFICalculator → OFI Signal
```

### 3. Génération de Quotes
```
DI + OFI + Market Data → AvellanedaStoikovV15Quoter → Optimal Quotes
```

### 4. Exécution
```
Quotes → QuoteManager → Trading Engine → Simulated Fills
```

## Problème Diagnostiqué

**Symptôme**: Les valeurs du LocalBook restent figées malgré la connexion WebSocket.

**Cause Racine**: La méthode `apply_ws_update()` était manquante dans la classe `LocalBook`.

**Solution Implémentée**:
1. ✅ Ajout de `apply_ws_update()` dans `LocalBook`
2. ✅ Amélioration du logging pour diagnostic
3. ✅ Réorganisation de la structure des modules

## Prochaines Étapes

1. **Tester** la nouvelle intégration WebSocket
2. **Valider** que les données du LocalBook se mettent à jour
3. **Vérifier** que le DI calcule des valeurs dynamiques
4. **Optimiser** les performances si nécessaire

