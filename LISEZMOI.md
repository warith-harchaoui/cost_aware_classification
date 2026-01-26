# 🚀 Classification Sensible au Coût + Benchmark Fraude (IEEE-CIS)

**Auteur :** Warith Harchaoui <wharchaoui@nexton-group.com>

Dépôt de recherche et d'ingénierie d'entreprise pour la **classification sensible au coût** avec des **coûts de mauvaise classification dépendants de l'exemple**. Cette boîte à outils transforme l'apprentissage automatique traditionnel du "simple rapprochement d'étiquettes" à la "**maximisation du profit commercial**".

## 🎯 Le Problème Métier

La classification traditionnelle (Entropie Croisée) traite toutes les erreurs comme égales. Dans le monde réel, **certaines erreurs sont beaucoup plus coûteuses que d'autres** :
- **Faux Déclin :** Refuser un client légitime coûte la marge de la transaction + la frustration du client + une possible perte de clientèle (churn).
- **Fausse Approbation (Fraude) :** Accepter une carte volée coûte le montant total de la transaction + les frais de rétrofacturation (chargeback) + les coûts opérationnels.

Ce dépôt implémente des fonctions de perte basées sur le **Transport Optimal (OT)** qui "comprennent" ces coûts pendant l'entraînement, permettant aux modèles de prendre des décisions qui minimisent le regret financier plutôt que de simplement compter les erreurs.

---

## 📍 Table des Matières

- [Le Problème Métier](#-le-problème-métier)
- [Démarrage Rapide pour les Décideurs](#-démarrage-rapide-pour-les-décideurs)
- [Fonctions de Perte Disponibles](#-fonctions-de-perte-disponibles)
  - [Pertes de Référence (Baselines)](#1-pertes-de-référence-baselines)
  - [Pertes Sensibles au Coût (Transport Optimal)](#2-pertes-sensibles-au-coût-transport-optimal)
- [Guide de Réglage d'Epsilon (ε)](#️-guide-de-réglage-depsilon-ε)
- [Conseils de Performance](#-conseils-de-performance)
- [Guide d'Utilisation Complet](#-guide-dutilisation-complet)
- [Métriques et Évaluation](#-métriques-et-évaluation)
- [Choisir une Fonction de Perte](#-choisir-une-fonction-de-perte)
- [Documentation et Ressources](#-documentation)
- [Tests](#-tests)
- [Citation](#-citation)
- [Licence](#-licence)

---

## 💡 Démarrage Rapide pour les Décideurs

Si vous souhaitez voir immédiatement l'impact métier, lancez le benchmark complet :

```bash
# Comparer toutes les pertes par rapport aux références financières
python -m examples.fraud_detection --loss all --epochs 15 --run-id impact_metier
```

**Ce qu'il faut regarder dans les résultats :**
- **Regret Réalisé :** L'argent réellement perdu en production.
- **Regret Optimal Attendu :** La perte théorique minimale possible.
- **Référence Naïve (Naive Baseline) :** Ce qui se passe si vous faites simplement "Tout Approuver" ou "Tout Refuser".

---

## 📋 Fonctions de Perte Disponibles

### Pertes de Référence (Baselines)

#### 1. **Entropie Croisée** (`cross_entropy`)
Perte d'entropie croisée standard sans sensibilisation au coût.

**Quand l'utiliser :** Comparaison de base lorsque toutes les mauvaises classifications ont un coût égal.

#### 2. **Entropie Croisée Pondérée** (`cross_entropy_weighted`)
Entropie croisée pondérée par échantillon avec des poids $w_i = C_i[y_i, 1-y_i]$ dérivés de la matrice de coût.

**Quand l'utiliser :** Baseline simple sensible au coût qui repondère les exemples par leur coût de mauvaise classification.

### Pertes Sensibles au Coût (Transport Optimal)

Toutes les pertes basées sur l'OT utilisent une matrice de coût $C$ où $C_{ij}$ représente le coût de prédire la classe $j$ quand la classe réelle est $i$.

#### Comprendre la Régularisation Epsilon (ε)

Le paramètre de régularisation entropique ε contrôle la fluidité du transport optimal. **Par défaut, ε est calculé de manière adaptative à partir de la matrice de coût.**

**Avantages d'ε adaptatif :**
- S'adapte automatiquement à l'ampleur de votre matrice de coût.
- Aucun réglage manuel requis.
- Robuste à travers différents domaines.

#### 3. **Perte Sinkhorn-Fenchel-Young** (`sinkhorn_fenchel_young`)
Utilise le théorème de l'enveloppe pour des gradients stables. Idéal pour une différenciation implicite.

#### 4. **Perte Sinkhorn Envelope** (`sinkhorn_envelope`)
Implémentation personnalisée avec gradients d'enveloppe. Efficace en mémoire et stable.

#### 5. **Perte Sinkhorn Full Autodiff** (`sinkhorn_autodiff`)
Différenciation complète à travers toutes les itérations de Sinkhorn. Plus "bout-en-bout" mais consomme plus de mémoire.

#### 6. **Perte Sinkhorn POT** (`sinkhorn_pot`) ⭐
Utilise la bibliothèque reconnue [Python Optimal Transport (POT)](https://pythonot.github.io/).
- **Recommandé pour la production.**
- Meilleure stabilité numérique.

---

## 🎛️ Guide de Réglage d'Epsilon (ε)

Le paramètre `--epsilon-scale` multiplie l'ε adaptatif :
- **< 1.0 :** Régularisation plus serrée, décisions plus tranchées.
- **= 1.0 :** Équilibre par défaut.
- **> 1.0 :** Régularisation plus souple, solutions plus robustes face au bruit.

---

## 📊 Métriques et Évaluation

Pour mesurer réellement le succès commercial, nous allons au-delà de la Précision et de l'AUC-ROC :

- **PR-AUC (Aire sous la courbe Précision-Rappel) :** Métrique principale pour les données de fraude déséquilibrées (**plus c'est haut, mieux c'est**).
- **Luck Baseline :** Ligne horizontale représentant un classificateur aléatoire.
- **Regret Optimal Attendu :** Coût métier attendu si nous prenons la décision mathématiquement optimale (**plus c'est bas, mieux c'est**).
- **Regret Réalisé :** L'argent réellement perdu. Inclut les pertes dues à la fraude acceptée et le manque à gagner des faux déclins.
- **Référence Naïve :** Compare la meilleure des stratégies simples ("Tout Approuver" ou "Tout Refuser"). **Votre modèle doit battre cette référence pour être utile.**

---

## ⚡ Conseils de Performance

### Chargement de Données Robuste
Nous recommandons l'utilisation du moteur Python pour lire les fichiers CSV volumineux d'IEEE-CIS afin d'éviter les erreurs `ParserError`.

### Stabilité Numérique
- **`RobustScaler`** : Pour gérer les valeurs aberrantes des montants.
- **`CosineAnnealingLR`** : Pour une convergence plus fluide.
- **Taux d'Apprentissage Faibles** : Commencer à `1e-5` pour des gradients plus stables.

---

## 🎯 Choisir une Fonction de Perte

| Perte | Avantages | Inconvénients | Idéal pour |
|-------|-----------|---------------|------------|
| `cross_entropy` | Simple, rapide | Ignore les coûts | Comparaison de base |
| `sinkhorn_pot` | **Prêt pour la production** | Dépendance externe | Déploiements réels ⭐ |
| `sinkhorn_envelope` | Stable, peu de mémoire | Implémentation maison | Mémoire limitée |

---

## ✍️ Citation

Si vous utilisez ce travail, merci de citer :

```bibtex
@inproceedings{harchaoui2026cacis,
  title={Cost-Aware Classification with Optimal Transport for E-commerce Fraud Detection},
  author={Harchaoui, Warith and Pantanacce, Laurent},
  booktitle={The 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '26)},
  year={2026}
}
```

## 📜 Licence

**Unlicense** — Ce logiciel est libre et appartient au domaine public.  
Voir [UNLICENSE](unlicense.org) pour plus de détails.
