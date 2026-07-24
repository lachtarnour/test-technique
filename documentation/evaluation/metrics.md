# Métriques d’évaluation GSM8K

Cette page explique comment les sorties du
[parseur](parsing.md) deviennent des scores.

## Vue d’ensemble

Pour chaque réponse, l’évaluateur mesure séparément :

1. l’exactitude du résultat final ;
2. l’exactitude des formules ;
3. la cohérence entre la dernière formule et le résultat final.

Une bonne réponse finale peut donc avoir un raisonnement invalide.

## 1. Comparaison numérique

La comparaison suit cet ordre :

1. égalité exacte ;
2. arrondi à la précision produite ;
3. troncature à cette même précision.

Pour une référence égale à `1/3` :

| Prédiction | Correcte |
| --- | ---: |
| `0.3`, `0.33`, `0.333` | oui |
| `0.3333333333` | oui |
| `0.2`, `0.34`, `0.3334` | non |

La tolérance porte uniquement sur la précision numérique. Une valeur proche
mais incompatible avec l’arrondi ou la troncature reste incorrecte.

Cette règle sert à comparer :

- la réponse finale à la référence ;
- le résultat annoncé d’une formule à son résultat calculé ;
- la dernière formule à la réponse finale.

## 2. Évaluation d’une réponse

### Accuracy de la réponse finale

```text
strict_correct =
    marqueur #### terminal valide
    ET valeur correcte

correct =
    valeur stricte ou fallback
    ET valeur correcte
```

| Sortie | `strict_correct` | `correct` si référence = `3` |
| --- | ---: | ---: |
| `#### 3` | oui | oui |
| `#### 3 months` | non | oui |
| aucune valeur exploitable | non | non |

Le fallback permet de mesurer la capacité mathématique sans confondre une
erreur de format avec une erreur de résultat.

L’accuracy indique la proportion de problèmes résolus.


### Erreur de la réponse finale

L’accuracy indique combien de problèmes sont résolus. Elle ne précise pas
l’importance des erreurs. `final_answer_error` ajoute cette information avec
une erreur relative symétrique :

```text
si p = r = 0 : e(p, r) = 0
sinon        : e(p, r) = 2 × |p - r| / (|p| + |r|)
```

`p` est la prédiction et `r` la référence.

Propriétés :

- `0` : prédiction et référence numériquement égales ;
- `2` : erreur maximale ;
- valeur indépendante de l’échelle ;
- prédiction absente : erreur fixée à `2`.

La métrique utilise la distance numérique réelle. Une approximation acceptée
par la règle de précision peut donc garder une petite erreur non nulle.


#### --> L’erreur mesure la proximité ; l’accuracy mesure la réussite.

### Formules

Pour une réponse contenant `F` formules :

```text
step_accuracy =
    formules correctes / F
```

Sans formule, `step_accuracy = 0`.

```text
all_steps_correct =
    au moins une formule
    ET toutes les formules sont correctes
```

Une formule parsable mais arithmétiquement fausse n’est pas correcte.

### Cohérence interne

```text
internal_arithmetic_consistency =
    all_steps_correct
    ET marqueur #### terminal valide
    ET résultat calculé par la dernière formule
       égal à la réponse finale
```

| Réponse | Cohérente |
| --- | ---: |
| `<<6-2=4>>` puis `#### 4` | oui |
| `<<3+4=7>>` puis `#### 8` | non |
| `#### 7` sans formule | non |
| réponse correcte obtenue par fallback | non |

La cohérence exige donc :

- au moins une formule ;
- toutes les formules correctes ;
- une réponse terminale stricte ;
- l’accord entre la dernière formule et cette réponse.

```text
correct_and_internally_consistent =
    correct
    ET internal_arithmetic_consistency
```

## 3. Métriques agrégées

Pour `N` réponses :

| Métrique | Calcul |
| --- | --- |
| `strict_final_answer_accuracy` | réponses `strict_correct` / `N` |
| `final_answer_accuracy` | réponses `correct` / `N` |
| `final_answer_error` | moyenne des erreurs relatives symétriques |
| `mean_step_arithmetic_accuracy` | moyenne des `step_accuracy` |
| `internal_arithmetic_consistency_rate` | réponses cohérentes / `N` |
| `correct_and_internally_consistent_rate` | réponses correctes et cohérentes / `N` |

### Agrégation des étapes

`mean_step_arithmetic_accuracy` est une macro-moyenne :

- calculer le taux de formules correctes de chaque réponse ;
- attribuer `0` aux réponses sans formule ;
- faire la moyenne des taux.

Chaque réponse a le même poids, quel que soit son nombre de formules.

### Diagnostics complémentaires

| Diagnostic | Calcul |
| --- | --- |
| `final_answer_format_compliance_rate` | réponses avec marqueur terminal / `N` |
| `formula_parse_rate` | formules parsables / toutes les formules |
| `elapsed_seconds` | durée totale |
| `samples_per_second` | `N / elapsed_seconds` |

`formula_parse_rate` est une micro-moyenne globale par formule.

## 4. Rapport JSON

Le rapport conserve :

- le modèle, le split et les paramètres de génération ;
- les métriques agrégées ;
- les textes générés et les valeurs extraites ;
- l’analyse de chaque formule ;
- les erreurs de parsing, d’exécution et d’arithmétique.

## Limite essentielle

Les métriques vérifient la syntaxe et l’arithmétique, pas la pertinence du
raisonnement par rapport à la question.

```text
<<10+9=19>>
#### 19
```

Cette réponse est cohérente arithmétiquement, même si l’opération choisie ne
répond pas à l’énoncé.

Fichiers principaux :

- `src/evaluation/reasoning.py` : analyse et agrégation ;
- `script/evaluation.py` : exécution et sauvegarde du rapport.
