# Métriques d’évaluation GSM8K

Cette page décrit les métriques calculées à partir des résultats produits par le
[parseur](parsing.md).

L’évaluation distingue trois aspects :

1. l’exactitude de la réponse finale ;
2. l’exactitude des calculs intermédiaires ;
3. la cohérence entre le dernier calcul et la réponse finale.

Une réponse peut donc contenir le bon résultat tout en présentant un raisonnement
arithmétique incorrect.

## Comparaison numérique

Deux valeurs sont comparées dans l’ordre suivant :

1. égalité exacte ;
2. égalité après arrondi à la précision de la prédiction ;
3. égalité après troncature à cette même précision.

Pour une valeur de référence égale à `1/3` :

| Prédiction | Considérée comme correcte |
| --- | ---: |
| `0.3`, `0.33`, `0.333` | oui |
| `0.3333333333` | oui |
| `0.2`, `0.34`, `0.3334` | non |

La tolérance dépend uniquement de la précision exprimée par la prédiction. Une
valeur proche, mais incompatible avec l’arrondi ou la troncature de la
référence, reste incorrecte.

Pour une valeur de référence égale à `6/7` :

| Prédiction | Règle appliquée | Correcte pour l’accuracy |
| --- | --- | ---: |
| `0.857142857` | troncature à neuf décimales | oui |
| `0.86` | arrondi à deux décimales | oui |
| `0.85` | troncature à deux décimales | oui |
| `0.87` | ni arrondi ni troncature | non |

Une approximation acceptée est donc **correcte pour l’accuracy**, sans être
présentée comme une égalité mathématique exacte. La valeur exacte reste stockée
en interne sous forme de `Fraction`.

Cette règle est utilisée pour comparer :

- la réponse finale à la référence ;
- le résultat annoncé d’une formule à son résultat calculé ;
- le résultat de la dernière formule à la réponse finale.

## Métriques par réponse

### Exactitude de la réponse finale

Deux indicateurs sont calculés :

```text
strict_correct =
    marqueur terminal #### valide
    ET valeur correcte

correct =
    valeur extraite par le marqueur strict ou par fallback
    ET valeur correcte
```

Pour une référence égale à `3` :

| Sortie | `strict_correct` | `correct` |
| --- | ---: | ---: |
| `#### 3` | oui | oui |
| `#### 3 months` | non | oui |
| aucune valeur exploitable | non | non |

Le fallback permet de distinguer une erreur de format d’une erreur de calcul.
L’accuracy mesure ensuite la proportion de problèmes résolus.

### Erreur numérique

L’accuracy indique si une réponse est correcte, mais ne mesure pas l’écart entre
une mauvaise prédiction et la référence.

La métrique `final_answer_error` utilise une erreur relative symétrique :

```text
si p = r = 0 : e(p, r) = 0
sinon        : e(p, r) = 2 × |p - r| / (|p| + |r|)
```

où `p` désigne la prédiction et `r` la référence.

Cette erreur possède les propriétés suivantes :

- `0` lorsque les deux valeurs sont numériquement égales ;
- `2` pour l’erreur maximale ;
- elle est indépendante de l’échelle ;
- une prédiction absente reçoit une erreur égale à `2`.

Une approximation acceptée par la règle de précision peut conserver une petite
erreur non nulle, car cette métrique utilise directement la distance numérique.

L’accuracy mesure donc la réussite, tandis que l’erreur mesure la proximité.

### Exactitude des formules

Pour une réponse contenant `F` formules :

```text
step_accuracy =
    nombre de formules correctes / F
```

Lorsqu’aucune formule n’est présente :

```text
step_accuracy = 0
```

L’indicateur `all_steps_correct` est défini par :

```text
all_steps_correct =
    au moins une formule présente
    ET toutes les formules sont correctes
```

Une formule correctement parsée mais arithmétiquement fausse n’est pas
considérée comme correcte.

### Cohérence interne

La cohérence interne vérifie que le raisonnement arithmétique produit le même
résultat que la réponse finale :

```text
internal_arithmetic_consistency =
    all_steps_correct
    ET marqueur terminal #### valide
    ET résultat calculé par la dernière formule
       égal à la réponse finale
```

| Réponse | Cohérente |
| --- | ---: |
| `<<6-2=4>>` puis `#### 4` | oui |
| `<<3+4=7>>` puis `#### 8` | non |
| `#### 7` sans formule | non |
| réponse correcte obtenue uniquement par fallback | non |

Une réponse cohérente doit donc comporter :

- au moins une formule ;
- uniquement des formules correctes ;
- une réponse terminale au format strict ;
- un accord entre le dernier calcul et cette réponse.

La métrique combinée est définie par :

```text
correct_and_internally_consistent =
    correct
    ET internal_arithmetic_consistency
```

## Métriques agrégées

Pour un ensemble de `N` réponses :

| Métrique | Définition |
| --- | --- |
| `strict_final_answer_accuracy` | réponses `strict_correct` divisées par `N` |
| `final_answer_accuracy` | réponses `correct` divisées par `N` |
| `final_answer_error` | moyenne des erreurs relatives symétriques |
| `mean_step_arithmetic_accuracy` | moyenne des valeurs `step_accuracy` |
| `internal_arithmetic_consistency_rate` | réponses cohérentes divisées par `N` |
| `correct_and_internally_consistent_rate` | réponses correctes et cohérentes divisées par `N` |

### Agrégation des formules

`mean_step_arithmetic_accuracy` est une macro-moyenne calculée en trois étapes :

1. calculer la proportion de formules correctes pour chaque réponse ;
2. attribuer une valeur de `0` aux réponses sans formule ;
3. faire la moyenne de ces proportions.

Chaque réponse possède ainsi le même poids, indépendamment du nombre de formules
qu’elle contient.

### Diagnostics complémentaires

| Diagnostic | Définition |
| --- | --- |
| `final_answer_format_compliance_rate` | réponses avec un marqueur terminal valide divisées par `N` |
| `formula_parse_rate` | formules parsables divisées par le nombre total de formules |
| `elapsed_seconds` | durée totale de l’évaluation |
| `samples_per_second` | `N / elapsed_seconds` |

Contrairement à l’accuracy arithmétique moyenne, `formula_parse_rate` est une
micro-moyenne globale calculée au niveau des formules.

## Rapport JSON

Le rapport d’évaluation contient :

- le modèle, le split utilisé et les paramètres de génération ;
- les métriques agrégées ;
- les textes générés et les valeurs extraites ;
- l’analyse de chaque formule ;
- les erreurs de parsing, d’exécution et d’arithmétique.

## Limite de l’évaluation

Ces métriques vérifient la syntaxe des annotations et la validité des calculs.
Elles ne permettent pas de déterminer si les opérations choisies répondent
réellement à la question.

```text
<<10+9=19>>
#### 19
```

Cette sortie est cohérente sur le plan arithmétique, même si l’opération
effectuée peut être sans rapport avec l’énoncé.

## Fichiers concernés

- `src/evaluation/reasoning.py` : analyse des réponses et agrégation des
  métriques ;
- `script/evaluation.py` : exécution de l’évaluation et sauvegarde du rapport.
