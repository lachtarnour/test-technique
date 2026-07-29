# Audit strict des programmes postfixés

Date de l’audit : 28 juillet 2026.

## Portée

L’audit couvre les trois splits gelés de GSM8K :

| Split | Exemples | Étapes mathématiques | Actions postfixées |
| --- | ---: | ---: | ---: |
| train | 6 352 | 19 828 | 22 746 |
| validation | 1 121 | 3 531 | 4 036 |
| test | 1 319 | 4 206 | 4 790 |
| **total** | **8 792** | **27 565** | **31 572** |

Chaque exemple est reconstruit avec 8 workers. Une seconde exécution complète
avec 1 worker est comparée par hash à l’exécution parallèle.

## Contrôles effectués

Pour chaque programme, l’audit vérifie :

1. l’égalité exacte entre le résultat du graphe récursif et celui du programme
   postfixé ;
2. la correspondance structurelle de chaque sous-arbre en post-ordre, et pas
   uniquement du résultat final ;
3. l’ordre et la répétition des opérandes ;
4. les opérateurs d’exécution et de supervision ;
5. les pointeurs `ProblemNumber`, `PreviousResult` et `LocalResult` ;
6. la conservation exacte des candidats de `Unresolved` ;
7. les dépendances entre étapes ;
8. l’absence d’action manquante, supplémentaire ou morte ;
9. les masques locaux, avec `false` uniquement pour `Unresolved` ;
10. `target_scale = max(abs(target_result), 1.0)` et sa finitude ;
11. la sérialisation JSON complète ;
12. le déterminisme entre 1 et 8 workers.

La suite inclut également 1 000 arbres synthétiques déterministes contenant
les opérateurs binaires, les opérateurs unaires, des fractions, des valeurs
négatives, des divisions et des divisions entières.

## Résultat critique

| Contrôle | Résultat |
| --- | ---: |
| Étapes structurellement vérifiées | 27 565 / 27 565 |
| Problèmes structurels | **0** |
| Résultats différents du graphe | **0** |
| Divergences 1 worker / 8 workers | **0 / 8 792 exemples** |
| Cibles non finies | **0** |
| Suite de tests automatisés | **réussie** |
| Ruff | **réussi** |

Des tests de corruption volontaire confirment que le validateur détecte
l’inversion des opérandes d’une soustraction ainsi que l’ajout d’une action
morte.

## Couverture des opérandes

Le programme contient 63 143 positions d’opérande :

| Référence | Occurrences |
| --- | ---: |
| `ProblemNumber` | 27 615 |
| `PreviousResult` | 18 041 |
| `LocalResult` | 4 007 |
| `Literal` | 6 189 |
| `Unresolved` | 7 291 |

Les 7 291 occurrences `Unresolved` sont les seules positions masquées :
88,453 % des opérandes restent supervisables. Les références répétées sont
conservées occurrence par occurrence ; l’audit en trouve 103 qui auraient été
perdues par une simple matrice de dépendances ensembliste.

Le nombre de candidats par occurrence ambiguë est :

| Candidats | Occurrences |
| ---: | ---: |
| 2 | 6 006 |
| 3 | 1 005 |
| 4 | 210 |
| 5 | 49 |
| 6 | 19 |
| 7 | 1 |
| 8 | 1 |

## Complexité observée

- 3 290 étapes, soit 11,94 %, contiennent plusieurs actions ;
- le maximum observé est de 11 actions dans une étape ;
- 103 occurrences réutilisent plusieurs fois le même résultat précédent ;
- les deux `floor_div` du train s’exécutent comme divisions entières, mais
  utilisent la classe de supervision `div`.

Les `target_scale` sont tous finis :

| Statistique | Valeur |
| --- | ---: |
| minimum | 1 |
| médiane | 36 |
| percentile 90 | 780 |
| percentile 99 | 68 000 |
| maximum | 2 920 000 000 |

## Limites détectées dans les annotations

Ces points ne sont pas des erreurs du compilateur, mais doivent être traités
explicitement par le pipeline d’entraînement :

1. **218 exemples sans étape exploitable, soit 2,48 %** :
   113 réponses ne contiennent aucune annotation `<<...>>` et 105 ne
   contiennent que des identités supprimées comme `<<7=7>>`. La loss langage
   reste active, mais toutes les losses auxiliaires doivent être masquées.
2. **505 exemples avec des étapes mais sans calcul final annoté, soit 5,74 %** :
   le dernier calcul menant à `####` est écrit en texte libre. Il ne faut pas
   inventer une étape cible ; la supervision auxiliaire s’arrête à la dernière
   annotation fiable.
3. **7 291 opérandes ambigus, soit 11,55 %** : ils sont exécutables grâce à
   leur valeur exacte, mais leur provenance est masquée localement.
4. **6 189 littéraux sans provenance identifiée** : ils restent des cibles
   `Literal` valides, mais pourront faire l’objet d’un audit séparé si la V2
   cherche une couverture sémantique plus élevée.
5. La provenance est entièrement résolue pour 5 664 exemples sur 8 792,
   soit 64,42 %. Ce taux mesure la certitude de provenance, pas la correction
   arithmétique.

## Artifacts

- `outputs/math_postfix_strict_audit_train_full_seed42_workers8.json`
- `outputs/math_postfix_strict_audit_validation_full_seed42_workers8.json`
- `outputs/math_postfix_strict_audit_test_full_seed42_workers8.json`

Le script `script/inspect_math_graphs.py` échoue désormais explicitement si
une divergence structurelle ou numérique est détectée. Le script
`script/verify_graph_audit_determinism.py` compare les résultats complets entre
1 et 8 workers.
