# Graphe de calcul V1 non sémantique

La V1 construit un graphe partiel à partir de la question et des annotations
`<<expression=result>>`. Elle n’utilise ni entités ni relations sémantiques.

## Deux niveaux de représentation

Le parseur produit d’abord un arbre purement syntaxique :

```text
Number
Operation
```

Un `Number` indique uniquement qu’une valeur apparaît dans l’expression. Il ne
prétend pas encore connaître sa provenance.

Le constructeur de graphe remplace ensuite chaque `Number` par l’un des types
suivants :

| Type | Signification |
| --- | --- |
| `ProblemNumber` | occurrence numérique unique trouvée dans la question |
| `Reference` | résultat unique d’une étape précédente |
| `Literal` | aucune provenance trouvée |
| `Unresolved` | plusieurs provenances sont possibles |

## Politique de résolution

Pour chaque nombre utilisé dans une expression :

1. rechercher toutes les occurrences de même valeur dans la question, sous
   forme numérique (`4`, `1/3`) ou en lettres (`four`, `twice`) ;
2. rechercher toutes les étapes précédentes ayant produit exactement cette
   valeur ;
3. créer la provenance seulement s’il existe un candidat unique ;
4. utiliser `Unresolved` lorsqu’il existe plusieurs candidats ;
5. ne jamais choisir arbitrairement entre l’énoncé et une étape précédente.

Par exemple :

```text
question : la valeur 4 est donnée
step_0   : 6 - 2 = 4
step_1   : 4 + 1 = 5
```

Le `4` de `step_1` possède deux candidats :

```text
ProblemNumber(4)
Reference(step_index=0, value=4)
```

Il devient donc `Unresolved`. Il n’alimente pas la supervision de dépendance.
Pour vérifier l’exécution arithmétique de référence, sa valeur littérale exacte
reste toutefois disponible.

## Extraction conservatrice des nombres de l’énoncé

L’extraction combine trois couches, par priorité :

1. les formes numériques exactes (`4`, `0.5`, `3/8`) ;
2. les cardinaux anglais reconnus par `text2num` (`four`,
   `twenty-five`) ;
3. une petite grammaire exacte pour les fractions et multiplicateurs
   (`one-third`, `three quarters of`, `two and a half`, `half a dozen`,
   `twice`).

Les ordinaux et les mots trompeurs observés dans GSM8K (`ones`, `millions`,
`P.T.O.`) sont rejetés. Les unités ambiguës telles que `a pair of` et les
dénominateurs isolés tels que `quarters` ne sont pas convertis. Les candidats
qui se chevauchent sont départagés en faveur de la forme composée la plus
précise.

Une division n’est remplacée par une fraction de l’énoncé que si sa structure
correspond aussi à la source : `3/8` peut être relié à `3/8`, tandis qu’un
calcul `1.5/3` n’est pas confondu avec `one-half` malgré la même valeur finale.

Les expressions écrites avec `//` conservent leur exécution par plancher, mais
elles utilisent la même classe de supervision `div` que les expressions
écrites avec `/`. Il n’existe donc pas de classe d’apprentissage
`floor_div` séparée.

## Compilation en actions postfixées

Le graphe résolu est ensuite compilé en un petit programme déterministe. Le
compilateur parcourt chaque `expression_tree` de gauche à droite et en
post-ordre : les opérations imbriquées sont émises avant l’opération qui les
consomme.

Par exemple, `1 + 1 / 2` devient :

```text
action_0 = DIV(Literal(1), Literal(2))
action_1 = ADD(Literal(1), LocalResult(action_0))
```

Chaque occurrence d’opérande est représentée par une référence ordonnée :

| Référence | Signification |
| --- | --- |
| `ProblemNumber` | pointeur vers une occurrence exacte de la question |
| `PreviousResult` | résultat final d’une étape précédente |
| `LocalResult` | résultat d’une action antérieure de la même étape |
| `Literal` | constante sans provenance identifiée |
| `Unresolved` | valeur exacte connue, mais plusieurs provenances candidates |

Une expression réduite à une feuille, par exemple une fraction `1/3` reconnue
directement dans la question, reçoit une action explicite
`COPY(ProblemNumber(...))`. Chaque étape valide possède donc toujours une
sortie de programme.

Les pointeurs sont validés à la construction :

- `PreviousResult(step_index)` doit viser une étape strictement antérieure ;
- `LocalResult(action_index)` doit viser une action déjà émise ;
- `ProblemNumber(problem_number_index)` doit viser une occurrence présente
  dans la table du programme.

### Masques locaux par opérande

Chaque action produit un `operand_mask` aligné sur ses opérandes. Les références
`ProblemNumber`, `PreviousResult`, `LocalResult` et `Literal` valent `true`.
Seule l’occurrence `Unresolved` vaut `false`.

Ainsi :

```text
ADD(Unresolved(4), Literal(1))
operand_mask = [false, true]
```

L’ambiguïté masque uniquement la première position. L’opérateur `ADD`, le
second opérande et les autres actions de l’étape restent supervisables.

### Échelle de la cible numérique

Chaque étape compilée expose directement :

```text
target_scale = max(abs(target_result), 1.0)
```

Cette valeur permet de normaliser `result_loss` et `execution_loss` avant
`smooth_l1`, sans laisser les quelques très grandes cibles dominer
l’optimisation. Le résultat et l’exécution de référence restent calculés
exactement avec `Fraction` ; seul le facteur destiné à l’apprentissage est
exporté comme scalaire flottant.

## Exécution

Le graphe est exécuté dans l’ordre des étapes avec `Fraction` :

- `ProblemNumber`, `Literal` et `Unresolved` fournissent leur valeur exacte ;
- `Reference` réutilise le résultat calculé de l’étape précédente ;
- chaque résultat reconstruit est comparé à la cible exacte de l’étape.

Il faut distinguer :

- **graphe exécutable** : toutes les étapes reproduisent leur cible ;
- **provenance complète** : aucune occurrence n’est `Unresolved`.

Un graphe peut donc être arithmétiquement exécutable tout en restant
partiellement supervisable.

Le programme postfixé possède un second exécuteur exact, indépendant de
l’exécuteur récursif du graphe. L’audit compare les deux résultats étape par
étape. Une compilation n’est validée que si :

```text
program_result[step] == graph_result[step]
```

Le JSON d’audit conserve le `program`, son `program_evaluation` et la
`program_verification` exacte pour chaque exemple.

## Audit parallèle

La commande suivante lit le train complet, construit et exécute les graphes
avec huit processus, puis conserve le détail de chaque exemple :

```bash
uv run python script/inspect_math_graphs.py \
  --split train \
  --sample-size 6352 \
  --workers 8 \
  --seed 42 \
  --output-file outputs/math_postfix_audit_train_full_seed42_workers8.json
```

Le JSON produit contient la question, la réponse, les étapes mathématiques, le
graphe, les candidats ambigus, les actions postfixées, les masques locaux et
les deux résultats d’exécution de chaque étape.

L’audit structurel et les résultats consolidés sur train, validation et test
sont détaillés dans
[`postfix_audit_all_splits.md`](postfix_audit_all_splits.md).
