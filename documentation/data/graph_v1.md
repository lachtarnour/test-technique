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

1. rechercher toutes les occurrences de même valeur dans la question ;
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

## Audit parallèle

La commande suivante lit un échantillon déterministe, construit et exécute les
graphes avec quatre processus, puis conserve le détail de chaque exemple :

```bash
uv run python script/inspect_math_graphs.py \
  --split train \
  --sample-size 1000 \
  --workers 4 \
  --seed 42
```

Le JSON produit contient la question, la réponse, les étapes mathématiques, le
graphe, les candidats ambigus et le résultat d’exécution de chaque étape.
