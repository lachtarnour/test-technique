# Parsing des réponses GSM8K

Cette page décrit le traitement appliqué aux sorties générées par le modèle afin
d’en extraire :

- une réponse finale normalisée ;
- les éventuelles annotations arithmétiques.

La définition des scores calculés à partir de ces informations est disponible
dans [metrics.md](metrics.md).

## Vue d’ensemble

```text
texte généré
  ├─→ extraction de la réponse terminale ####
  │     └─→ fallback si le format strict est absent
  └─→ extraction des annotations <<...>>
        └─→ validation AST et calcul arithmétique
```

## Réponse finale

### Format strict

Une réponse est considérée comme strictement valide si le texte se termine par
l’un des formats suivants :

```text
#### nombre
#### numérateur/dénominateur
```

L’espace après `####` est facultatif. Aucun texte, symbole ou unité ne doit
suivre la valeur.

| Fin de réponse | Valide | Valeur normalisée |
| --- | ---: | ---: |
| `#### 42` | oui | `42` |
| `#### -.5` | oui | `-0.5` |
| `#### 3/5` | oui | `0.6` |
| `#### 3 months` | non | — |
| `#### 3/5 kg` | non | — |
| `#### 1/0` | non | — |

Dans le dernier exemple, la valeur est reconnue comme une fraction, puis rejetée
car son dénominateur est nul.

### Fallback numérique

Lorsqu’aucune réponse terminale valide n’est trouvée, le parseur :

1. recherche les nombres et fractions autonomes présents dans le texte ;
2. conserve la dernière valeur trouvée ;
3. la normalise selon les mêmes règles que la réponse stricte.

| Sortie | Valeur obtenue par fallback |
| --- | ---: |
| `#### 3 months` | `3` |
| `#### 3/5 kg` | `0.6` |
| `#### 1/0` | aucune |

Exemple :

```text
John pays 3.6 in a 30-day month.
```

Le fallback retourne `3.6`. La valeur `30` est ignorée, car elle fait partie de
l’expression composée `30-day`.

Ce mécanisme est uniquement syntaxique. Il ne cherche pas à comprendre le sens
de la phrase. Pour la sortie suivante :

```text
40 liters in 2 buckets
```

la valeur retournée est donc `2`.

La provenance de la valeur extraite est enregistrée dans le résultat du
parsing :

- `final_marker` lorsque le marqueur terminal est valide ;
- `fallback` lorsque la dernière valeur autonome est utilisée ;
- `None` lorsqu’aucune valeur exploitable n’est trouvée.

## Formats numériques

Le parseur reconnaît les formats suivants :

- entiers : `42`, `-7` ;
- nombres avec séparateurs de milliers : `1,000`, `12,500.25` ;
- décimaux : `3.5`, `.5`, `-.5` ;
- fractions : `3/5`, `-6/8`, `1.5/.5`.

Les formats suivants sont refusés :

- notation scientifique, par exemple `1e3` ;
- valeurs non finies ;
- fractions dont le dénominateur est nul ;
- signe `+` devant une réponse finale.

Les nombres sont analysés avec `Decimal`, puis les fractions avec `Fraction`.
Aucun calcul ne repose sur un `float` binaire.

| Entrée | Valeur normalisée |
| --- | ---: |
| `.5` | `0.5` |
| `12.000` | `12` |
| `3/5` | `0.6` |
| `-6/8` | `-0.75` |
| `1/3` | `0.3333333333` |

Une fraction dont l’écriture décimale est finie conserve sa valeur exacte. Les
fractions périodiques sont enregistrées avec dix décimales.

## Annotations arithmétiques

Les calculs intermédiaires doivent être écrits sous la forme :

```text
<<expression=result>>
```

Toutes les annotations sont conservées dans leur ordre d’apparition.

Lorsqu’une annotation commence par `<<` mais ne contient pas de fermeture `>>`,
elle est tout de même enregistrée avec l’erreur `unclosed_annotation`.

### Grammaire acceptée

| Élément | Valeurs acceptées |
| --- | --- |
| Opérateurs binaires | `+`, `-`, `*`, `/`, `//` |
| Signes unaires | `+`, `-` |
| Groupement | parenthèses |
| Valeurs | entiers et décimaux |
| Division `//` | division entière par plancher |

Exemples valides :

```text
<<6-2=4>>
<<1+1+2+4=8>>
<<-2*.25=-.5>>
<<30*(1/3)=10>>
<<8//2=4>>
```

La partie située après le dernier `=` doit être un entier ou un nombre décimal.

```text
<<1/3=0.3>>   valide
<<1/3=1/3>>   invalide
```

Les fractions sont donc acceptées dans l’expression, mais pas dans le résultat
annoncé.

Le prompt d’entraînement demande uniquement les opérateurs `+`, `-`, `*` et
`/`. Le parseur accepte également `//`. Une modification du parseur ne modifie
donc pas automatiquement le format demandé au modèle.

## Évaluation sécurisée des expressions

Les expressions ne sont jamais exécutées avec `eval`.

```text
expression
  → construction de l’arbre AST
  → validation des nœuds
  → exécution récursive
  → résultat sous forme de Fraction
```

Seuls les nœuds et opérateurs explicitement autorisés sont exécutés.

| Formule refusée | Motif |
| --- | --- |
| `<<19=19>>` | aucune opération |
| `<<2+2>>` | signe `=` absent |
| `<<2+2=4=4>>` | expression invalide |
| `<<5%2=1>>` | modulo non autorisé |
| `<<2**3=8>>` | puissance non autorisée |
| `<<x+1=2>>` | variable non autorisée |
| `<<1/0=0>>` | division par zéro |

Des limites supplémentaires sont appliquées :

- 256 caractères par expression ;
- 64 nœuds AST ;
- 50 chiffres par littéral ;
- 256 chiffres par numérateur ou dénominateur intermédiaire.

## Résultat associé à une formule

Chaque annotation produit notamment les champs suivants :

| Champ | Description |
| --- | --- |
| `expression` | partie située avant le dernier `=` |
| `claimed_result` | résultat annoncé, après normalisation |
| `evaluated_result` | résultat obtenu par le calcul |
| `parse_success` | indique si la syntaxe et l’AST sont valides |
| `execution_success` | indique si l’exécution s’est terminée correctement |
| `arithmetic_correct` | indique si le résultat annoncé est acceptable |
| `error` | erreur éventuellement détectée |
| `is_correct` | vrai si le parsing, l’exécution et le résultat sont valides |

Les erreurs possibles sont :

```text
missing_equals
empty_expression
invalid_claimed_result
invalid_expression
execution_error
incorrect_result
unclosed_annotation
```

## Fichiers concernés

- `src/evaluation/generation.py` : extraction de la réponse finale ;
- `src/evaluation/numeric.py` : parsing des nombres et des fractions ;
- `src/evaluation/arithmetic.py` : extraction, validation AST et exécution des
  annotations.
