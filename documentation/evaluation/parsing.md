# Parsing des réponses GSM8K

Cette page décrit comment un texte généré devient :

- une réponse finale normalisée ;
- une liste de formules analysées.

Le calcul des scores est décrit dans [metrics.md](metrics.md).

## Vue d’ensemble

```text
texte généré
  ├─→ réponse terminale ####
  │     └─→ fallback si nécessaire
  └─→ annotations <<...>>
        └─→ AST sécurisé → résultat arithmétique
```

## 1. Extraction de la réponse finale

### Format strict

La sortie doit se terminer par :

```text
#### nombre
#### numérateur/dénominateur
```

L’espace après `####` est facultatif. Aucun texte ni unité ne peut suivre la
valeur.

| Fin de réponse | Valide | Valeur normalisée |
| --- | ---: | ---: |
| `#### 42` | oui | `42` |
| `#### -.5` | oui | `-0.5` |
| `#### 3/5` | oui | `0.6` |
| `#### 3 months` | non | — |
| `#### 3/5 kg` | non | — |
| `#### 1/0` | non | — |

`1/0` est détecté comme une fraction, puis rejeté car son dénominateur est nul.

### Fallback

Sans réponse terminale valide :

1. rechercher les nombres et fractions autonomes ;
2. prendre la dernière valeur ;
3. la normaliser.

| Sortie | Fallback |
| --- | ---: |
| `#### 3 months` | `3` |
| `#### 3/5 kg` | `0.6` |
| `#### 1/0` | aucune valeur |

Exemple :

```text
John pays 3.6 in a 30-day month.
```

Le fallback retourne `3.6`. `30` est ignoré car il appartient à `30-day`.

Le fallback ne comprend pas le sens du texte. Dans `40 liters in 2 buckets`,
il retourne `2`.

Le résultat du parsing indique sa provenance :

- `final_marker` : marqueur terminal valide ;
- `fallback` : dernière valeur autonome ;
- `None` : aucune valeur exploitable.

## 2. Parsing numérique

Formats reconnus :

- entiers : `42`, `-7` ;
- milliers : `1,000`, `12,500.25` ;
- décimaux : `3.5`, `.5`, `-.5` ;
- fractions : `3/5`, `-6/8`, `1.5/.5`.

Formats refusés :

- notation scientifique : `1e3` ;
- valeur non finie ;
- dénominateur nul ;
- signe `+` devant une réponse finale.

Le parsing utilise `Decimal`, puis `Fraction`. Aucun calcul n’utilise un
`float` binaire.

| Entrée | Normalisation |
| --- | ---: |
| `.5` | `0.5` |
| `12.000` | `12` |
| `3/5` | `0.6` |
| `-6/8` | `-0.75` |
| `1/3` | `0.3333333333` |

Une fraction finie produit sa valeur exacte. Une fraction périodique est
enregistrée sur 10 décimales.

## 3. Extraction des formules

Une formule doit utiliser ce format :

```text
<<expression=result>>
```

Toutes les annotations sont conservées dans leur ordre d’apparition.

Une annotation sans `>>` est également conservée, mais marquée
`unclosed_annotation`.

### Grammaire de l’expression

| Élément | Support |
| --- | --- |
| Opérateurs | `+`, `-`, `*`, `/`, `//` |
| Signes unaires | `+`, `-` |
| Groupement | parenthèses |
| Valeurs | entiers et décimaux |
| `//` | division entière par plancher |

Exemples valides :

```text
<<6-2=4>>
<<1+1+2+4=8>>
<<-2*.25=-.5>>
<<30*(1/3)=10>>
<<8//2=4>>
```

Après le dernier `=`, le résultat annoncé doit être un entier ou un décimal :

```text
<<1/3=0.3>>   valide
<<1/3=1/3>>   invalide
```

Une fraction est donc autorisée dans l’expression, pas dans le résultat
annoncé.

Le prompt demande `+`, `-`, `*`, `/`. Le parseur accepte aussi `//`. Une
évolution du parseur ne modifie donc pas automatiquement le prompt.

## 4. Exécution sécurisée

Le code n’utilise pas `eval`.

```text
expression
  → arbre AST
  → validation des nœuds
  → exécution récursive
  → résultat Fraction
```

| Formule refusée | Motif |
| --- | --- |
| `<<19=19>>` | aucune opération |
| `<<2+2>>` | `=` absent |
| `<<2+2=4=4>>` | expression invalide |
| `<<5%2=1>>` | modulo interdit |
| `<<2**3=8>>` | puissance interdite |
| `<<x+1=2>>` | variable interdite |
| `<<1/0=0>>` | division par zéro |

Limites de sécurité :

- 256 caractères par expression ;
- 64 nœuds AST ;
- 50 chiffres par littéral ;
- 256 chiffres par numérateur ou dénominateur intermédiaire.

## 5. Résultat produit pour une formule

Chaque annotation produit notamment :

| Champ | Signification |
| --- | --- |
| `expression` | partie située avant le dernier `=` |
| `claimed_result` | résultat annoncé et normalisé |
| `evaluated_result` | résultat réellement calculé |
| `parse_success` | syntaxe et AST valides |
| `execution_success` | exécution terminée |
| `arithmetic_correct` | résultat annoncé acceptable |
| `error` | erreur détectée |
| `is_correct` | parsing, exécution et résultat tous valides |

Erreurs possibles :

`missing_equals`, `empty_expression`, `invalid_claimed_result`,
`invalid_expression`, `execution_error`, `incorrect_result`,
`unclosed_annotation`.

Fichiers principaux :

- `src/evaluation/generation.py` : extraction finale ;
- `src/evaluation/numeric.py` : nombres et fractions ;
- `src/evaluation/arithmetic.py` : annotations, AST et exécution.
