# Graphe mathématique et programme postfixé

Cette partie convertit les annotations `<<expression=result>>` en représentations structurées exécutables.

Elle sépare la syntaxe du calcul, la provenance des nombres et le programme utilisé par les objectifs structurés.

- Toutes les structures sont des dataclasses immuables (`frozen=True`).
- Toute valeur exacte est portée par `Fraction` puis sérialisée en `{numerator, denominator}`.

## Vue d’ensemble

```text
question ───────────────► nombres du problème + positions source
                              │
answer ─► MathStep ─► arbre syntaxique
                              │ résolution de provenance
                              ▼
                       CalculationGraph
                              │ compilation postfixée
                              ▼
                         PostfixProgram
                              │
                    exécution + validations
```

## Deux niveaux de représentation

| Niveau | Nœuds | Rôle |
|---|---|---|
| Syntaxique | `NumberNode`, `OperationNode` | Représenter exactement l’expression annotée |
| Provenance | `ProblemNumberNode`, `ReferenceNode`, `LiteralNode`, `UnresolvedNode` | Identifier l’origine de chaque opérande |

- Le parsing construit d’abord l’arbre syntaxique.
- Le builder remplace ensuite chaque `NumberNode` par un nœud de provenance.
- Une provenance ambiguë reste explicite ; elle n’est jamais choisie arbitrairement.

## Opérateurs arithmétiques

### `ArithmeticOperator`

| Valeur | Symbole | Arité |
|---|---:|---:|
| `ADD` | `+` | 2 |
| `SUBTRACT` | `-` | 2 |
| `MULTIPLY` | `*` | 2 |
| `DIVIDE` | `/` | 2 |
| `FLOOR_DIVIDE` | `//` | 2 |
| `POSITIVE` | `+x` | 1 |
| `NEGATE` | `-x` | 1 |

- L’arité est validée dès la création d’un `OperationNode`.
- `FLOOR_DIVIDE` est regroupé avec `DIVIDE` pour la supervision d’opérateur.
- L’évaluation utilise des fractions exactes, pas des flottants.

## Positions source

### `SourceSpan`

| Champ | Définition |
|---|---|
| `start` | Index du premier caractère inclus |
| `end` | Index du premier caractère exclu |

Invariant : `0 <= start <= end`.

Les spans sont toujours relatifs au texte qui les porte : question, réponse ou expression annotée.

## Nœuds du graphe

| Classe | Champs principaux | Signification |
|---|---|---|
| `NumberNode` | `value` | Nombre brut issu du parsing syntaxique |
| `ProblemNumberNode` | `value`, `source_span`, `source_text` | Occurrence exacte dans la question |
| `ReferenceNode` | `step_index`, `value` | Résultat d’une étape antérieure |
| `LiteralNode` | `value` | Constante absente des sources connues |
| `UnresolvedNode` | `value`, `candidates` | Plusieurs provenances exactes possibles |
| `OperationNode` | `operator`, `operands` | Opération unaire ou binaire |

### Contraintes par nœud

- `ProblemNumberNode` conserve le texte source original et son span.
- `ReferenceNode.step_index` doit viser une étape strictement antérieure.
- `UnresolvedNode` contient au moins deux candidats de même valeur.
- Les candidats non résolus sont uniquement des nombres du problème ou des références antérieures.
- `OperationNode.operands` doit respecter l’arité de l’opérateur.

## Étape syntaxique — `MathStep`

Une annotation produit une entrée `MathStep`, valide ou invalide.

| Champ ou propriété | Définition |
|---|---|
| `index` | Position stable de l’annotation |
| `raw_annotation` | Texte complet `<<...>>` |
| `annotation_span` | Position de l’annotation dans la réponse |
| `expression` | Partie avant le dernier `=` |
| `expression_span` | Position de l’expression dans la réponse |
| `claimed_result_text` | Texte déclaré après le dernier `=` |
| `claimed_result_span` | Position de ce texte dans la réponse |
| `claimed_result` | Valeur exacte extraite du texte déclaré |
| `target_result` | Résultat exact calculé depuis l’expression |
| `expression_tree` | Arbre syntaxique |
| `operator` | Opérateur racine normalisé, ou `None` pour une feuille |
| `valid` | Parsing et évaluation réussis |
| `error` | Motif d’invalidité |
| `is_final` | Étape correspondant au marqueur terminal `####` |

Une étape valide impose :

- une expression non vide ;
- un arbre syntaxique disponible ;
- une cible calculable ;
- un résultat déclaré égal au résultat calculé ;
- aucune erreur de parsing.

`claimed_result` et `target_result` restent séparés : le premier vient du texte, le second de l’exécution exacte.

## Extraction et parsing

### Expressions acceptées

- nombres entiers, décimaux et fractions ;
- parenthèses ;
- opérateurs `+`, `-`, `*`, `/`, `//` ;
- signes unaires positif et négatif.

### Sécurité

- L’expression est analysée par AST autorisé.
- Aucun nom, appel de fonction ou attribut n’est exécuté.
- Les calculs sont convertis en `Fraction`.
- Une division par zéro invalide l’étape.

### Étape finale

`is_final=true` uniquement si :

- un marqueur terminal strict `#### valeur` existe ;
- la dernière annotation est valide ;
- sa cible exacte correspond à la réponse finale.

## Nombres du problème

Le builder extrait les valeurs et leurs occurrences exactes dans la question.

Formes prises en charge :

- chiffres, décimaux et fractions ;
- cardinaux anglais ;
- fractions textuelles et nombres mixtes ;
- formes lexicales comme `half`, `double`, `triple`, `dozen`, `couple` ou `once`.

Chaque occurrence produit un `ProblemNumberNode` distinct, même lorsque plusieurs occurrences ont la même valeur.

## Résolution de provenance

Pour chaque `NumberNode`, les candidats sont recherchés par égalité exacte parmi :

- les occurrences de même valeur dans la question ;
- les résultats des étapes antérieures valides.

| Nombre de candidats | Nœud produit |
|---:|---|
| `0` | `LiteralNode` |
| `1` | `ProblemNumberNode` ou `ReferenceNode` |
| `>= 2` | `UnresolvedNode` |

Exemple d’ambiguïté :

```text
Question : Alice possède 5 billes et Bob possède 5 billes.
Expression : 5 + 2
```

Le premier opérande possède deux occurrences compatibles : il devient `UnresolvedNode` avec les deux candidats.

## Étape de graphe — `GraphStep`

| Champ | Définition |
|---|---|
| `index` | Position stable de l’étape |
| `expression` | Expression originale |
| `target_result` | Résultat exact attendu |
| `expression_tree` | Arbre enrichi par la provenance |
| `dependencies` | Indices des étapes antérieures référencées |
| `unresolved_operand_count` | Nombre d’opérandes ambigus |
| `valid` | Étape syntaxiquement exploitable |
| `error` | Motif d’invalidité conservé |
| `is_final` | Correspondance avec la réponse terminale |

Invariants :

- les dépendances sont triées et uniques ;
- chaque dépendance est strictement antérieure à l’étape ;
- une étape invalide reste présente pour conserver l’alignement.

## Graphe complet — `CalculationGraph`

| Champ | Contenu |
|---|---|
| `problem_numbers` | Toutes les occurrences numériques de la question |
| `steps` | Toutes les étapes, valides ou invalides |

Propriétés dérivées :

- `unresolved_operand_count` : somme des ambiguïtés du graphe ;
- `provenance_complete` : vrai si le graphe contient des étapes, si elles sont toutes valides et si aucune provenance n’est ambiguë.

## Exécution du graphe

L’exécuteur parcourt les étapes dans l’ordre et réutilise les résultats antérieurs.

| Nœud | Valeur exécutée |
|---|---|
| `ProblemNumberNode` | Valeur exacte extraite de la question |
| `ReferenceNode` | Résultat exact de l’étape référencée |
| `LiteralNode` | Constante exacte |
| `UnresolvedNode` | Valeur exacte commune à tous les candidats |
| `OperationNode` | Résultat de l’opération sur ses opérandes |

### `GraphStepEvaluation`

| Champ | Définition |
|---|---|
| `index` | Étape évaluée |
| `result` | Résultat calculé ou `None` |
| `target_result` | Cible du graphe |
| `matches_target` | Égalité exacte résultat / cible |
| `error` | Erreur d’exécution éventuelle |

Une ambiguïté de provenance peut rester numériquement exécutable : tous ses candidats représentent la même valeur. Elle reste néanmoins masquée pour les losses qui exigent une provenance certaine.

## Programme postfixé

Le programme postfixé traduit l’arbre en actions ordonnées, déterministes et directement exécutables.

### Références d’opérandes

| Classe | Référence |
|---|---|
| `ProblemNumberReference` | Index dans la table des nombres du problème |
| `PreviousResultReference` | Résultat d’une étape antérieure |
| `LocalResultReference` | Résultat d’une action antérieure de la même étape |
| `LiteralReference` | Constante exacte |
| `UnresolvedReference` | Ensemble de provenances candidates |

### `PostfixAction`

| Champ ou propriété | Définition |
|---|---|
| `index` | Index local de l’action |
| `operator` | Opérateur postfixé |
| `operands` | Références consommées par l’action |
| `operand_mask` | `false` uniquement pour une référence non résolue |

- Les actions sont produites en parcours post-ordre, de gauche à droite.
- Une `LocalResultReference` ne peut viser qu’une action antérieure.
- Une feuille sans opération est convertie en action `COPY`.
- Les opérateurs postfixés reprennent les opérateurs arithmétiques et ajoutent `COPY`.

### Exemple de compilation

```text
Expression : (3 + 4) * 2
problem_numbers : [3, 4, 2]

action 0 : ADD(problem[0], problem[1])
action 1 : MULTIPLY(local[0], problem[2])
```

Le résultat de `action 0` devient un opérande local de `action 1`.

### `PostfixProgramStep`

| Champ ou propriété | Définition |
|---|---|
| `index` | Index identique à celui du `GraphStep` |
| `expression` | Expression originale |
| `target_result` | Cible exacte |
| `target_scale` | `max(abs(target_result), 1)` |
| `actions` | Actions postfixées contiguës |
| `operand_mask` | Masques ordonnés par action et par opérande |
| `operand_count` | Nombre total d’opérandes |
| `masked_operand_count` | Nombre d’opérandes ambigus |
| `valid` | Programme compilable |
| `error` | Erreur conservée pour une étape invalide |
| `is_final` | Marqueur d’étape finale |

Contraintes :

- les indices d’actions commencent à `0` et sont contigus ;
- une étape valide possède une cible et au moins une action ;
- une étape invalide ne contient aucune action et conserve son erreur ;
- les références inter-étapes pointent uniquement vers le passé.

### `PostfixProgram`

| Champ ou propriété | Contenu |
|---|---|
| `problem_numbers` | Table stable des nombres de la question |
| `steps` | Programmes d’étapes alignés avec le graphe |
| `action_count` | Nombre total d’actions |
| `operand_count` | Nombre total d’opérandes |
| `masked_operand_count` | Nombre total d’opérandes ambigus |

Le dataset sérialise cette structure en JSON afin de conserver un schéma Arrow stable.

## Masques de supervision

| Niveau | Masque `true` | Masque `false` |
|---|---|---|
| Opérande | Provenance unique | `UnresolvedReference` |
| Étape | Parsing, cible, position causale et programme valides | Au moins une condition invalide |

- Un masque `false` retire l’élément de la loss concernée.
- Il ne retire pas l’élément de la représentation structurée.
- Les index restent donc stables entre parsing, graphe, programme et batch.

## Validation

Deux validations complémentaires sont disponibles.

| Validation | Compare | Détecte |
|---|---|---|
| Numérique | Résultat du graphe, résultat postfixé et cible | Erreur de calcul ou de référence |
| Structurelle | Graphe et programme compilé | Action, opérateur, ordre, dépendance ou masque incorrect |

La validation structurelle contrôle notamment :

- la table des nombres du problème ;
- le nombre et l’alignement des étapes ;
- les métadonnées et `target_scale` ;
- l’ordre, l’arité et la signature des actions ;
- les dépendances inter-étapes ;
- les masques et le nombre d’opérandes non résolus.

Une égalité numérique finale ne suffit pas : deux programmes qui donnent la même valeur peuvent avoir des structures différentes.

### Classes de rapport

| Classe | Contenu |
|---|---|
| `GraphEvaluation` | Résultats du graphe et indicateurs globaux d’exécution |
| `PostfixStepEvaluation` | Résultat postfixé d’une étape et comparaison à sa cible |
| `PostfixProgramEvaluation` | Résultats postfixés et indicateurs globaux |
| `PostfixVerificationStep` | Comparaison exacte graphe / programme pour une étape |
| `PostfixProgramVerification` | Équivalence numérique sur tout le programme |
| `PostfixValidationIssue` | Code, message et index d’étape optionnel |
| `PostfixStructuralValidation` | Nombre d’étapes contrôlées et liste des problèmes structurels |

## Limites

- La provenance repose sur l’égalité exacte des valeurs, pas sur une compréhension sémantique complète du texte.
- Les occurrences répétées de même valeur restent ambiguës sans indice supplémentaire.
- Une valeur absente de la question et des étapes précédentes devient un littéral.
- Les calculs sont exacts dans le graphe ; le flottant n’est utilisé qu’à l’export des cibles et échelles d’entraînement.
- Une étape non annotée ne produit pas automatiquement une cible structurée.

## Fichiers concernés

| Fichier | Responsabilité |
|---|---|
| `src/data/graph/schemas.py` | Schémas du graphe et invariants |
| `src/data/graph/parser.py` | Extraction des annotations et parsing sûr |
| `src/data/graph/builder.py` | Nombres du problème et résolution de provenance |
| `src/data/graph/execution.py` | Exécution exacte du graphe |
| `src/data/graph/postfix_schemas.py` | Schémas du programme postfixé |
| `src/data/graph/postfix.py` | Compilation, exécution et validations |
| `src/data/features.py` | Projection du graphe vers les features d’entraînement |
