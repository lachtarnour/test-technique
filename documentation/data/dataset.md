# Dataset et features d’entraînement

Cette partie transforme chaque paire `question` / `answer` en exemple causal prêt pour l’entraînement.

Le graphe, le programme postfixé et leurs classes sont décrits dans [graph.md](graph.md).

## Vue d’ensemble

```text
question + answer brut
        │
        ├── canonicalisation des annotations <<expression=result>>
        ├── création prompt / completion
        ├── application du chat template
        ├── labels limités à la completion
        ├── features optionnelles selon l’ablation
        └── padding dynamique par batch
```

## Contrat d’entrée

| Champ | Rôle | Contrainte |
|---|---|---|
| `question` | Problème fourni au modèle | Texte non vide |
| `answer` | Raisonnement et réponse cible | Texte contenant les annotations disponibles |

- Une annotation intermédiaire suit la forme `<<expression=result>>`.
- La réponse finale `#### valeur` reste dans la completion.
- Le split `test` n’est pas préparé par le pipeline d’entraînement.

## Canonicalisation de la réponse

### Objectif

- Conserver une seule représentation supervisée de chaque calcul.
- Éviter qu’une expression visible juste avant `<<...>>` révèle directement le contenu de l’annotation.
- Éviter qu’un résultat répété juste après `<<...>>` duplique la cible.
- Préserver le texte naturel lorsque la suppression n’est pas sûre.

### Transformations

| Cas détecté | Transformation |
|---|---|
| `<<valeur=valeur>>` | Annotation triviale supprimée |
| Expression identique avant l’annotation | Expression externe supprimée |
| Résultat identique après l’annotation | Résultat externe supprimé |
| Contexte ambigu ou suppression risquée | Texte conservé |

Exemple :

```text
Avant : 13 + 8 + 18 + 12 = <<13+8+18+12=51>>51 shells
Après : <<13+8+18+12=51>> shells
```

Hors identité triviale, la canonicalisation conserve l’annotation : `expression` et `result` restent les cibles structurées utilisées par les features mathématiques.

## Format conversationnel

| Objet | Messages | Supervision |
|---|---|---|
| `prompt` | `system` + `user` | Masqué dans les labels |
| `completion` | `assistant` | Apprise par cross-entropy |

```text
prompt_text = chat_template(system, user, add_generation_prompt=True)
full_text   = chat_template(system, user, assistant)
```

Invariants vérifiés :

- `full_text` commence exactement par `prompt_text`.
- Les tokens du prompt sont un préfixe exact des tokens complets.
- Une completion entièrement tronquée provoque une erreur.

## Tokenisation completion-only

| Colonne | Contenu |
|---|---|
| `input_ids` | Prompt et completion tokenisés |
| `attention_mask` | Tokens réels à prendre en compte |
| `labels` | `-100` sur le prompt, token cible sur la completion |

```text
labels = [-100, ..., -100] + completion_token_ids
```

- Le modèle voit le prompt comme contexte causal.
- La cross-entropy n’est calculée que sur la réponse assistant.
- La réponse finale après `####` est apprise par cette même cross-entropy.
- Il n’existe pas de `final_answer_loss` séparée dans le pipeline actuel.

## Features activées par l’ablation

Les colonnes sont demandées dynamiquement par l’objectif choisi.

| Ablation | Colonnes supplémentaires |
|---|---|
| A1 | Aucune |
| A2 | `token_loss_weights` |
| A3 | `step_positions`, `step_targets`, `step_target_scales`, `step_mask` |
| A4–A7 | Colonnes A3 + `postfix_program` |

- Les colonnes inutiles ne sont ni calculées ni stockées.
- Le format de base reste identique pour toutes les ablations.
- A8 réutilise la meilleure variante entraînée ; il ne définit pas un nouveau format de dataset.

## Poids token-level — A2

`token_loss_weights` a la même longueur que `input_ids`.

| Région | Poids |
|---|---:|
| Prompt ou label `-100` | `0.0` |
| Token de completion hors annotation mathématique | `1.0` |
| Token chevauchant `<<expression=result>>` | Poids mathématique configuré |

- Le chevauchement est calculé à partir des offsets caractères du tokenizer.
- Les offsets sont demandés uniquement lorsque des features les utilisent.
- Le poids modifie la contribution à la CE ; il ne crée pas une cible supplémentaire.

## Features step-level — A3 et suivantes

Une entrée est créée pour chaque étape mathématique détectée.

| Colonne | Définition |
|---|---|
| `step_positions` | Dernière position causale disponible avant le début de l’étape |
| `step_targets` | Résultat exact de l’expression, exporté en flottant fini |
| `step_target_scales` | `max(abs(target), 1)` |
| `step_mask` | Indique si l’étape est exploitable par la loss |
| `postfix_program` | Programme structuré sérialisé en JSON |

Une étape est masquée si au moins une condition échoue :

- parsing de l’annotation invalide ;
- programme postfixé invalide ;
- cible non finie ;
- aucune position causale disponible ;
- annotation partiellement tronquée.

Le masque conserve l’alignement des étapes : une étape invalide n’est pas supprimée de la séquence.

## Position causale

La supervision d’une étape est placée avant le texte propre à cette étape.

```text
étapes précédentes | position supervisée | texte de l’étape + <<expression=result>>
```

- Le début du contexte est le premier caractère non blanc depuis le début de la ligne ou la fin de l’annotation précédente.
- La position retenue est le dernier token strictement antérieur à ce contexte.
- Le head auxiliaire ne reçoit donc ni l’explication ni le résultat de l’étape cible.
- La visibilité complète de l’annotation reste exigée pour valider sa cible.

## Collation dynamique

Le padding est calculé séparément pour chaque batch.

| Colonne | Valeur de padding |
|---|---:|
| `input_ids` | `pad_token_id` |
| `attention_mask` | `0` |
| `labels` | `-100` |
| `token_loss_weights` | `0.0` |
| `step_positions` | `0` |
| `step_targets` | `0.0` |
| `step_target_scales` | `1.0` |
| `step_mask` | `false` |

- Tous les exemples d’un batch doivent exposer le même ensemble de features optionnelles.
- Les dimensions token et step sont paddées indépendamment.
- `postfix_program` est désérialisé depuis le JSON avant son utilisation par le trainer.

## Fichiers concernés

| Fichier | Responsabilité |
|---|---|
| `src/data/language/formatting.py` | Canonicalisation et messages conversationnels |
| `src/data/language/tokenization.py` | Chat template, labels et offsets |
| `src/data/language/dataset.py` | Préparation des splits d’entraînement |
| `src/data/language/collator.py` | Padding dynamique |
| `src/data/features.py` | Features token-level, step-level et postfixées |
| `src/training/objective.py` | Sélection dynamique des colonnes par ablation |
