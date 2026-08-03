# Fine-tuning de Qwen2.5 sur GSM8K

Ce dépôt contient une expérience de fine-tuning de
[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
avec LoRA sur le dataset
[openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k).

L’objectif est de comparer le modèle d’origine au modèle adapté, en évaluant
séparément :

* la justesse de la réponse numérique finale ;
* la validité des calculs intermédiaires générés par le modèle.

## Protocole de comparaison

Afin de garantir une comparaison équitable et reproductible, la méthode
d’évaluation a été définie et figée avant le lancement des expériences. Le même
protocole sera ainsi appliqué aux différentes approches étudiées :

* une approche classique, entraînée avec une fonction de perte d’entropie
  croisée ;
* une variante intégrant l’objectif **Math-Consistent**, afin de renforcer la
  cohérence des calculs intermédiaires.

Le fonctionnement du parseur et la définition des métriques sont détaillés
dans :

* [documentation/evaluation/parsing.md](https://github.com/lachtarnour/test-technique/blob/master/documentation/evaluation/parsing.md) ;
* [documentation/evaluation/metrics.md](https://github.com/lachtarnour/test-technique/blob/master/documentation/evaluation/metrics.md).

## Méthodologie

Le pipeline comprend :

* un découpage train/validation/test déterministe et sans recouvrement ;
* l’enregistrement de la version du dataset, de la seed et des empreintes
  SHA-256 ;
* un entraînement LoRA en completion-only ;
* une représentation canonique où chaque calcul apparaît une seule fois,
  uniquement dans son annotation `<<expression=result>>` ;
* une loss normalisée par le nombre exact de tokens cibles ;
* la sélection du meilleur checkpoint à partir de `eval_loss` ;
* une évaluation périodique sur deux sous-ensembles fixes de 300 exemples ;
* un parseur arithmétique basé sur l’AST Python, sans appel à `eval` ;
* un suivi optionnel des expériences avec Weights & Biases.

La suite de tests couvre le modèle, le tokenizer, les données, l’entraînement,
l’évaluation, le graphe arithmétique et le compilateur postfixé.

## Configuration de référence

| Paramètre                 | Valeur                                               |
| ------------------------- | ---------------------------------------------------- |
| Modèle                    | `Qwen/Qwen2.5-1.5B-Instruct`                         |
| Dataset                   | `openai/gsm8k`, configuration `main`                 |
| Train / validation / test | 6 352 / 1 121 / 1 319                                |
| Seed                      | 42                                                   |
| Objectif                  | Causal language modeling, completion-only            |
| LoRA                      | `r=8`, alpha `16`, dropout `0.05`, couches linéaires |
| Nombre maximal d’epochs   | 30                                                   |
| Batch d’entraînement      | 24                                                   |
| Accumulation de gradients | 3                                                    |
| Dernier batch train incomplet | Ignoré                                            |
| Batch de validation       | 32                                                   |
| Batch d’évaluation périodique | 300                                               |
| Batch de génération       | 300                                                  |
| Learning rate initial     | `1e-4`                                               |
| Scheduler                 | ReduceLROnPlateau sur `eval_loss`                    |
| Réduction du LR           | facteur `0,5`, après 4 validations, minimum `1e-5`   |
| Warmup                    | Aucun                                                |
| Early stopping            | patience `6`, seuil absolu `0,001`                   |
| Longueur maximale         | 1 024 tokens                                         |
| Décodage                  | Greedy                                               |
| Nouveaux tokens maximum   | 768                                                  |
| Validation `eval_loss`    | Chaque epoch                                         |
| Évaluation générative     | Tous les 2 epochs                                    |

Le split de test officiel n’est pas utilisé pendant le développement. Il est
réservé à l’évaluation finale du modèle préentraîné et du checkpoint LoRA
sélectionné.

## Installation

Le projet nécessite Python 3.10 à 3.13. Un GPU CUDA est recommandé pour
l’entraînement.

```bash
python3 -m pip install uv
uv sync --frozen
```

Le suivi avec Weights & Biases est facultatif. Pour l’activer :

```bash
cp .env.example .env
# Ajouter WANDB_KEY dans le fichier .env
```

Pour désactiver W&B sur une commande :

```bash
--wandb-mode disabled
```

## Préparation du dataset

Le split reproductible doit être créé une seule fois :

```bash
uv run python script/create_data_split.py
```

Les données générées et leur manifeste sont enregistrés dans :

```text
data/gsm8k_train_validation15_test_seed42/
```

Ce dossier n’est pas suivi par Git.

## Entraînement

```bash
uv run python script/train.py \
  --ablation A1 \
  --num-train-epochs 30 \
  --eval-every 2 \
  --wandb-project qwen-gsm8k \
  --require-cuda
```

Les tailles de batch par défaut correspondent au run de référence effectué sur
une L40S de 46 Go. Le batch effectif d'entraînement est `24 × 3 = 72`.

### Architecture des ablations

Les expériences d'entraînement A1 à A7 utilisent la même chaîne de données, le
même modèle et le même `Trainer`. L'option `--ablation` sélectionne une recette
dans les deux tables de `src/training/objective.py` : `ABLATIONS` décrit les
losses actives et `LOSSES` en déduit dynamiquement les features à produire, les
têtes à créer et les fonctions à exécuter.

| Ablation | Objectif |
| -------- | -------- |
| A0 | Modèle préentraîné, évalué sans entraînement |
| A1 | Cross-entropy standard |
| A2 | CE pondérée sur les tokens mathématiques |
| A3 | A2 + résultat normalisé |
| A4 | A3 + exécution normalisée |
| A5 | A4 + opérateur + dépendance non ordonnée |
| A6 | A4 + action structurée ordonnée |
| A7 | A6 + composition |
| A8 | Meilleure variante + best-of-N + vérificateur à l'inférence |

A0 et A8 ne sont donc pas des losses artificielles : A0 appartient au protocole
d'évaluation et A8 à la génération/vérification. Les besoins de données et de
modèle de A1 à A7 sont déjà déclarés dans ces tables. A1 et A2 possèdent une
implémentation exécutable ; les losses structurées de A3 à A7 restent à réaliser.

Par défaut, le modèle, le tokenizer et le rapport d’entraînement sont
enregistrés dans :

```text
outputs/qwen2.5-1.5b-gsm8k-a1-control/
```

## Évaluation

### Modèle préentraîné

```bash
uv run python script/evaluation.py \
  --experiment-name a0-pretrained-test \
  --output-file outputs/a0_test.json
```

### Modèle adapté avec LoRA

```bash
uv run python script/evaluation.py \
  --checkpoint-path outputs/qwen2.5-1.5b-gsm8k-a1-control \
  --experiment-name a1-lora-test \
  --output-file outputs/a1_test.json
```

Pour vérifier rapidement que le pipeline fonctionne, il est possible de limiter
l’évaluation à quelques exemples :

```bash
uv run python script/evaluation.py \
  --subset-size 20 \
  --experiment-name smoke-test \
  --output-file outputs/smoke_test.json
```

Les rapports JSON contiennent les prédictions individuelles ainsi que les
métriques suivantes :

* accuracy stricte, basée sur le format `#### nombre` ;
* accuracy numérique avec méthode de fallback ;
* erreur numérique relative ;
* exactitude des annotations `<<expression=result>>` ;
* cohérence entre le dernier calcul généré et la réponse finale.

## Résultats des ablations

Chaque nouvelle ablation est ajoutée à cette table avec son meilleur checkpoint,
sans changer le split, le prompt ni le protocole d'évaluation.

| Ablation | Modèle / checkpoint | Epoch | `eval_loss` locale | `eval_loss` CUDA | Accuracy finale | Accuracy stricte | Cohérence interne | Correcte et cohérente |
| -------- | ------------------- | ----: | ------------------: | ---------------: | ---------------: | ----------------: | -----------------: | --------------------: |
| A0 | Qwen2.5-1.5B-Instruct préentraîné | 0 | 0,9131 | — | 66,67 % | 15,00 % | 1,67 % | 1,33 % |
| A1 | CE, `checkpoint-176` | 2 | **0,5350** | **0,4324** | **70,33 %** | **70,33 %** | **80,67 %** | **63,00 %** |

Conditions de comparaison :

* `eval_loss` locale : validation complète de 1 121 exemples et 93 854 tokens
  cibles, calculée en BF16 sur MPS pour A0 et A1 ;
* `eval_loss` CUDA : validation complète calculée par le `Trainer` sur L40S ;
  la valeur A0 n'a pas été enregistrée sur CUDA ;
* métriques génératives : même sous-ensemble fixe de 300 exemples de validation,
  décodage greedy et 768 nouveaux tokens maximum ;
* données : seed `42`, hash validation `9f75006a…`, hash du sous-ensemble
  `dbe73712…` ;
* prompt : version `2.4.0`, hash `1aae71f8…` ;
* run A1 : [Weights & Biases](https://wandb.ai/nourlachtar/qwen-gsm8k/runs/tbevvvna).

La comparaison locale homogène donne une baisse de `eval_loss` de **41,4 %**
entre A0 et le meilleur checkpoint A1. Les valeurs MPS et CUDA ne doivent pas
être comparées directement entre elles ; les ablations futures seront comparées
dans une même colonne et sur le même matériel.

## Vérification du code

```bash
uv run ruff check .
uv run pytest
```

## Organisation du dépôt

```text
documentation/        contrats du parsing, des métriques et du graphe
script/               points d’entrée exécutables
src/data/loading.py   chargement des splits GSM8K
src/data/splits.py    création déterministe des splits gelés
src/data/language/    formatage, tokenisation et batches du LLM
src/data/graph/       parsing, graphe, exécution, programme postfixé et audit
src/model/            modèle, tokenizer, dtype et têtes auxiliaires
src/training/         configuration, objectifs, callbacks et Trainer
src/evaluation/       réponses, génération et métriques
tests/                tests unitaires et d’intégration
```
