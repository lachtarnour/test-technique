# Fine-tuning de Qwen2.5 sur GSM8K

Ce dépôt contient une expérience de fine-tuning de
[`[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
avec LoRA sur le dataset
[`[openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)`](https://huggingface.co/datasets/openai/gsm8k).

L’objectif est de comparer le modèle d’origine au modèle adapté, en évaluant
séparément :

* la justesse de la réponse numérique finale ;
* la validité des calculs intermédiaires générés par le modèle.

## Méthodologie

Le pipeline comprend :

* un découpage train/validation/test déterministe et sans recouvrement ;
* l’enregistrement de la version du dataset, de la seed et des empreintes
  SHA-256 ;
* un entraînement LoRA en completion-only ;
* une loss normalisée par le nombre exact de tokens cibles ;
* la sélection du meilleur checkpoint à partir de `eval_loss` ;
* une évaluation périodique sur deux sous-ensembles fixes de 300 exemples ;
* un parseur arithmétique basé sur l’AST Python, sans appel à `eval` ;
* un suivi optionnel des expériences avec Weights & Biases.

Le dépôt contient également quatre smoke tests portant sur le forward du
modèle, le tokenizer, le parsing arithmétique et la construction des prompts.

## Configuration de référence

| Paramètre                 | Valeur                                               |
| ------------------------- | ---------------------------------------------------- |
| Modèle                    | `Qwen/Qwen2.5-1.5B-Instruct`                         |
| Dataset                   | `openai/gsm8k`, configuration `main`                 |
| Train / validation / test | 6 352 / 1 121 / 1 319                                |
| Seed                      | 42                                                   |
| Objectif                  | Causal language modeling, completion-only            |
| LoRA                      | `r=8`, alpha `16`, dropout `0.05`, couches linéaires |
| Nombre d’epochs           | 3                                                    |
| Batch d’entraînement      | 24                                                   |
| Accumulation de gradients | 2                                                    |
| Batch de validation       | 16                                                   |
| Learning rate             | `2e-4`                                               |
| Scheduler                 | Cosine                                               |
| Warmup                    | 3 %                                                  |
| Longueur maximale         | 1 024 tokens                                         |
| Décodage                  | Greedy                                               |
| Nouveaux tokens maximum   | 768                                                  |
| Fréquence d’évaluation    | Tous les 2 epochs                                    |

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
  --num-train-epochs 3 \
  --train-batch-size 24 \
  --gradient-accumulation-steps 2 \
  --eval-batch-size 16 \
  --periodic-eval-batch-size 300 \
  --generation-batch-size 300 \
  --eval-every 2 \
  --wandb-project qwen-gsm8k \
  --require-cuda
```

Cette configuration correspond au run de référence effectué sur une V100S de
32 Go. Les tailles de batch peuvent être ajustées selon la mémoire disponible.

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

Le fonctionnement du parseur et la définition des métriques sont détaillés
dans :

* [`[documentation/evaluation/parsing.md](https://chatgpt.com/c/documentation/evaluation/parsing.md)`](documentation/evaluation/parsing.md) ;
* [`[documentation/evaluation/metrics.md](https://chatgpt.com/c/documentation/evaluation/metrics.md)`](documentation/evaluation/metrics.md).

## Vérification du code

```bash
uv run ruff check .
uv run pytest
```

## Organisation du dépôt

```text
configs/        configuration des expériences
documentation/  documentation du parsing et des métriques
script/         scripts d’entraînement, d’évaluation et de préparation
src/data/       formatage, tokenisation et collateur
src/model/      chargement du modèle et configuration LoRA
src/training/   loss, Trainer et planning d’évaluation
src/evaluation/ génération, parsing et calcul des métriques
tests/          smoke tests
```
