# Fine-tuning de Qwen2.5 sur GSM8K

Ce projet compare
[`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
à une version adaptée par LoRA sur
[`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k).

L’évaluation mesure séparément la justesse de la réponse finale et la cohérence
du raisonnement arithmétique intermédiaire.

## Points clés

- split train/validation/test figé, déterministe et sans recouvrement ;
- révision du dataset, seed et empreintes SHA-256 enregistrés ;
- entraînement LoRA completion-only avec normalisation exacte par token cible ;
- validation et sélection du meilleur checkpoint par `eval_loss` ;
- évaluation périodique sur deux sous-ensembles fixes de 300 exemples ;
- parsing arithmétique sécurisé par AST, sans utilisation de `eval` ;
- suivi optionnel des hyperparamètres et métriques avec Weights & Biases ;
- quatre smoke tests publics couvrant le forward du modèle, le tokenizer,
  le parsing et le contrat question/prompt.

## Protocole

| Élément | Valeur |
| --- | --- |
| Modèle | `Qwen/Qwen2.5-1.5B-Instruct` |
| Dataset | `openai/gsm8k`, configuration `main` |
| Train / validation / test | 6 352 / 1 121 / 1 319 |
| Seed | 42 |
| Objectif | causal language modeling completion-only |
| LoRA | `r=8`, alpha `16`, dropout `0.05`, couches linéaires |
| Epochs | 3 |
| Batch train / accumulation | 24 / 2 |
| Batch validation | 16 |
| Learning rate | `2e-4`, scheduler cosine, warmup 3 % |
| Longueur maximale | 1 024 tokens |
| Génération | greedy, 768 nouveaux tokens maximum |
| Évaluation périodique | epochs 0, 2, 4… |

Le test officiel reste isolé pendant le développement. Il est utilisé
uniquement pour l’évaluation finale du modèle préentraîné ou du checkpoint
LoRA retenu.

## Installation

Prérequis : Python 3.10 à 3.13. Un GPU CUDA est recommandé pour l’entraînement.

```bash
python3 -m pip install uv
uv sync --frozen
```

Le suivi W&B est optionnel :

```bash
cp .env.example .env
# Renseigner WANDB_KEY dans .env
```

Utiliser `--wandb-mode disabled` pour exécuter une commande sans W&B.

## Préparation des données

Créer une fois le split reproductible :

```bash
uv run python script/create_data_split.py
```

Le dataset généré et son manifeste sont placés dans
`data/gsm8k_train_validation15_test_seed42/`. Ce dossier est ignoré par Git.

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

Les tailles de batch correspondent au run de référence sur une V100S 32 Go et
restent configurables selon le GPU. Le modèle, le tokenizer et le rapport JSON
sont enregistrés par défaut dans
`outputs/qwen2.5-1.5b-gsm8k-a1-control/`.

## Évaluation

Évaluer le modèle préentraîné sur le test officiel :

```bash
uv run python script/evaluation.py \
  --experiment-name a0-pretrained-test \
  --output-file outputs/a0_test.json
```

Évaluer l’adaptateur entraîné :

```bash
uv run python script/evaluation.py \
  --checkpoint-path outputs/qwen2.5-1.5b-gsm8k-a1-control \
  --experiment-name a1-lora-test \
  --output-file outputs/a1_test.json
```

Pour un smoke test, ajouter par exemple `--subset-size 20`. Les rapports
contiennent les prédictions ainsi que :

- l’accuracy stricte au format `#### nombre` ;
- l’accuracy numérique avec fallback ;
- l’erreur numérique relative ;
- l’exactitude des annotations `<<expression=result>>` ;
- la cohérence entre le dernier calcul et la réponse finale.

Le [parsing](documentation/evaluation/parsing.md) et les
[métriques](documentation/evaluation/metrics.md) sont documentés séparément.

## Qualité du code

```bash
uv run ruff check .
uv run pytest
```

## Structure

```text
configs/        configuration de l’expérience
documentation/  définition du parsing et des métriques
script/         points d’entrée reproductibles
src/data/       formatage, tokenisation et collateur
src/model/      construction du modèle LoRA
src/training/   objectif, Trainer et planning
src/evaluation/ génération, parsing et métriques
tests/          quatre smoke tests publics
```
