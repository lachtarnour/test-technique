# Qwen2.5-1.5B-Instruct sur GSM8K

Ce projet mesure une baseline sur GSM8K, puis fine-tune le même modèle avec
LoRA et réévalue ses réponses numériques.

## Installation

Python 3.10+ et un GPU CUDA sont recommandés.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requierement.txt
```

## 1. Baseline

La commande suivante évalue `Qwen/Qwen2.5-1.5B-Instruct` sur 100 exemples
déterministes du test set :

```bash
python3 script/evaluation.py
```

Les résultats complets sont écrits dans
`outputs/baseline_results.json`. Le fichier contient l'exact match, le taux de
réponses numériques valides, le taux de respect du marqueur `####` et chaque
prédiction individuelle.

## 2. Fine-tuning LoRA

```bash
python3 script/train.py
```

La configuration par défaut utilise :

- tout le train set GSM8K, avec 10 % réservés à la validation ;
- une époque ;
- LoRA sur les couches linéaires de Qwen ;
- une cross-entropy uniquement sur la completion ;
- une longueur maximale de 1024 tokens ;
- un batch effectif de 16 (`2 × 8` par GPU) ;
- une évaluation finale sur les mêmes 100 exemples test que la baseline.

L'adapter, le tokenizer et le rapport final sont sauvegardés dans
`outputs/qwen2.5-1.5b-gsm8k/`.

Pour un essai plus court :

```bash
python3 script/train.py \
  --train-subset-size 2000 \
  --num-train-epochs 1
```

En cas de mémoire GPU insuffisante :

```bash
python3 script/train.py \
  --train-batch-size 1 \
  --gradient-accumulation-steps 16
```

## Métrique

L'exact match compare la dernière réponse numérique générée à la référence
GSM8K normalisée. Le marqueur `####` est utilisé en priorité. S'il est absent,
le dernier nombre généré est utilisé, et la non-conformité au format est
comptabilisée séparément.

## Baseline Docker sur une instance OVH GPU

Prérequis sur l'instance :

- architecture Linux x86-64 ;
- pilote NVIDIA fonctionnel (`nvidia-smi`) ;
- Docker avec NVIDIA Container Toolkit ;
- au moins 15 Go d'espace disque libre pour l'image et le cache.

Le script construit l'image, vérifie le GPU, télécharge automatiquement Qwen
et GSM8K, sélectionne aléatoirement 1000 exemples du test set avec le seed 42,
puis lance l'évaluation :

```bash
chmod +x script/run_ovh_baseline.sh
./script/run_ovh_baseline.sh
```

Le résultat est écrit sur l'hôte dans :

```text
outputs/baseline_results_1000.json
```

Le cache Hugging Face est conservé dans `.cache/huggingface`, ce qui évite de
télécharger à nouveau le modèle lors d'un second lancement.

Les paramètres peuvent être adaptés avec des variables d'environnement :

```bash
BATCH_SIZE=4 SEED=123 ./script/run_ovh_baseline.sh
```

Valeurs conseillées selon la VRAM :

- 8-12 Go : `BATCH_SIZE=2` ;
- 16 Go : `BATCH_SIZE=4` ;
- 24 Go ou plus : `BATCH_SIZE=8`.

Pour reprendre le run déterministe par défaut, conserver `SEED=42`.
