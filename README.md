# DUDA Clickscore v1

> Le DUDA Clickscore est le MVP d'une Webapp streamlit de Clickscoring réalisée pour le DU Sorbonne Data Analytics 2025-2026 par Alexandre Cameron BORGES.
Basé sur 2 modèles utilisant PyTorch, BERT, Huggingface avec un Fine-tuning multi-tâche (classification clickbaitness + régression linéaire CTR) sur plusieurs dataset d'interactions en ligne (MIND, Webis, Kaggle..)

Contexte: Les investissements publicitaires en ligne sont de plus en plus omniprésents pour les petites et grandes entreprises, cet outil vise à aider à la prise de décision des responsables marketing quant à quelles publicités privilégier afin d'économiser en budget A/B test.
L'idée est également de récupérer une part de la connaissance de l'efficacité publicitaire, connaissance qui est cloisonnée par les plateformes publicitaires comme Google ou Meta

Google Colab de préparation: https://colab.research.google.com/drive/1lrgvBJ_1BHrT732r1RnnV1ZBOvZKzZN5?usp=sharing 

Vidéo explicative: https://drive.google.com/drive/folders/1pvxEq-HsV99_zG1A3AIUmYXZUuTeT0SN?usp=drive_link 

<p align="center">
  <a href="https://acb-dudaclickscore.streamlit.app/" target="_blank"><img alt="Streamlit app" src="https://img.shields.io/badge/DEMO-online-success?logo=streamlit"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-blue?logo=python">
</p>

---

## · 1️⃣ ✨ Objectif

* **Classer** automatiquement des messages publicitaires en trois niveaux : `Nobait`, `Softbait`, `Clickbait`.
* **Prédire** le *Click‑Through Rate* (CTR) attendu d’un texte publicitaire.
* Fournir une **interface simple** permettant aux marketers d’importer un CSV de publicités (texte + image) et d’obtenir en quelques secondes des recommandations basées sur des modèles BERT fine‑tunés.

## · 2️⃣ 🚀 Démo rapide

1. Ouvrez la WebApp hébergée → [https://acb-dudaclickscore.streamlit.app/](https://acb-dudaclickscore.streamlit.app/)

2. Choisissez le **cible démographique** (âge, genre).

3. Importez un CSV comportant deux colonnes, avec jusqu'à 10 textes publicitaires à tester :

   | image              | texte                       |
   | ------------------ | --------------------------- |
   | path/to/banner.png | "Free shipping this week !" |

4. Cliquez sur **Prédire** et laissez la magie opérer !

## · 3️⃣ 📊 Jeux de données

| Domaine               | Source                         | Lien                                                                                                   |
| --------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| Clickbait (EN)        | Kaggle « Clickbait Headlines » | [amananandrai/clickbait-dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset)                                                                       |
| Ads clicks            | Kaggle « Ad Click Prediction » | [marius2303/ad-click-prediction-dataset](https://www.kaggle.com/datasets/marius2303/ad-click-prediction-dataset )`                                                               |
| Advertising clicks          | Kaggle « Advertising CSV »     | [souvik1618/advertising-dataset](https://www.kaggle.com/datasets/marius2303/ad-click-prediction-dataset)`                                                                       |
| News CTR              | Microsoft **MIND**             | [https://msnews.github.io/](https://msnews.github.io/)                                                 |
| Clickbait multi‑modal | **Webis Clickbait 2017**       | [https://webis.de/competitions/clickbait-2017.html](https://zenodo.org/records/5530410) |

Les jeux Kaggle & WEBIS ont été nettoyés, normalisés (colonnes: texte, age, genre, clickbait o/n) et fusionnés puis enrichis (imputation d’âge, de genre et de *truthMean=probabilité de clickbait*) afin d'entraîner le modèle de classification clickbait, le Dataset MIND a été transformé seul pour le modèle de régression linéaire CTR (calcul du CTR à partir du nombre de clics et nombres d'affichages d'un Id publicitaire correspondant à des titres publicitaires) dans le notebook [`data_and_finetuning.ipynb`](data_and_finetuning.ipynb).

## · 4️⃣ 🧠 Modèles

Base: BERT est un modèle de langage lancé fin 2018 : il repose sur un bloc Transformer qui “regarde” chaque phrase simultanément vers la gauche et vers la droite ; durant son pré-apprentissage, il apprend la grammaire et le sens en devinant des mots cachés et en testant si deux phrases se suivent ; une fois ce socle acquis, on ne remplace que la petite couche finale pour adapter BERT à presque n’importe quelle tâche (analyse de sentiments, FAQ, prédiction de clics…)

| Tâche           | Architecture                                                         | Dataset d’entraînement     | Métriques (val set)      |
| --------------- | -------------------------------------------------------------------- | -------------------------- | ------------------------ |
| **Clickbait ±** | BERT + features (âge, genre) <br> multi‑task `CB_F1 + truthMean Acc` | (62767 lignes * 5 colonnes) Fusion Kaggle & Webis      | F1 ≈ 0 .90 (sur 1 ; cela mesure à la fois les bons “oui” et les bons “non”) / Acc ≈ 0 .71 (71 % des titres bien classés)|
| **CTR %**       | BERT regression <br> (sigmoïde 0‑1: pour qu’il reste dans 0-100 %)                                 | (10189 lignes * 2 colonnes) MIND (≥ 100 impr.) | RMSE ≈ 0 .018 (en moyenne le modèle se trompe de 1,8 points sur 100 dans le pourcentage cliqué)           |

Les poids entraînés (< 5 epoch) sont stockés sur mon compte privé **Hugging Face** et chargés *à la volée* via l’API. Le dépôt Hugging Face: https://huggingface.co/alexandre-cameron-borges

## · 5️⃣ 🏗️ Architecture de l’application

```
app.py (Streamlit)
 ├─ models/
 │   └─ predict.py  ← lazy‑loading & inference
 ├─ requirements.txt
 └─ notebooks/ (préparation & fine‑tuning)
```

1. L’utilisateur charge son CSV.
2. `predict.py` télécharge (la première fois) puis met en cache les deux modèles à partir de Hugging Face.
3. Pour chaque texte :

   * Classification probabilité *clickbait* ;
   * CTR prévu.
4. Les résultats sont normalisés, triés et affichés sous forme de tableau + de deux graphiques (scatter & pie chart).

## · 6️⃣ ⚙️ Installation locale

```bash
# 1. Cloner le repo
$ git clone https://github.com/alexandre-cameron-borges/duda_clickscore.git
$ cd duda_clickscore

# 2. Créer l’environnement
$ python3 -m venv .venv && source .venv/bin/activate

# 3. Installer les dépendances
$ pip install -r requirements.txt

# 4. Définir votre token HF (modèles privés)
$ export HUGGINGFACE_TOKEN="<votre_token_personnel>"

# 5. Lancer la WebApp
$ streamlit run app.py
```

> **Note :** si la variable `HUGGINGFACE_TOKEN` est absente, l’application s’arrête avec le message d’erreur approprié. Ajoutez‑la dans **Secrets** lorsque vous déployez la WebApp sur *Streamlit Community Cloud*.

## · 7️⃣ 🗂️ Organisation du dépôt

```
├── app.py               # Script Streamlit principal
├── models/
│   └── predict.py       # Fonctions d’inférence + chargement des poids
├── requirements.txt     # Dépendances Python
├── *.ipynb / *.py       # Notebooks & scripts de data‑prep / fine‑tuning
└── docs/                # Schémas & ressources (optionnel)
```

## 🙋 Auteur

 **Alexandre Cameron BORGES** ─ [LinkedIn](https://fr.linkedin.com/in/alexandre-cameron-borges) 

> *“In God We Trust, All the Others We Want Data”*

---

*Dernière mise à jour : 18 mai 2025*
