
# Google Drive + Colab Setup Guide

## Drive'a Yüklenecek Dosyalar

```

MyDrive/fraudDataset/

├── creditcard/

│   └── creditcard.csv                    ← Kaggle'dan indir

├── leie/

│   └── (LEIE exclusion files)            ← OIG'dan indir

├── medicare/

│   └── Combined_LEIE_Medicare_2017_2019_DOWNSIZED_1mil.csv

└── Fraud-Detection-GT/                   ← bu proje klasörü

    ├── notebooks/                        ← 6 adet .ipynb

    │   ├── 01_EDA_and_Baseline.ipynb

    │   ├── 02_Graph_Construction.ipynb

    │   ├── 03_Classical_Graph_Analysis.ipynb

    │   ├── 04_Spectral_Methods.ipynb

    │   ├── 05_GNN_Models.ipynb

    │   └── 06_Comparison_and_Results.ipynb

    ├── src/                              ← Python modules

    │   ├── __init__.py

    │   ├── graph_builder.py

    │   ├── graph_features.py

    │   ├── gnn_models.py

    │   ├── baseline_loader.py

    │   ├── evaluation.py

    │   └── visualization.py

    ├── config.yaml                       ← (opsiyonel, notebook kendi ayarlıyor)

    ├── data/graphs/                      ← notebook'lar burayı otomatik oluşturur

    │   ├── credit_card/

    │   └── medicare/

    └── outputs/                          ← notebook'lar burayı otomatik oluşturur

        ├── figures/

        ├── metrics/

        └── models/

```

## Adım Adım Colab Setup

### 1. Drive'a Yükle

Şunu Drive'a kopyala (tüm klasörü sürükle-bırak):

```

Fraud-Detection-GT/ → MyDrive/fraudDataset/Fraud-Detection-GT/

```

Dataset'leri ayrı yükle:

```

creditcard.csv     → MyDrive/fraudDataset/creditcard/creditcard.csv

Medicare CSV       → MyDrive/fraudDataset/medicare/Combined_LEIE_Medicare_2017_2019_...csv

```

### 2. Colab'da Notebook Aç

Google Drive > fraudDataset > Fraud-Detection-GT > notebooks > 01_EDA_and_Baseline.ipynb

→ Sağ tıkla > "Open with Google Colab"

### 3. Her Notebook'un ilk 2 Cell'ini Çalıştır

- **Cell 1** → Drive mount (izin ver)
- **Cell 2** → Paket kurulum (sadece ilk oturumda gerekli)

### 4. Sırayla Çalıştır

01 → 02 → 03 → 04 → 05 → 06

Her notebook sonucunu **Drive'a kaydeder**

## Notebook 05 için GPU Ayarı (GNN Eğitimi)

Runtime → Change runtime type → T4 GPU → Save

## Dataset İndirme Linkleri

| Dataset             | Link                                                                                 |

| ------------------- | ------------------------------------------------------------------------------------ |

| Credit Card Fraud   | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud                              |

| LEIE Exclusion List | https://oig.hhs.gov/exclusions/exclusions_list.asp                                   |

| Medicare Part D     | https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers |
