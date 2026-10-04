# Laptop & Desktop Price Prediction

Feature engineering project to predict computer prices based on hardware 
specifications, built as part of a 4-person team.

### Version

*en: English

*esp: Spanish

## Problem Statement

With thousands of laptop and desktop configurations on the market, pricing is 
driven by a complex combination of hardware specs, brand, OS, and connectivity. 
This project applies advanced feature engineering techniques to prepare a 
100,000-record dataset for price prediction modeling.

## Dataset

- **Source:** Kaggle — Computer Prices Dataset
- **Records:** 100,000 devices
- **Features:**
  - `device_type` — Laptop or Desktop
  - `brand`, `model`, `release_year`, `os`, `form_factor`
  - `cpu_brand`, `cpu_model`, `cpu_tier`, `cpu_cores`, `cpu_threads`
  - `gpu_brand`, `gpu_model`, `gpu_tier`, `vram_gb`
  - `ram_gb`, `storage_gb`, `storage_drive_count`
  - `wifi`, `bluetooth`, `weight_kg`, `warranty_months`
- **Target:** `price` (USD)

## Tech Stack

- Python 3.12
- Pandas, NumPy, SciPy
- Scikit-learn (KBinsDiscretizer, PowerTransformer, MinMaxScaler, 
  OneHotEncoder, OrdinalEncoder)
- Category Encoders (BinaryEncoder)
- Matplotlib, Seaborn
- Google Colab

## Methodology

Focus: **Feature Engineering (FE)** — transforming raw specs into 
model-ready features

1. **EDA** — Analyzed 27 features across 100,000 records, identified 
   missing values and outliers
2. **Encoding Strategy** — Compared OrdinalEncoder, OneHotEncoder, and 
   BinaryEncoder for high-cardinality categorical features (e.g., gpu_model 
   with 49 unique values)
3. **Scaling** — Applied MinMaxScaler and PowerTransformer (Yeo-Johnson) 
   to normalize skewed distributions
4. **Feature Creation** — Engineered `years_since_release` from 
   `release_year` to capture depreciation effect
5. **Discretization** — Applied KBinsDiscretizer for relevant numerical 
   features

## Key Findings

- Dataset was clean with zero missing values across most features
- GPU model was the highest-cardinality feature (49 unique values) — 
  BinaryEncoder reduced dimensionality significantly vs OneHotEncoder
- PowerTransformer effectively normalized price distribution skewness
- `cpu_tier` and `gpu_tier` were the strongest predictors of price

## Team

Built collaboratively as part of a 4-person team (Equipo 16) in the 
Applied Data Science course — Tecnológico de Monterrey MNA Program.

## What I Learned

- Handling high-cardinality categorical features at scale (100K records)
- Choosing the right encoding strategy based on cardinality and model type
- Feature engineering as the most impactful step in the ML lifecycle
- Collaborative data science workflow with version control
