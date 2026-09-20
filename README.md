# Clasificarea Salariilor în Domeniul AI & Data (CatBoost GPU / Tabular ML)

O soluție de Machine Learning tabular menită să prezică probabilitatea ca un rol tehnic din ecosistemul AI & Data să depășească pragul salarial de **150.000 USD/an** (`High_Salary`).

Proiectul procesează atribute profesionale reale (senioritate, rol, mod de lucru, locație și clasificare ocupațională ISCO), antrenând un model `CatBoostClassifier` optimizat pe GPU cu tratarea nativă a variabilelor categoriale.

## Structura proiectului

- `main.ipynb` — Pipeline-ul complet: discretizarea salariului continuu în etichetă binară (`High_Salary`), izolarea și eliminarea variabilelor de tip _data leakage_, împărțirea train/test (80-20), configurarea atributelor categoriale și antrenarea pe GPU a modelului `CatBoostClassifier`.
- `.gitignore` — Exclude setul de date brut (`ai_jobs_salaries_clean.csv`).

## Metodologie & Arhitectură

1. **Definirea Țintei & Prevenirea Scurgerilor de Date (Leakage Prevention):**
   - Crearea țintei binare: `High_Salary = 1` pentru `salary_in_usd >= 150000`, altfel `0`.
   - Eliminarea completă a coloanelor asociate direct remunerației financiare: `salary`, `salary_currency`, `salary_in_usd`, `salary_outlier_flag`.
2. **Procesarea Variabilelor Categoriale:**
   - Utilizarea suportului nativ CatBoost (`cat_features`) pentru atribute de cardinalitate ridicată și medie (`job_title`, `role_family`, `employee_residence`, `company_location`, `isco_group_hint`).
3. **Optimizare Hardware & Antrenare:**
   - Model: `CatBoostClassifier` (1.000 de iterații, `task_type='GPU'`).
   - Split de date: 80% antrenare / 20% testare cu amestecare (`shuffle=True`).
4. **Rezultate:**
   - **Accuracy:** **~71.30%** pe setul de testare, un scor realist pentru distribuții de date din piața reală a muncii.

## Cum se rulează

1. Clonează repository-ul.
2. Descarcă și plasează fișierul `ai_jobs_salaries_clean.csv` în directorul rădăcină.
3. Instalează dependențele necesare:
   ```bash
   pip install pandas numpy scikit-learn catboost
   ```
