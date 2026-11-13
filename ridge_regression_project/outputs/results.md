# Results

- Dataset: sklearn Diabetes (442 samples, 10 features)
- Split: 80% train / 20% test (random_state=42)
- Best alpha (5-fold CV): **31.6228**
- Best CV RMSE: **55.805**

## Test-set comparison

|                    |     R2 |    RMSE |     MAE |
|:-------------------|-------:|--------:|--------:|
| LinearRegression   | 0.4526 | 53.8534 | 42.7941 |
| Ridge (alpha=31.6) | 0.4599 | 53.4933 | 42.9495 |

## Coefficients

|     |   LinearRegression |   Ridge |
|:----|-------------------:|--------:|
| age |              1.754 |   1.991 |
| sex |            -11.512 | -10.198 |
| bmi |             25.607 |  24.535 |
| bp  |             16.829 |  15.614 |
| s1  |            -44.449 |  -6.282 |
| s2  |             24.641 |  -3.696 |
| s3  |              7.677 |  -8.715 |
| s4  |             13.139 |   7.411 |
| s5  |             35.161 |  19.518 |
| s6  |              2.351 |   3.562 |

Saved model: `outputs/final_ridge_model.joblib`
