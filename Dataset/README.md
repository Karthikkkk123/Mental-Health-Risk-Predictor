# NHANES Dataset Description

This directory contains data from the National Health and Nutrition Examination Survey (NHANES), which has been preprocessed and merged for use in mental health risk prediction.

## Files Description

### Main Dataset File
- `nhanes_merged.csv`: The preprocessed and merged dataset containing all relevant features

### Source Files (NHANES Components)
- `SLQ_L.xpt`: Sleep Disorders data
- `ALQ_L.xpt`: Alcohol Use data
- `MCQ_L.xpt`: Medical Conditions data
- `DPQ_L.xpt`: Depression Screener data
- And other relevant NHANES components

## Variables Description

### Sleep-related Variables (SLQ)
- `SLQ300`: Trouble falling asleep
- `SLQ310`: Wake up during sleep
- `SLQ320`: Wake up too early
- `SLQ330`: Feel unrested during day
Scale: 0 (Never) to 4 (Almost Always)

### Alcohol Use Variables (ALQ)
- `ALQ111`: Frequency of alcohol consumption
- `ALQ121`: Average drinks per day
- `ALQ130`: Frequency of binge drinking
Scale: 0 (Never) to 4 (Very Frequent)

### Medical Conditions Variables (MCQ)
- `MCQ010`: Presence of chronic conditions
- `MCQ160A`: Diagnosed sleep disorders
- `MCQ160B`: Diagnosed anxiety/depression
- `MCQ160C`: Other mental health conditions
Scale: 0 (No) or 1 (Yes)

## Data Processing

The merged dataset has undergone several preprocessing steps:
1. Removal of irrelevant variables
2. Handling of missing values
3. Normalization of numerical features
4. Encoding of categorical variables
5. Creation of composite risk scores

## Usage Notes

- The dataset is provided for research and educational purposes only
- Personal identifiers have been removed
- Missing values have been handled appropriately
- All variables have been normalized/standardized

## Citation

When using this dataset, please cite both this repository and the original NHANES source:

```
National Center for Health Statistics (NCHS). National Health and Nutrition Examination Survey Data. 
Hyattsville, MD: U.S. Department of Health and Human Services, Centers for Disease Control and Prevention.
```

## Links

- [NHANES Website](https://www.cdc.gov/nchs/nhanes/index.htm)
- [NHANES Data Documentation](https://wwwn.cdc.gov/nchs/nhanes/)