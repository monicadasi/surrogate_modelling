# Thesis Outline

Date: March 1, 2023

1. Abstract
2. Introduction
    1. Problem Description
    2. Motivation
    3. Scope of Work and Research Question
    4. Outline of this exposition
3. Fundamentals
    1. Frequency Response Functions (FRFs)
    2. Taylor’s series second order approximation
    3. Perpendicular bi-sector
    4. Circle equation
4. Literature Review
    1. Surrogate Modelling Framework
        1. Applications and Limitations in Frequency Domain
    2. Review of surrogate methods
        1. Polynomial Chaos Expansion (PCA)
        2. Orthogonal Basis Surrogate (OBS) / Frequency Mapped OBS (FMOBS)
        3. Kriging
    3. Summary
5. Methodology and Implementation
    1. Implementation overview
        1. Tools and Technology stack
        2. Data Parsing and Visualization
        3. Extraction of Neighboring Frequencies
        4. Investigation on various circle extraction techniques
            1. Perpendicular bi-sector method
            2. Circle equation method
            3. Least squares method for circle fit
        5. Polynomial Regression
            1. Model building
            2. Data Down Sampling 
            3. Predictions
        6. Mean Squared Error (MSE)
        7. Summary
    2. Optimization and automation
        1. Circle extraction - adapting the number of neighboring freq. points
        2. Adaptation of Polynomial Degree (based on model score)
        3. Summary
6. Results and Discussion
7. Conclusion and Future Research

Appendix

Bibliography