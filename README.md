# RClass — Classification by Rational Approximation

RClass is a classification framework based on rational function approximation. Our approach approximates complex decision boundaries by modeling the classifier as a ratio of two polynomials, offering an interpretable alternative to deep learning models. The project leverages SageMath, Gurobi (under an academic license), and various Python libraries for data processing and visualization.

## Overview

- **Classifier Approach:**  
  Our classifier uses a rational function of the form
$`R(x)=\frac{p(x)}{q(x)}`\`$
  where \(p(x)\) and \(q(x)\) are polynomials whose degrees can be tuned independently. Extensive experiments on the MNIST dataset have shown that a (2,1) configuration (numerator degree = 2, denominator degree = 1) often provides an optimal trade-off between accuracy and training time.

- **Experimental Framework:**  
  The project includes a complete experimental pipeline:
  - Data loading and preprocessing (scaling, PCA, polynomial feature expansion)
  - Optimization using SageMath and the Gurobi optimizer to compute the best rational approximation
  - Detailed evaluation metrics and analysis (accuracy, macro F1, confusion matrices)
  - Visualization of results and performance trends

## Features

- **Rational Function Classifier:**  
  Implements a novel classifier using rational approximation techniques.
- **Flexible Experimentation:**  
  Easily configurable polynomial degrees, PCA dimensionality, and training sample sizes.
- **Optimization with Gurobi:**  
  Utilizes the powerful Gurobi optimizer within a SageMath framework.
- **Comprehensive Evaluation:**  
  Generates detailed performance metrics and visualizations for in-depth analysis.

## Installation

### Prerequisites

- Python 3.x
- SageMath (version 14 or later recommended)
- Gurobi Optimizer (with an academic license)
- Common Python packages: `numpy`, `matplotlib`, `scikit-learn`, and `tensorflow` (for data loading)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Gambotch1/RClass-project.git
   cd RClass
