# Bin Packing Approximation Algorithms: Analysis and Comparison

![Algorithm Comparison Visualization](Figure_1.png)

## Abstract
This project explores and compares several approximation algorithms for the Bin Packing Problem, including:
- Next Fit (NF)
- First Fit (FF)
- First Fit Decreasing (FFD)
- Harmonic-based methods
- Linear programming-based approach (Gilmore-Gomory LP + Karmarkar-Karp rounding)

We evaluate algorithms using:
- Average waste per bin
- Average bin utilization
- Total number of bins used
- Computational efficiency

Using Monte Carlo simulations across multiple input sizes, we analyze how each algorithm scales and identify key trade-offs between solution quality and runtime performance.

## Key Features
- Implementation of 5+ bin packing algorithms
- Comprehensive performance metrics
- Monte Carlo simulation framework
- Comparative analysis across input sizes
- Visualization of performance trends

## Installation
```bash
git clone https://github.com/ZeelDanani/Bin-Packing-Algorithm-and-Analysis.git
cd Bin-Packing-Algorithm-and-Analysis
pip install -r requirements.txt
