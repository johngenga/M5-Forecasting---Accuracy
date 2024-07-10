# M5 Forecasting Challenge

Welcome to the M5 Forecasting Challenge project! This repository contains code and resources for participating in the M5 Forecasting Accuracy competition hosted on Kaggle.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The M5 Forecasting Challenge aims to predict sales for Walmart over a period of time. Accurate forecasts are critical for effective supply chain management and can help reduce costs, optimize inventory, and improve customer satisfaction.

## Dataset

The dataset consists of several files, including:

- `sales_train_validation.csv`: Historical daily sales data for various products across different stores.
- `calendar.csv`: Contains information on the dates the sales data correspond to, including holidays and special events.
- `sell_prices.csv`: Historical selling prices of the products.
- `sample_submission.csv`: A sample submission file for the competition.

## Project Structure

The repository is organized as follows:


- `data/`: Contains the dataset files.
- `scripts/`: Contains the scripts for data exploration and model training.
- `results/`: Contains the results, including feature importance plots.
- `README.md`: This file.
- `requirements.txt`: Contains the required Python packages.

## Setup Instructions

1. **Clone the repository**:

    ```bash
    git clone https://github.com/johngenga/M5-Forecasting-Challenge.git
    cd M5-Forecasting-Challenge
    ```

2. **Create a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:

    ```bash
    pip install -r requirements.txt
    ```

4. **Download the dataset**: Download the dataset from Kaggle and place the files in the `data/` directory.

## Usage

1. **Run the data exploration script**:

    ```bash
    python scripts/data_exploration_v2.py
    ```

2. **View the results**: The results, including feature importance plots, will be saved in the `results/` directory.

## Results

- Detailed results and analysis will be added here once the models are trained and evaluated.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
