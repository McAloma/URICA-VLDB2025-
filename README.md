# URICA: a Uniformity Region affinity Identifier Capture Algorithm for pathology whole slide image retrieval

## Overview
This repository contains the codebase for the **URICA-SIGMOD26** project, which includes tools and experiments related to slice retrieval, region alignment, and performance evaluation on datasets such as TCGA and Camelyon. The repository is structured into several key components, including baseline retrieval methods, experimental modules, utilities for processing and evaluation, and scripts for visualization and drawing.

## Repository Structure

```
├── README.md           # Project documentation
├── baseline            # Baseline retrieval methods
│   ├── adjacent_retrieval.py         # Adjacent retrieval logic
│   ├── thumbnail_retrieval.py       # Thumbnail retrieval logic
├── ckpts               # Directory for model checkpoints
├── data                # Directory for input datasets
├── draw                # Visualization scripts
│   ├── draw_alpha.py                # Alpha-related visualizations
│   ├── draw_alpha_exp.py            # Alpha experimental visualizations
│   ├── draw_line_chart.py           # Line chart drawing
│   └── pics                          # Output visualizations
├── experiment          # Experimental workflows
│   ├── exp_alpha_efficience.py      # Alpha efficiency experiments
│   ├── exp_slice_retrieval.py       # Slice retrieval experiments
│   ├── materials                    # Experimental data files
│   └── results                      # Experimental results
├── requirements.txt    # Dependencies
├── src                 # Source code modules
│   ├── build                          # Data building and encoding scripts
│   ├── modules                        # Core functionalities
│   ├── test                           # Testing scripts and results
│   └── utils                          # Utility functions
```

## Key Components

### 1. **Baseline**
Contains basic implementations for adjacent and thumbnail-based retrieval algorithms.

### 2. **ckpts**
Reserved for storing model checkpoints generated during training and experiments.

### 3. **data**
Placeholder for datasets used in experiments and training.

### 4. **draw**
Includes scripts for visualizing experimental data:
- `draw_alpha.py`: Generates bar charts for alpha values.
- `draw_alpha_exp.py`: Handles experimental alpha visualizations.
- `draw_line_chart.py`: Plots speed and efficiency line charts.
- `pics/`: Stores output visualizations (e.g., `.png` files).

### 5. **experiment**
Contains scripts for running experimental workflows:
- `materials/`: JSON files with query information for different datasets.
- `results/`: Text files containing experimental outcomes.

### 6. **src**
Core source code is divided into submodules:
- `build/`: Scripts for creating and encoding datasets.
- `modules/`: Key functionalities such as anchor selection, alignment, and retrieval.
- `test/`: Testing utilities and output results.
- `utils/`: Helper scripts for evaluation, metadata handling, and WSI operations.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- Required dependencies (see `requirements.txt`)

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

### Usage

#### Running Baseline Methods
Navigate to the `baseline/` directory and execute retrieval scripts:
```bash
python adjacent_retrieval.py
python thumbnail_retrieval.py
```

#### Running Experiments
Navigate to the `experiment/` directory and run the desired experiment script:
```bash
python exp_alpha_efficience.py
```

#### Visualization
Use scripts in the `draw/` directory to generate visualizations:
```bash
python draw_alpha.py
```

### Directory Customization
- Place input datasets in the `data/` folder.
- Store checkpoints in the `ckpts/` folder for model reuse.
- Output visualizations are saved in `draw/pics/`.

## Contribution
Contributions are welcome! Please follow the steps below:
1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
Special thanks to the contributors and the open-source community for providing tools and datasets utilized in this project.
