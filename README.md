# Early warning of foodborne disease outbreaks using an enhanced graph attention network on patient association graphs
## Requirements

1. Install Python >= 3.10.

2. Required dependencies can be installed by: 
   
   ```
   pip install -r requirements.txt
   ```

### graph construction.py 
This code processes raw foodborne disease data and constructs standardized graph structures. It generates patient node feature tables and edge relation tables for graph neural network models. 

### main.py 
This code implements the GatedResGAT with 5-fold cross-validation for foodborne disease case classification. It trains the model on patient graph data, evaluates performance, and outputs prediction results. This code takes preprocessed node features and edge relation tables as input, completes standardized training and testing, and identifies clustered and sporadic disease cases.
