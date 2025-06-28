import streamlit as st
import torch
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn

# -----------------------------
# 1. Define the neural network model architecture
# -----------------------------
class SimpleFFNN(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512, hidden2=128, output_size=1):
        super(SimpleFFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# -----------------------------
# 2. Load the trained model from file
# -----------------------------
model = SimpleFFNN()
model_path = os.path.join("models", "BRD4_model.pth")  # Path to model file
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# -----------------------------
# 3. Convert SMILES string to molecular fingerprint
# -----------------------------
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# -----------------------------
# 4. Streamlit Web Application Interface
# -----------------------------
st.title("BRD4 pIC50 Predictor")

# App description and importance
st.markdown("""
**BRD4 pIC50 Predictor** is a web application that estimates the pIC50 values of compounds targeting the BRD4 protein, based on their canonical SMILES.

**Why it matters:**  
BRD4 (Bromodomain-containing protein 4) is a key epigenetic regulator involved in cancer, inflammation, and other diseases.  
This tool helps researchers virtually screen and prioritize compounds before experimental testing, saving time and resources in early-stage drug discovery.
""")

# Input field for SMILES
smiles_input = st.text_input("Enter Canonical SMILES:", placeholder="Example: CC(=O)OC1=CC=CC=C1C(=O)O")

# Predict button
if st.button("Predict"):
    if smiles_input:
        fingerprint = smiles_to_fingerprint(smiles_input)
        if fingerprint is not None:
            x = torch.tensor(fingerprint, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prediction = model(x).item()

            # Display result with color based on threshold
            if prediction < 6:
                st.markdown(f"<p style='color:red; font-size:24px;'>Predicted pIC50: {prediction:.3f}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='color:green; font-size:24px;'>Predicted pIC50: {prediction:.3f}</p>", unsafe_allow_html=True)
