from pprint import pprint
import functools
 
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, CamembertForMaskedLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, f1_score
 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm.notebook import tqdm

camembert = CamembertForMaskedLM.from_pretrained('camembert-base')
print(camembert)
# Visualiser la couches d'embeddings
camembert.roberta.embeddings