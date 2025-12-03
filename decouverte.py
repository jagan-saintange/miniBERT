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

# Visualiser la couches d'embeddings
print(camembert.roberta.embeddings)

batch_sentences = [
    "Vous savez où est le <mask> le plus proche?",
    "Mon jeux-video préféré est <mask>.",
    "Je cherche urgemment un endroit pour <mask>.",
    "Jagan est vraiment pas une personne <mask>"
]
tokenizer = AutoTokenizer.from_pretrained('camembert-base')

tokenizer_output = tokenizer(
    batch_sentences
)
pprint(tokenizer_output, width=150)

pprint([tokenizer.convert_ids_to_tokens(input_ids) for input_ids in tokenizer_output['input_ids']], width=150)

tokenizer_output = tokenizer(
    batch_sentences,
    padding="max_length"
)
pprint(tokenizer_output, compact=True, width=150)

tokenizer_output = tokenizer(
    batch_sentences,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)
pprint(tokenizer_output, width=150)

with torch.no_grad():
    model_output = camembert(**tokenizer_output, output_hidden_states=True)
    model_output

def get_probas_from_logits(logits):
    return logits.softmax(-1)


def visualize_mlm_predictions(tokenizer_output, model_output, tokenizer, nb_candidates=10):
    # Decode the tokenized sentences and clean-up the special tokens
    decoded_tokenized_sents = [sent.replace('<pad>', '').replace('<mask>', ' <mask>') for sent in tokenizer.batch_decode(tokenizer_output.input_ids)]

    # Retrieve the probas at the masked positions
    masked_tokens_mask = tokenizer_output.input_ids == tokenizer.mask_token_id
    batch_mask_probas = get_probas_from_logits(model_output.logits[masked_tokens_mask])

    for sentence, mask_probas in zip(decoded_tokenized_sents, batch_mask_probas):
        # Get top probas and plot them
        top_probas, top_token_ids = mask_probas.topk(nb_candidates, -1)
        top_tokens = tokenizer.convert_ids_to_tokens(top_token_ids)
        bar_chart = px.bar({"tokens": top_tokens[::-1], "probas": list(top_probas)[::-1]},
                        x="probas", y="tokens", orientation='h', title=sentence, width=800)
        bar_chart.show(config={'staticPlot': True})

visualize_mlm_predictions(tokenizer_output, model_output, tokenizer)

def take_first_embedding(embeddings, attention_mask=None):
    return embeddings[:, 0]

def average_embeddings(embeddings, attention_mask):
    return (attention_mask[..., None] * embeddings).mean(1)

first_tok_sentence_representations = take_first_embedding(token_embeddings, tokenizer_output.attention_mask)
avg_sentence_representations = average_embeddings(token_embeddings, tokenizer_output.attention_mask)

first_tok_sentence_representations.shape, avg_sentence_representations.shape

for sent_id_1, sent_id_2 in [[0, 1], [2, 1], [2, 0]]:
    first_tok_similarity_score = F.cosine_similarity(first_tok_sentence_representations[sent_id_1], first_tok_sentence_representations[sent_id_2], dim = -1)
    avg_similarity_score = F.cosine_similarity(avg_sentence_representations[sent_id_1], avg_sentence_representations[sent_id_2], dim = -1)

    print(f"{batch_sentences[sent_id_1]}    vs.    {batch_sentences[sent_id_2]}")
    print(f"Score (first_tok) : {first_tok_similarity_score}")
    print(f"Score (average) : {avg_similarity_score}\n")