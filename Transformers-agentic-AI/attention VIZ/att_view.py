from transformers import AutoTokenizer, AutoModel, utils , BertTokenizer, BertModel
from bertviz import model_view , head_view
from bertviz.neuron_view import show
from bertviz.transformers_neuron_view import BertModel, BertTokenizer

utils.logging.set_verbosity_error()  # Suppress standard warnings


def MV(input_text):

    model_name = "microsoft/xtremedistil-l12-h384-uncased"
    # Find popular HuggingFace models here: https://huggingface.co/models
    # input_text = "The cat sat on the mat"
    model = AutoModel.from_pretrained(model_name, output_attentions=True)  # Configure model to return attention values
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer.encode(input_text, return_tensors="pt")  # Tokenize input text
    outputs = model(inputs)  # Run model
    attention = outputs[-1]  # Retrieve attention from model outputs
    tokens = tokenizer.convert_ids_to_tokens(inputs[0])  # Convert input ids to token strings
    m_view = model_view(attention, tokens, html_action="return")  # Display model view

    return m_view



def HV(input_text):
    model_version = 'bert-base-uncased'
    model = BertModel.from_pretrained(model_version, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_version)
    # sentence_a = "The cat sat on the mat"
    sentence_b = "The cat lay on the rug"
    inputs = tokenizer.encode_plus(input_text, sentence_b, return_tensors='pt')
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']
    attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    sentence_b_start = token_type_ids[0].tolist().index(1)
    input_id_list = input_ids[0].tolist() # Batch index 0
    tokens = tokenizer.convert_ids_to_tokens(input_id_list)
    h_view=head_view(attention, tokens, sentence_b_start)
    
    return h_view


def NV(input_text):
    model_type = 'bert'
    sentence_b = "The cat lay on the rug"
    model_version = 'bert-base-uncased'
    model = BertModel.from_pretrained(model_version, output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=True)
    n_view=show(model, model_type, tokenizer, input_text, sentence_b, layer=4, head=3)
    
    
    return n_view