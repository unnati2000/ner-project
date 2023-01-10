import json 

f = open('dataset.json')
data = json.load(f)

training_data = {'classes': [], 'annotations':[]}
training_data['classes'] = data['classes']

for item in data['annotations']:
    temp_dict = {}
    temp_dict['text'] = item[0]
    temp_dict['entities'] = []
   
    for annotation in item[1]['entities']:
        temp_dict['entities'].append((annotation[0], annotation[1], annotation[2]))
    
    training_data['annotations'].append(temp_dict)


import pandas as pd
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

for text, annot in tqdm(training_data): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    doc.ents = ents # label the text with the ents
    db.add(doc)

db.to_disk("./train.spacy") # save the docbin object
