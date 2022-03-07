from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("bvanaken/clinical-assertion-negation-bert")
model = AutoModelForSequenceClassification.from_pretrained("bvanaken/clinical-assertion-negation-bert")

input = "The patient recovered during the night and now denies any [entity] shortness of breath [entity]."

tokenized_input = tokenizer(input, return_tensors="pt")
output = model(**tokenized_input)

import numpy as np
predicted_label = np.argmax(output.logits.detach().numpy())  ## 1 == ABSENT