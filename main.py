import os
import warnings
from ontology_dc8f06af066e4a7880a5938933236037.simple_text import SimpleText

from openfabric_pysdk.context import OpenfabricExecutionRay
from openfabric_pysdk.loader import ConfigClass
from time import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from openfabric_pysdk.utility import SchemaUtil



tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")

############################################################
# Callback function called on update config
############################################################
def config(configuration: ConfigClass):
    # TODO Add code here
    pass


############################################################
# Callback function called on each execution pass
############################################################
def execute(request: SimpleText, ray: OpenfabricExecutionRay) -> SimpleText:
    output = []
    for text in request.text:
        # Tokenize the input text
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_tensors="pt"
        )

        # Generate the model's response
        output_ids = model.generate(
            inputs["input_ids"],
            max_length=50,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        # Decode the response from the model
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output.append(response)

    return SimpleText(dict(text=output))
