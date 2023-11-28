'''
install relevant packages including:
pip install transformers datasets evaluate
'''
'''
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf
import keras

import evaluate

from keras import layers
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import create_optimizer
from transformers import TFAutoModelForSequenceClassification

#Callback class for time history 
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# Function for approximate training time with each layer independently trained
def get_average_layer_train_time(epochs):
    
    #Loop through each layer setting it Trainable and others as non trainable
    results = []
    for i in range(len(model.layers)):
        
        layer_name = model.layers[i].name    # storing name of layer for printing layer
        
        # Setting all layers as non-Trainable
        for layer in model.layers:
            layer.trainable = False
            
        # Setting ith layers as trainable
        model.layers[i].trainable = True
        
        # Compile
        model.compile(optimizer=optimizer, metrics=['acc'])
        
        # Fit on a small number of epochs with callback that records time for each epoch
        model.fit(x=tf_train_set, validation_data=tf_validation_set,      
              epochs=epochs,
              callbacks = [time_callback])
        
        results.append(np.average(time_callback.times))
        # Print average of the time for each layer
        print(f"{layer_name}: Approx (avg) train time for {epochs} epochs = ", np.average(time_callback.times))
    return results

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


# Main func
if __name__ == "__main__":
    print(f"Creating Callback")
    time_callback = TimeHistory()

    imdb = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
    
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    batch_size = 16
    num_epochs = 5
    batches_per_epoch = len(tokenized_imdb["train"])
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)


    # Model definition
    """
    inp_dims = (3,)
    out_dims = 2
    model = keras.Sequential(
        [
            keras.Input((inp_dims,)),
            layers.Embedding(vocab_size, 256, input_length=inp_dims),
            layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(out_dims, activation='softmax'),
        ]
    )
    model = tf.keras.Model(inp, out)
    """
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    tf_train_set = model.prepare_tf_dataset(
        tokenized_imdb["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_imdb["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )
    model.summary()

    runtimes = get_average_layer_train_time(5)
    plt.plot(runtimes)

'''

from datasets import load_dataset

dataset = load_dataset("yelp_review_full")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)