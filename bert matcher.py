#bert matcher
from transformers import pipeline
from transformers import DistilBertTokenizerFast
import pandas as pd
from summa import summarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
from sklearn.metrics import confusion_matrix
import numpy as np

#setup classifier and tokenizer
classifier = pipeline("text-classification")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

#read the data
df=messages = pd.read_table("/uufs/chpc.utah.edu/common/home/fengj-group1/Code/BERT/Data/~BERTdata.tsv", sep="\t", names=["id","approved", "description"])
df['description'] = df.apply(lambda row : summarizer.summarize(row["description"], words=300), axis = 1)
df.head()
X = list(df['description'])
Y = list(df['approved'])

#split the data

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

#build encodings
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
))


training_args = TFTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation (MEMORY ALLOWING UP TO 8 and 16)
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset             # evaluation dataset
)
trainer.train()

#make the prediction and the confusion matrix
output=trainer.predict(test_dataset)[1]
cm=confusion_matrix(y_test,output)

f=open('/scratch/general/vast/u1188824/bertMatrix1.txt', 'w')

#WRITE THE CM
np.apply_along_axis(f.write, axis=1, arr=cm)



#import the post 2014 data
test = pd.read_table("/uufs/chpc.utah.edu/common/home/fengj-group1/Code/BERT/Data/BERTdataPreAlice.tsv", sep="\t", names=["id","approved", "description"])
test['description'] = test.apply(lambda row : summarizer.summarize(row["description"], words=300), axis = 1)
X_test =list(test['description'])
Y_test = list(test['approved'])

#retrain the model on the entreity of the dataset
X_train = list(df['description'])
Y_train = list(df['approved'])


#build encodings
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

#convert to tensors
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
))

#run predictions

training_args = TFTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation (MEMORY ALLOWING UP TO 8 and 16)
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset             # evaluation dataset
)

trainer.train()

#write the confusion matrix, and the predictions
output=trainer.predict(test_dataset)[1]
cm=confusion_matrix(y_test,output)



#WRITE THE CM
np.apply_along_axis(f.write, axis=1, arr=cm)

f.close
f=open('/scratch/general/vast/u1188824/bertPredictions1.csv', 'w')
f.write("id, predicted, actual")

#WRITE THE PREDICTED OUTCOMES OF THE PRE ALICE DATA
for i in range(0,len(output)):
    f.write(test[id][i] + ", " + output[i] + "," + y_test[i])
f.close

