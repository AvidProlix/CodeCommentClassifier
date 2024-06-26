import time
import random
import re
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import pickle
import common as lib
import model as tcm

# Training file to ingest and process labeled data and train model

## const
INPUT_FILE = lib.INSTALLATION_DIRECTORY + "\\data\\Refined2702_2KSetV2.txt"
OUTPUT_DIR = lib.INSTALLATION_DIRECTORY + "\\model\\"
VOCAB_OUTPUT_PATH = lib.INSTALLATION_DIRECTORY + "\\embedding\\vocab.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data parameters
TRAINING_DATA_BATCH_SIZE = 64
VALIDATION_DATA_BATCH_SIZE = 32
TEST_DATA_BATCH_SIZE = 32
TRAIN_RATIO = 0.8 # ratio of data that will be used for training
VALIDATION_TEST_THRESH = 0.9
DATA_SHUFFLE = True # will shuffle all data before the split for training, validation, eval

# Hyperparameters
EPOCHS = 8  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training
TRAIN_LOG_INTERVAL = 24

## ingest labeled data

# read the data file
# expecting: label, comment data
labeledData = ""
with open(INPUT_FILE, encoding='utf-8-sig') as f:
    try:
        labeledData = f.readlines()
        f.close()
    except:
        print("could not open file: ", INPUT_FILE)

comments = []
# foreach line, build Comment object
for entry in labeledData:
    # remove article words
    redux_entry = re.sub('(\s+)'+ "|".join(lib.articleList) + '(\s+)','\1\3', entry)
    # add Comment Object into ordered list
    comments.append(redux_entry)

# check data labels and sequence lengths in dataset to find invalid lines
data_inconsistancy_found = False
index = 1
for data in labeledData:
    # looking for a 2 length arr of data
    if(len(data.split(",")) != 2):
        print("Line: ", index, ". Incorrect number of arguments found: ", data.replace('\n', ''))
        data_inconsistancy_found = True
    else:
        # check if the label is in labelMap
        try:
            label = data.split(",")[0]
            labelId = lib.labelMap[label]
        except:
            print("Line: ", index, ". Invalid label found: ", data.replace('\n', ''))
            data_inconsistancy_found = True
    index += 1

# exit if any data read errors were found
if(data_inconsistancy_found):
    quit()

# Build pipelines to transform data into integers / tokens

# get basic tokenizer
tokenizer = get_tokenizer("basic_english") # normalizes string then splits by space

# gets tokens used in all comments to build vocab
def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

# uses vocab to translate words in a comment into int token
def tokenizeString(vocab, com):
    # syntax specific to torchtext 0.6.0
    return [vocab[token] for token in com.strip().split(" ")]

# saves vocab embed layer to a pickle file
def save_vocab(vocab):
    output = open(VOCAB_OUTPUT_PATH, 'wb')
    pickle.dump(vocab, output)
    output.close()

# get list of unlableled comments only to feed to vocab
commentData = list()
for com in comments:
    commentData.append(com.split(",")[-1])

# build vocab using our tokenizer and comment data
vocab = build_vocab_from_iterator(yield_tokens(commentData))
save_vocab(vocab)

# create lambda function pipelines to streamline tokenization
text_pipeline = lambda x: tokenizeString(vocab, x)
label_pipeline = lambda x: int(lib.labelMap[x])

# Datasets

# shuffle all data and split up for training, validation, and testing
if DATA_SHUFFLE:
    random.shuffle(comments) # shuffle data (volatile but helpful for the small dataset)
train_iter = comments[:int(TRAIN_RATIO*len(comments))] # train with TRAIN_RATIO% data
valid_iter = comments[int(TRAIN_RATIO*len(comments)):int(VALIDATION_TEST_THRESH*len(comments))]
test_iter = comments[int(VALIDATION_TEST_THRESH*len(comments)):]

# takes data input and uses pipelines to encode data (labels and features to int)
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for item in batch:
        _label, _text = item.split(",")
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

# DataLoader
train_dataloader = DataLoader(
    train_iter, 
    batch_size=TRAINING_DATA_BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_batch)
valid_dataloader = DataLoader(
    valid_iter, 
    batch_size=VALIDATION_DATA_BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_batch)
test_dataloader = DataLoader(
    test_iter, 
    batch_size=TEST_DATA_BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_batch)

# get the number of labels used in the dataset
num_class = len(set([label.split(",")[0] for label in train_iter]))

# if there is a mismatch in number of labels, then exit before training. Will except in loss fxn
if num_class != len(lib.labelMap):
    print("Dataset label representation mismatch with expected label set.")
    quit()

# define the Model
vocab_size = len(vocab) # length of vocab tokenizer map
emsize = 64             # embedding layer size
model = tcm.TextClassificationModel(vocab_size, emsize, num_class).to(device)

# define what training the model means
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = TRAIN_LOG_INTERVAL

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| accuracy {:8.3f}".format(
                    epoch, idx, len(dataloader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0

# accuracy eval during training
def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        avg_loss = 0
        item_count = 0
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            avg_loss = avg_loss + loss.item()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            item_count += 1
        avg_loss = avg_loss / item_count
        print("avg loss: {:1.4f}".format(avg_loss))
    return total_acc / total_count

## train model

# optimizer, loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

# train over n epochs
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | "
        "valid accuracy {:8.3f} ".format(
            epoch, time.time() - epoch_start_time, accu_val
        )
    )
    print("-" * 59)
    print("")

# check the model against test dataset
print("Checking the results of test dataset.")
accu_test = evaluate(test_dataloader)
print("test accuracy {:8.3f}".format(accu_test))

# save model
# filename is model_###(accuracy ie.82.6%=>826)_####(timetics).pth
model_save_filepath = OUTPUT_DIR + "model_" + str(int((accu_test % 1) * 1000)) + "_" + str(int(time.time() % 10000)) + ".pth"
torch.save(model.state_dict(), model_save_filepath) # saves model weights to file/disk
