import os
import torch
import pickle
import common as lib
import model as tcm
import gitfilter as gfs
# Application of the model to a coding project

# directory const for project input, the model you want to use, and the vocab file made alongside the model
INPUT_DIRECTORY = lib.INSTALLATION_DIRECTORY + "\\src"
MODEL_FILEPATH = lib.INSTALLATION_DIRECTORY + "\\model\\model_825_3699.pth"
VOCAB_FILEPATH = lib.INSTALLATION_DIRECTORY + "\\embedding\\vocab.pkl"

def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()

def read_vocab(path):
    #read vocabulary pkl
    pkl_file = open(path, 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()
    return vocab

# uses vocab to translate words in a comment into int token
def tokenizeString(vocab, com):
    # specific to torchtext 0.6.0
    return [vocab[token] for token in com.strip().split(" ")]

# load vocab
vocab = read_vocab(VOCAB_FILEPATH)
text_pipeline = lambda x: tokenizeString(vocab, x)
# define model
model = tcm.TextClassificationModel(len(vocab), 64, len(lib.labelMap)).to("cpu")
try:
    model.load_state_dict(torch.load(MODEL_FILEPATH))
except:
    print("Loaded model vocab dimension does not match vocab dimension loaded from file.")
    print("Model and Vocab should come from the same training run.")
    quit()
model.eval() # sets the dropout and batch normalization layers to evaluation mode. Else will get inconsistant inference results...

# build our GitFilterService
git_filter = gfs.GitFilterService(INPUT_DIRECTORY)

# place to store all comments
all_files=[]

# go through all project files, ignoring ones that do not get through gitfilterservice
# and label the file's comments
# get all the files in the src dir and loop though the directory
for root, dirs, files in os.walk(INPUT_DIRECTORY):
    # get directory depth and print accordingly
    level = root.replace(INPUT_DIRECTORY, '').count(os.sep)
    indent = ' ' * 4 * (level)
    subindent = ' ' * 4 * (level + 1)

    rel_directory = root.replace(INPUT_DIRECTORY, '')

    if git_filter.git_filter_allow(rel_directory):
        for filename in files:
            if not git_filter.git_filter_allow(filename):
                continue

            file_path = os.path.join(root, filename)

            #open the file at filepath, and read all code in the file
            code = ""
            with open(file_path, encoding='utf-8-sig') as f:
                try:
                    code = f.readlines()
                    f.close()
                except:
                    print("could not open file: ", filename)
                    continue

            # skip if file is empty
            if len(code) == 0:
                continue

            # define comment objects in file object
            new_file = lib.FileContent(file_path, len(code))

            # find comments and comment blocks
            # adjacent single comments on multiple lines are treated as one comment
            comment_prev_flag = True
            comment_block_flag = False
            comment_block_written = False
            comment = ""
            lineNumber = 0
            for line in code:
                lineNumber += 1
                # determine if is a comment
                if "//" in line and not comment_prev_flag:
                    comment_prev_flag = True
                    comment = line
                elif "//" in line and comment_prev_flag:
                    comment = comment + " " + line
                elif "/*" in line:
                    comment_block_flag = True
                    comment = line
                elif "*/" in line:
                    comment_block_flag = False
                    # define comment objects in file object
                    new_comment = lib.Comment(lineNumber, filename, lib.cleanComment(comment))
                    new_comment.labelId = predict(new_comment.comment, text_pipeline) # label the comment
                    new_comment.label = lib.commentLabel[new_comment.labelId]
                    new_file.add_comment(new_comment)
                elif comment_block_flag:
                    comment = comment + " " + line
                else:
                    if comment != "":
                        comment_prev_flag = False
                        # define comment objects in file object
                        new_comment = lib.Comment(lineNumber, filename, lib.cleanComment(comment))
                        new_comment.labelId = predict(new_comment.comment, text_pipeline) # label the comment
                        new_comment.label = lib.commentLabel[new_comment.labelId]
                        new_file.add_comment(new_comment)
                        comment = ""

            # add file with comments to full list of files
            all_files.append(new_file)

# print the results to console
# console print with file tree and label results
for root, dirs, files in os.walk(INPUT_DIRECTORY):
    # get directory depth and print accordingly
    level = root.replace(INPUT_DIRECTORY, '').count(os.sep)
    indent = ' ' * 4 * (level)
    subindent = ' ' * 4 * (level + 1)
    dataindent = ' ' * 4 * (level + 2)

    # print the current directory
    print('{}{}\\'.format(indent, os.path.basename(root)))

    # print files in the directory with their comment score/labels if any
    # perform simple search here to match up current os walk file with our results
    for filename in files:
        if filename is not None:
            for f in all_files:
                if f.file_name == filename:
                    score_message = f.score_message()
                    if score_message != '' and score_message is not None:
                        print('{}{}'.format(subindent, filename))
                        print('{}{}'.format(dataindent, score_message))
