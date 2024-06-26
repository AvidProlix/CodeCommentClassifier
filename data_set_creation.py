import os
from datetime import datetime
import common as lib

# takes all of the cs files from open source projects and extracts the comments from them into a new
# file where the comment is stripped of articles and is on one line

INPUT_DIRECTORY = lib.INSTALLATION_DIRECTORY + "\\src code\\allCSFiles" # the source code directory
OUTPUT_DIRECTORY = lib.INSTALLATION_DIRECTORY + "\\data\\AllComments"
READ_TEST_LIMIT = 0 # debug to only partially read the file, 0 will read all lines
read_test = 0

# place to store all comments
all_comments=[]

# get all the files in the src dir and loop though the directory
directory = os.fsencode(INPUT_DIRECTORY)
for file in os.listdir(directory):

    #DEBUG
    if READ_TEST_LIMIT != 0:
        if read_test > READ_TEST_LIMIT:
            break
        else:
            read_test = read_test + 1

    filePath = directory + b'\\' + file
    #open the file at filepath, and read all code in the file
    code = ""
    with open(filePath, encoding='utf-8-sig') as f:
        try:
            code = f.readlines()
            f.close()
        except:
            print("could not open file: ", file)
            print("consider adding the extension to the filter list in Parameters.")
            continue

    # skip if file is empty
    if len(code) == 0:
        continue

    # find comments and comment blocks
    # adjacent single comments on multiple lines are treated as one comment
    comment_prev_flag = True
    comment_block_flag = False
    comment_block_written = False
    comment = ""
    for line in code:
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
            all_comments.append(lib.cleanComment(comment))
        elif comment_block_flag:
            comment = comment + " " + line
        else:
            if comment != "":
                comment_prev_flag = False
                all_comments.append(lib.cleanComment(comment))
                comment = ""

    # if there are no comments, skip
    if len(all_comments) == 0:
        continue

# check to see if the processing occoured
print("Comments found:", len(all_comments))

# remove duplicate comments and scramble the list order
all_comments = list(set(all_comments))

print("Unique Comments found:", len(all_comments))

# write results to file (ovewrite mode)
output_filepath = OUTPUT_DIRECTORY + "\\" + datetime.today().strftime('%Y%m%d%H%M%S') + ".txt"
with open(output_filepath, "w", encoding='utf-8-sig') as f:
    for com in all_comments:
        # only take comments that have more than 4 words
        if len(com.split()) >= 4:
            f.write(com + "\n") # add back in the newline to seperate records
    f.close()
