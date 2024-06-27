# Code Comment Classifier

## Introduction

This AI classifier enables insight into a coding project in order to identify parts of code that need attention. Often code projects can be dauntingly large and important comments get lost. Furthermore, some coders may include useful suggestions for future development.

Project tested with C# code.

## Installation

git clone https://github.com/AvidProlix/CodeCommentClassifier

## Usage

This app is broken up into a few different pieces that each execute separately and can use their own data. The steps are as follows: extract data from program files in order to later label them, training the model with the labeled data, applying the trained model to your target coding project.

### Data Set Creation

Define the input and output subdirectory
run the script to extract the comment/comment blocks into newlines

label the data in the format label,comment\n

### Training

Define the input and output subdirectories
Define the training hyperparameters
run the training on given labeled data

### Application

Define the input subdirectories
note that the model will need to load a paired/compatable model path file and vocab embed layer file
run the application to classify the files/comments in the input project directory
optionally write output of script to file to retain results
