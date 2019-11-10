#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool

label: Fact validation library. A step to generate embeddings, Ammar Ammar <ammar257ammar@gmail.com> 

baseCommand: [python]

arguments: [ "$(inputs.working_directory)src/TrainingFactModel.py",
"-train", "$(inputs.train_examples_dir.path)/train.txt",  "-test", "$(inputs.train_examples_dir.path)/test.txt",
"-emb", "$(inputs.vectors_file)", "-relmap", "$(inputs.graph_folder)", "-otrain", "train_output.csv", "-otest", "test_output.csv"]

inputs:
  
  working_directory:
    type: string
  train_examples_dir:
    type: Directory
  vectors_file:
    type: File
  graph_folder:
     type: Directory

outputs:
  
  classifier_train_output:
    type: File
    outputBinding:
      glob: train_output.csv
  classifier_test_output:
    type: File
    outputBinding:
      glob: test_output.csv

