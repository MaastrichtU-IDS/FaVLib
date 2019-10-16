#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool

label: Fact validation library. A step to generate embeddings, Ammar Ammar <ammar257ammar@gmail.com> 

baseCommand: [python]

arguments: [ "$(inputs.working_directory)src/TrainingFactModel.py" , "-neg", "$(inputs.negative_examples)", 
"-pos", "$(inputs.positive_examples)", 
"-emb", "$(inputs.vectors_file)", "-relmap", "$(inputs.graph_folder)", "-o", "output.txt", "-test", "$(inputs.test_set)"]

inputs:
  
  working_directory:
    type: string
  positive_examples:
    type: File
  negative_examples:
    type: File
  vectors_file:
    type: File
  graph_folder:
     type: Directory
  test_set: 
     type: string

outputs:
  
  classifier_output:
    type: File
    outputBinding:
      glob: output.txt

