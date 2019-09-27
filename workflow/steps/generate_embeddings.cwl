#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool

label: Fact validation library. A step to generate embeddings, Ammar Ammar <ammar257ammar@gmail.com> 

baseCommand: [python]

arguments: [ "$(inputs.working_directory)src/RDF2Vec.py" , "-tr", "$(inputs.training_file)",  "-w", "walks", "-kg", "graph", "-m", "models", "-v", "$(inputs.vectors_file)"]

inputs:
  
  working_directory:
    type: string
  dataset:
    type: string
  training_file:
    type: File
  vectors_file:
    type: string

outputs:
  
  walks_output:
    type: Directory
    outputBinding:
      glob: walks

  graph_output:
    type: Directory
    outputBinding:
      glob: graph

  model_output:
    type: Directory
    outputBinding:
      glob: models

  vector_output:
    type: File
    outputBinding:
      glob: $(inputs.vectors_file)

