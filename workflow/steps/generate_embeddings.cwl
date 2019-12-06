class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
baseCommand:
  - python3
inputs:
  - id: dataset
    type: string
  - id: training_file
    type: Directory
  - id: vectors_file
    type: string
  - id: working_directory
    type: string
outputs:
  - id: graph_output
    type: Directory
    outputBinding:
      glob: graph
  - id: model_output
    type: Directory
    outputBinding:
      glob: models
  - id: vector_output
    type: File
    outputBinding:
      glob: $(inputs.vectors_file)
  - id: walks_output
    type: Directory
    outputBinding:
      glob: walks
arguments:
  - $(inputs.working_directory)src/RDF2Vec.py
  - '-tr'
  - $(inputs.training_file.path)/train_positives.tsv
  - '-w'
  - walks
  - '-kg'
  - graph
  - '-m'
  - models
  - '-v'
  - $(inputs.vectors_file)
requirements:
  - class: InlineJavascriptRequirement
