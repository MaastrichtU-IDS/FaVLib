class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
baseCommand:
  - python
inputs:
  - id: graph_folder
    type: Directory
  - id: train_examples_dir
    type: Directory
  - id: vectors_file
    type: File
  - id: working_directory
    type: string
outputs:
  - id: classifier_test_output
    type: File
    outputBinding:
      glob: test_output.csv
  - id: classifier_train_output
    type: File
    outputBinding:
      glob: train_output.csv
  - id: results
    type: File?
    outputBinding:
      glob: results.csv
arguments:
  - $(inputs.working_directory)src/TrainingFactModel.py
  - '-train'
  - $(inputs.train_examples_dir.path)/train.txt
  - '-test'
  - $(inputs.train_examples_dir.path)/test.txt
  - '-emb'
  - $(inputs.vectors_file)
  - '-relmap'
  - $(inputs.graph_folder)
  - '-otrain'
  - train_output.csv
  - '-otest'
  - test_output.csv
requirements:
  - class: InlineJavascriptRequirement
