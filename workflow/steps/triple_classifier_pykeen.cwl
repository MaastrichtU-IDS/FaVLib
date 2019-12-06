$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: triple_classifier_pykeen
label: triple_classifier_pykeen
class: CommandLineTool
cwlVersion: v1.0
baseCommand:
  - python3
inputs:
  - id: embedding_output_folder
    type: Directory
  - id: train_test_examples_dir
    type: Directory
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
  - $(inputs.working_directory)src/TrainingTripleFacts.py
  - '-train'
  - $(inputs.train_test_examples_dir.path)/train.txt
  - '-test'
  - $(inputs.train_test_examples_dir.path)/test.txt
  - '-emb'
  - $(inputs.embedding_output_folder.path)/entities_to_embeddings.json
  - '-relmap'
  - $(inputs.embedding_output_folder.path)/relation_to_id.json
  - '-otrain'
  - train_output.csv
  - '-otest'
  - test_output.csv
requirements:
  - class: InlineJavascriptRequirement
