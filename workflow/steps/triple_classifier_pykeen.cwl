class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: triple_classifier_pykeen
baseCommand:
  - python3
inputs:
  - id: embedding_output_folder
    type: Directory
  - id: train_test_examples_dir
    type: Directory
  - id: working_directory
    type: string
  - id: output_train
    type: int
  - id: output_test
    type: int
  - id: predict
    type: int
outputs:
  - id: output_folder
    type: Directory
    outputBinding:
      glob: clsout
label: triple_classifier_pykeen
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
  - $(inputs.output_train)
  - '-otest'
  - $(inputs.output_test)
  - '-predict'
  - $(inputs.predict)
requirements:
  - class: InlineJavascriptRequirement
