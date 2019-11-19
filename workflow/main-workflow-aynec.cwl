class: Workflow
cwlVersion: v1.0
label: >-
  Fact validation pipeline general workflow, Ammar Ammar
  <ammar257ammar@gmail.com>
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
inputs:
  - id: vectors_file
    type: string
    'sbg:x': 0
    'sbg:y': 106.71875
  - id: working_directory
    type: string
    'sbg:x': -57.34417724609375
    'sbg:y': -52.35772705078125
  - id: inputFile
    type: string
    'sbg:x': 0
    'sbg:y': 426.90625
  - id: propNegatives
    type: float
    'sbg:x': -219.9027557373047
    'sbg:y': 173.48622131347656
  - id: negStrategy
    type: string
    'sbg:x': -199.2707977294922
    'sbg:y': 333.7439270019531
  - id: outFolder
    type: string
    'sbg:x': 443.9027099609375
    'sbg:y': 255.72747802734375
  - id: fractionTest
    type: float?
    'sbg:x': 0
    'sbg:y': 533.59375
  - id: minNumRel
    type: int
    'sbg:x': -237.071044921875
    'sbg:y': 482.778564453125
outputs:
  - id: graph_output
    outputSource:
      - step2-generate-embeddings/graph_output
    type: Directory
    'sbg:x': 1536.3916015625
    'sbg:y': 703.1699829101562
  - id: model_output
    outputSource:
      - step2-generate-embeddings/model_output
    type: Directory
    'sbg:x': 1496.5
    'sbg:y': 394.5
  - id: vector_output
    outputSource:
      - step2-generate-embeddings/vector_output
    type: File
    'sbg:x': 1496.5
    'sbg:y': 139.0625
  - id: walks_output
    outputSource:
      - step2-generate-embeddings/walks_output
    type: Directory
    'sbg:x': 1496.5
    'sbg:y': 32.375
  - id: classifier_train_output
    outputSource:
      - step3-train-classifier/classifier_train_output
    type: File
    'sbg:x': 1513.75048828125
    'sbg:y': 273.30615234375
  - id: classifier_test_output
    outputSource:
      - step3-train-classifier/classifier_test_output
    type: File
    'sbg:x': 1511.2572021484375
    'sbg:y': 547.0397338867188
  - id: examples_file_output
    outputSource:
      - generate_examples_aynec/examples_file_output
    type: Directory
    'sbg:x': 1484.813232421875
    'sbg:y': 897.3413696289062
  - id: results
    outputSource:
      - step3-train-classifier/results
    type: File?
    'sbg:x': 1488.8759765625
    'sbg:y': -94.5193862915039
steps:
  - id: step2-generate-embeddings
    in:
      - id: dataset
        source: outFolder
      - id: training_file
        source: generate_examples_aynec/examples_file_output
      - id: vectors_file
        source: vectors_file
      - id: working_directory
        source: working_directory
    out:
      - id: graph_output
      - id: model_output
      - id: vector_output
      - id: walks_output
    run: steps/generate_embeddings.cwl
    label: >-
      Fact validation library. A step to generate embeddings, Ammar Ammar
      <ammar257ammar@gmail.com>
    'sbg:x': 559.24951171875
    'sbg:y': -35.95317840576172
  - id: step3-train-classifier
    in:
      - id: graph_folder
        source: step2-generate-embeddings/graph_output
      - id: train_examples_dir
        source: generate_examples_aynec/examples_file_output
      - id: vectors_file
        source: step2-generate-embeddings/vector_output
      - id: working_directory
        source: working_directory
    out:
      - id: classifier_test_output
      - id: classifier_train_output
      - id: results
    run: steps/train_classifier.cwl
    label: >-
      Fact validation library. A step to generate embeddings, Ammar Ammar
      <ammar257ammar@gmail.com>
    'sbg:x': 751.0256958007812
    'sbg:y': 299.19317626953125
  - id: generate_examples_aynec
    in:
      - id: inputFile
        source: inputFile
      - id: outFolder
        source: outFolder
      - id: propNegatives
        source: propNegatives
      - id: negStrategy
        source: negStrategy
      - id: working_directory
        source: working_directory
      - id: minNumRel
        default: 50
        source: minNumRel
      - id: fractionTest
        source: fractionTest
    out:
      - id: examples_file_output
    run: steps/generate_examples_aynec.cwl
    label: >-
      Fact validation library. A step to generate negative and positive
      examples, Ammar Ammar <ammar257ammar@gmail.com>
    'sbg:x': 193.58506774902344
    'sbg:y': 320.15625
requirements: []
