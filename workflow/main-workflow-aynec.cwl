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
    'sbg:x': 106.72399139404297
    'sbg:y': -153.3798370361328
  - id: working_directory
    type: string
    'sbg:x': -57.34417724609375
    'sbg:y': -52.35772705078125
  - id: inputFile
    type: string
    'sbg:x': -145.90115356445312
    'sbg:y': 586.761474609375
  - id: propNegatives
    type: float
    'sbg:x': -166.3134765625
    'sbg:y': 193.26405334472656
  - id: negStrategy
    type: string
    'sbg:x': -154.41909790039062
    'sbg:y': 326.2453308105469
  - id: outFolder
    type: string
    'sbg:x': -177.0799102783203
    'sbg:y': 61.81605911254883
  - id: fractionTest
    type: float?
    'sbg:x': -137.7955322265625
    'sbg:y': 715.37646484375
  - id: minNumRel
    type: int
    'sbg:x': -142.43443298339844
    'sbg:y': 459.0340881347656
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
    'sbg:x': 858.1845092773438
    'sbg:y': 271.5503234863281
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
    'sbg:x': 330.7955322265625
    'sbg:y': 282.17376708984375
requirements: []
