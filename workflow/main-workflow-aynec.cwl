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
    'sbg:x': -33.420528411865234
    'sbg:y': -37.822296142578125
  - id: working_directory
    type: string
    'sbg:x': -51.987491607666016
    'sbg:y': -828.0864868164062
  - id: inputFile
    type: string
    'sbg:x': 60.01408386230469
    'sbg:y': 806.6358032226562
  - id: propNegatives
    type: float
    'sbg:x': -33.420528411865234
    'sbg:y': -601.5695190429688
  - id: negStrategy
    type: string
    'sbg:x': -22.280353546142578
    'sbg:y': -341.632080078125
  - id: outFolder
    type: string
    'sbg:x': -40.55130386352539
    'sbg:y': 353.65472412109375
  - id: fractionTest
    type: float?
    'sbg:x': 56.557437896728516
    'sbg:y': 982.427978515625
outputs:
  - id: graph_output
    outputSource:
      - step2-generate-embeddings/graph_output
    type: Directory
    'sbg:x': 1550.4691162109375
    'sbg:y': -114.77040100097656
  - id: model_output
    outputSource:
      - step2-generate-embeddings/model_output
    type: Directory
    'sbg:x': 1554.63232421875
    'sbg:y': 42.59519577026367
  - id: vector_output
    outputSource:
      - step2-generate-embeddings/vector_output
    type: File
    'sbg:x': 1513.765625
    'sbg:y': 293
  - id: walks_output
    outputSource:
      - step2-generate-embeddings/walks_output
    type: Directory
    'sbg:x': 1513.765625
    'sbg:y': 186
  - id: classifier_train_output
    outputSource:
      - step3-train-classifier/classifier_train_output
    type: File
    'sbg:x': 1926.9561767578125
    'sbg:y': 448.9598388671875
  - id: classifier_test_output
    outputSource:
      - step3-train-classifier/classifier_test_output
    type: File
    'sbg:x': 1912.19091796875
    'sbg:y': 756.921142578125
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
    'sbg:x': 1041.1361083984375
    'sbg:y': 179.1442413330078
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
    run: steps/train_classifier.cwl
    label: >-
      Fact validation library. A step to generate embeddings, Ammar Ammar
      <ammar257ammar@gmail.com>
    'sbg:x': 1322.24755859375
    'sbg:y': 797.0128173828125
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
      - id: fractionTest
        source: fractionTest
    out:
      - id: examples_file_output
    run: steps/generate_examples_aynec.cwl
    label: >-
      Fact validation library. A step to generate negative and positive
      examples, Ammar Ammar <ammar257ammar@gmail.com>
    'sbg:x': 665.6767578125
    'sbg:y': 821.9795532226562
requirements: []
