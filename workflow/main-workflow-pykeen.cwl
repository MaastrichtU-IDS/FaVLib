class: Workflow
cwlVersion: v1.0
id: main_workflow_pykeen
label: main-workflow-pykeen
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
inputs:
  - id: working_directory
    type: string
    'sbg:x': 0
    'sbg:y': 0
  - id: seed
    type: int
    'sbg:x': 0
    'sbg:y': 106.78125
  - id: embedding_model_name
    type: string
    'sbg:x': 260.19049072265625
    'sbg:y': 1347.15869140625
  - id: norm_of_entities
    type: int?
    'sbg:x': 0
    'sbg:y': 747.46875
  - id: margin_loss
    type: float
    'sbg:x': 0
    'sbg:y': 1067.8125
  - id: learning_rate
    type: float
    'sbg:x': 0
    'sbg:y': 1174.59375
  - id: num_epochs
    type: int
    'sbg:x': 0
    'sbg:y': 640.6875
  - id: preferred_device
    type: string
    'sbg:x': 0
    'sbg:y': 427.125
  - id: negStrategy
    type: string
    'sbg:x': 0
    'sbg:y': 854.25
  - id: minNumRel
    type: int
    'sbg:x': 0
    'sbg:y': 961.03125
  - id: fractionTest
    type: float?
    'sbg:x': 0
    'sbg:y': 1388.15625
  - id: scoring_function
    type: int
    'sbg:x': 0
    'sbg:y': 213.5625
  - id: embedding_dim
    type: int
    'sbg:x': 696.8253784179688
    'sbg:y': 1375.8492431640625
  - id: inputFile
    type: string
    'sbg:x': 0
    'sbg:y': 1281.375
  - id: batch_size
    type: int
    'sbg:x': 469.015869140625
    'sbg:y': 1349.9603271484375
  - id: predicate
    type: string?
    'sbg:x': -442.39019775390625
    'sbg:y': 151.89405822753906
  - id: predict
    type: int
    'sbg:x': -454.02911376953125
    'sbg:y': 4.300266742706299
  - id: numTestNegatives
    type: int?
    'sbg:x': -450.89825439453125
    'sbg:y': 396.4483337402344
  - id: numTrainNegatives
    type: int?
    'sbg:x': -555.7281494140625
    'sbg:y': 253.3624725341797
  - id: output_test
    type: int
    'sbg:x': -326.12030029296875
    'sbg:y': 791.013916015625
  - id: output_train
    type: int
    'sbg:x': -378.38824462890625
    'sbg:y': 1007.3016967773438
outputs:
  - id: embedding_output_folder
    outputSource:
      - generate_embeddings_pykeen/output_folder
    type: Directory
    'sbg:x': 1255.015869140625
    'sbg:y': 1100.579345703125
  - id: examples_file_output
    outputSource:
      - generate_examples_aynec/examples_file_output
    type: Directory
    'sbg:x': 1333.359375
    'sbg:y': 110.99861907958984
  - id: output_folder
    outputSource:
      - triple_classifier_pykeen/output_folder
    type: Directory
    'sbg:x': 1420.779052734375
    'sbg:y': 662.6502685546875
steps:
  - id: generate_embeddings_pykeen
    in:
      - id: training_file
        source: generate_examples_aynec/examples_file_output
      - id: working_directory
        source: working_directory
      - id: output_folder_name
        default: embedding
      - id: seed
        source: seed
      - id: embedding_model_name
        source: embedding_model_name
      - id: embedding_dim
        source: embedding_dim
      - id: scoring_function
        source: scoring_function
      - id: norm_of_entities
        default: 2
        source: norm_of_entities
      - id: margin_loss
        source: margin_loss
      - id: learning_rate
        source: learning_rate
      - id: num_epochs
        source: num_epochs
      - id: preferred_device
        default: cpu
        source: preferred_device
      - id: batch_size
        source: batch_size
    out:
      - id: output_folder
    run: steps/generate_embeddings_pykeen.cwl
    label: Embedding Learning
    'sbg:x': 676.2063598632812
    'sbg:y': 1029.111083984375
  - id: generate_examples_aynec
    in:
      - id: inputFile
        source: inputFile
      - id: outFolder
        default: datagen
      - id: negStrategy
        source: negStrategy
      - id: working_directory
        source: working_directory
      - id: minNumRel
        source: minNumRel
      - id: fractionTest
        source: fractionTest
      - id: numTestNegatives
        source: numTestNegatives
      - id: numTrainNegatives
        source: numTrainNegatives
      - id: predicate
        source: predicate
      - id: predict
        source: predict
    out:
      - id: examples_file_output
    run: steps/generate_examples_aynec.cwl
    label: Data Generation
    'sbg:x': 580.547607421875
    'sbg:y': 82.84127044677734
  - id: triple_classifier_pykeen
    in:
      - id: embedding_output_folder
        source: generate_embeddings_pykeen/output_folder
      - id: train_test_examples_dir
        source: generate_examples_aynec/examples_file_output
      - id: working_directory
        source: working_directory
      - id: output_train
        source: output_train
      - id: output_test
        source: output_test
      - id: predict
        source: predict
    out:
      - id: output_folder
    run: steps/triple_classifier_pykeen.cwl
    label: triple_classifier_pykeen
    'sbg:x': 1045.263427734375
    'sbg:y': 676.910888671875
requirements: []
