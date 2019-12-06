class: Workflow
cwlVersion: v1.0
id: main_workflow_pykeen
label: main-workflow-pykeen
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
inputs:
  - id: working_directory
    type: string
    'sbg:x': -893.7548217773438
    'sbg:y': 231
  - id: seed
    type: int
    'sbg:x': -767.6201171875
    'sbg:y': -34.768558502197266
  - id: embedding_model_name
    type: string
    'sbg:x': -736.803466796875
    'sbg:y': -256.7161560058594
  - id: norm_of_entities
    type: int?
    'sbg:x': -772.9112548828125
    'sbg:y': -389.29888916015625
  - id: margin_loss
    type: float
    'sbg:x': -682.38037109375
    'sbg:y': -524.2952270507812
  - id: learning_rate
    type: float
    'sbg:x': -690
    'sbg:y': -638.1082153320312
  - id: num_epochs
    type: int
    'sbg:x': -682
    'sbg:y': -742
  - id: preferred_device
    type: string
    'sbg:x': -680.5476684570312
    'sbg:y': -884.610107421875
  - id: outFolder
    type: string
    'sbg:x': -829.4150390625
    'sbg:y': 405.7232971191406
  - id: propNegatives
    type: float
    'sbg:x': -817.5718994140625
    'sbg:y': 544.5314331054688
  - id: negStrategy
    type: string
    'sbg:x': -843.4426879882812
    'sbg:y': 807.5977783203125
  - id: minNumRel
    type: int
    'sbg:x': -830.5955200195312
    'sbg:y': 940.5955200195312
  - id: fractionTest
    type: float?
    'sbg:x': -850.8707885742188
    'sbg:y': 1098.9224853515625
  - id: scoring_function
    type: int
    'sbg:x': -801.6873168945312
    'sbg:y': -149.20843505859375
  - id: embedding_dim
    type: int
    'sbg:x': -818.6644287109375
    'sbg:y': 69.4970932006836
  - id: inputFile
    type: string
    'sbg:x': -799.7382202148438
    'sbg:y': 672.4539184570312
  - id: batch_size
    type: int
    'sbg:x': -905.2664794921875
    'sbg:y': 2.09871768951416
outputs:
  - id: embedding_output_folder
    outputSource:
      - generate_embeddings_pykeen/output_folder
    type: Directory
    'sbg:x': 811.05615234375
    'sbg:y': 45.84831619262695
  - id: results
    outputSource:
      - triple_classifier_pykeen/results
    type: File?
    'sbg:x': 840.4976806640625
    'sbg:y': 444.3939514160156
  - id: classifier_train_output
    outputSource:
      - triple_classifier_pykeen/classifier_train_output
    type: File
    'sbg:x': 836.964111328125
    'sbg:y': 561.0017700195312
  - id: classifier_test_output
    outputSource:
      - triple_classifier_pykeen/classifier_test_output
    type: File
    'sbg:x': 861.6990966796875
    'sbg:y': 755.3480224609375
steps:
  - id: generate_embeddings_pykeen
    in:
      - id: training_file
        source: generate_examples_aynec/examples_file_output
      - id: working_directory
        source: working_directory
      - id: output_folder_name
        source: outFolder
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
    label: generate_embeddings_pykeen
    'sbg:x': 610
    'sbg:y': 115.02214813232422
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
        source: minNumRel
      - id: fractionTest
        source: fractionTest
    out:
      - id: examples_file_output
    run: steps/generate_examples_aynec.cwl
    'sbg:x': -164.25753784179688
    'sbg:y': 802.8246459960938
  - id: triple_classifier_pykeen
    in:
      - id: embedding_output_folder
        source: generate_embeddings_pykeen/output_folder
      - id: train_test_examples_dir
        source: generate_examples_aynec/examples_file_output
      - id: working_directory
        source: working_directory
    out:
      - id: classifier_test_output
      - id: classifier_train_output
      - id: results
    run: steps/triple_classifier_pykeen.cwl
    label: triple_classifier_pykeen
    'sbg:x': 506.9078674316406
    'sbg:y': 502.02581787109375
requirements: []
