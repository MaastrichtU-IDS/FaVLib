class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
id: generate_embeddings_pykeen
baseCommand:
  - python3
inputs:
  - id: training_file
    type: Directory
  - id: working_directory
    type: string
  - id: output_folder_name
    type: string
  - id: seed
    type: int
  - id: embedding_model_name
    type: string
  - id: embedding_dim
    type: int
  - id: scoring_function
    type: int
  - 'sbg:toolDefaultValue': '2'
    id: norm_of_entities
    type: int?
  - id: margin_loss
    type: float
  - id: learning_rate
    type: float
  - id: num_epochs
    type: int
  - id: preferred_device
    type: string
  - id: batch_size
    type: int
outputs:
  - id: output_folder
    type: Directory
    outputBinding:
      glob: $(inputs.output_folder_name)
label: generate_embeddings_pykeen
arguments:
  - $(inputs.working_directory)src/PyKEEN.py
  - '-tr'
  - $(inputs.training_file.path)/train_positives.tsv
  - '-o'
  - $(inputs.output_folder_name)
  - '-seed'
  - $(inputs.seed)
  - '-emb_model'
  - $(inputs.embedding_model_name)
  - '-emb_dim'
  - $(inputs.embedding_dim)
  - '-scoring_fun'
  - $(inputs.scoring_function)
  - '-norm_of_entities'
  - $(inputs.norm_of_entities)
  - '-margin_loss'
  - $(inputs.margin_loss)
  - '-learning_rate'
  - $(inputs.learning_rate)
  - '-num_epochs'
  - $(inputs.num_epochs)
  - '-batch_size'
  - $(inputs.batch_size)
  - '-preferred_device'
  - $(inputs.preferred_device)
requirements:
  - class: InlineJavascriptRequirement
