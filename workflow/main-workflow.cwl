#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: Workflow

label: Fact validation pipeline general workflow, Ammar Ammar <ammar257ammar@gmail.com> 


inputs: 
  
  working_directory: string 
  dataset: string
  vectors_file: string
  test_set: string
  operation_positive: string
  operation_negative: string
  sparql_endpoint: string
  graph_base_uri: string
  predicate_uri_prefix: string
  predicate: string
  limit: string

outputs:
  
  examples_file_output_pos:
    type: File
    outputSource: step1_1-generate-positive-examples/examples_file_output
  examples_stats_output_pos:
    type: File
    outputSource: step1_1-generate-positive-examples/examples_stats_output
  examples_file_output_neg:
    type: File
    outputSource: step1_2-generate-negative-examples/examples_file_output
  examples_stats_output_neg:
    type: File
    outputSource: step1_2-generate-negative-examples/examples_stats_output
  walks_output:
    type: Directory
    outputSource: step2-generate-embeddings/walks_output
  graph_output:
    type: Directory
    outputSource: step2-generate-embeddings/graph_output
  model_output:
    type: Directory
    outputSource: step2-generate-embeddings/model_output
  vector_output:
    type: File
    outputSource: step2-generate-embeddings/vector_output

  classifier_output:
    type: File
    outputSource: step3-train-classifier/classifier_output
  
steps:

  step1_1-generate-positive-examples:
    run: steps/generate_examples.cwl
    in:
      working_directory: working_directory
      operation: operation_positive
      sparql_endpoint: sparql_endpoint
      graph_base_uri: graph_base_uri
      predicate_uri_prefix: predicate_uri_prefix
      predicate: predicate
      limit: limit
    out: [examples_file_output, examples_stats_output]

  step1_2-generate-negative-examples:
    run: steps/generate_examples.cwl
    in:
      working_directory: working_directory
      operation: operation_negative
      sparql_endpoint: sparql_endpoint
      graph_base_uri: graph_base_uri
      predicate_uri_prefix: predicate_uri_prefix
      predicate: predicate
      limit: limit
    out: [examples_file_output, examples_stats_output]

  step2-generate-embeddings:
    run: steps/generate_embeddings.cwl
    in:
      working_directory: working_directory
      dataset: dataset
      training_file: step1_1-generate-positive-examples/examples_file_output
      vectors_file: vectors_file

    out: [walks_output, graph_output, model_output, vector_output]

  step3-train-classifier:
    run: steps/train_classifier.cwl
    in:
      working_directory: working_directory
      positive_examples: step1_1-generate-positive-examples/examples_file_output
      negative_examples: step1_2-generate-negative-examples/examples_file_output
      vectors_file: step2-generate-embeddings/vector_output
      graph_folder: step2-generate-embeddings/graph_output 
      test_set: test_set
    out: [classifier_output]

