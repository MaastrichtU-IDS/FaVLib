#!/usr/bin/env cwl-runner

cwlVersion: v1.0
class: CommandLineTool

label: Fact validation library. A step to generate negative and positive examples, Ammar Ammar <ammar257ammar@gmail.com> 

baseCommand: [java]

arguments: [ "-jar", "$(inputs.working_directory)ExampleGeneration/favlib-0.0.1-SNAPSHOT-jar-with-dependencies.jar", "-op", "$(inputs.operation)", "-o", "$(inputs.operation).nq", "-log", "stats-$(inputs.operation).log"]

inputs:
  
  working_directory:
    type: string
  operation:
    type: string
  sparql_endpoint:
    type: string
    inputBinding:
      position: 1
      prefix: -e
  graph_base_uri:
    type: string
    inputBinding:
      position: 2
      prefix: -gr
  predicate_uri_prefix:
    type: string
    inputBinding:
      position: 3
      prefix: -pfx
  predicate:
    type: string?
    default: all
    inputBinding:
      position: 4
      prefix: -pr
  limit:
    type: string?
    default: 100
    inputBinding:
      position: 5
      prefix: -l

outputs:
  
  examples_file_output:
    type: File
    outputBinding:
      glob: $(inputs.operation).nq

  examples_stats_output:
    type: File
    outputBinding:
      glob: stats-$(inputs.operation).log
