class: CommandLineTool
cwlVersion: v1.0
$namespaces:
  sbg: 'https://www.sevenbridges.com/'
baseCommand:
  - python3
inputs:
  - id: inputFile
    type: string
  - id: outFolder
    type: string
  - id: negStrategy
    type: string
  - id: working_directory
    type: string
  - id: minNumRel
    type: int
  - id: fractionTest
    type: float?
  - id: numTestNegatives
    type: int?
  - id: numTrainNegatives
    type: int?
  - id: predicate
    type: string?
  - id: predict
    type: int
outputs:
  - id: examples_file_output
    type: Directory
    outputBinding:
      glob: $(inputs.outFolder)
arguments:
  - $(inputs.working_directory)src/datagen/DataGen.py
  - '--inF'
  - $(inputs.inputFile)
  - '--outF'
  - $(inputs.outFolder)
  - '--negStrategy'
  - $(inputs.negStrategy)
  - '--minNumRel'
  - $(inputs.minNumRel)
  - '--fractionTest'
  - $(inputs.fractionTest)
  - '--numTestNegatives'
  - $(inputs.numTestNegatives)
  - '--numTrainNegatives'
  - $(inputs.numTrainNegatives)
  - '--predicate'
  - $(inputs.predicate)
  - '--predict'
  - $(inputs.predict)
requirements:
  - class: InlineJavascriptRequirement
