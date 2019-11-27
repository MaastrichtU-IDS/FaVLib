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
  - default: all
    id: propNegatives
    type: float
  - id: negStrategy
    type: string
  - id: working_directory
    type: string
  - id: minNumRel
    type: int
  - id: fractionTest
    type: float?
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
  - '--numNegatives'
  - $(inputs.propNegatives)
  - '--negStrategy'
  - $(inputs.negStrategy)
  - '--minNumRel'
  - $(inputs.minNumRel)
  - '--fractionTest'
  - $(inputs.fractionTest)
requirements:
  - class: InlineJavascriptRequirement
