import os
from reader import Reader
from random import random
from rdflib import Graph
import multiprocessing
from tqdm import tqdm
class NTriplesReader(Reader):
	"""Reader of rdf graphs"""

	def __init__(self, file_path, prob, include_dataprop):
		"""
		Arguments:

		file_path -- the path to the single file containing the knwoledge graphs
		prob -- probability of keeping each triple when reading the graph.
		If 1.0, the entire graph is kept. If lesser than one, the final graph has reduced size.
		include_dataprop -- whether or not to store data properties
		"""

		self.file_path = file_path
		self.prob = prob
		self.include_dataprop = include_dataprop
		self.number_lines = 0

	def read(self):
		"""
		Reads the graph using the parameters specified in the constructor.
		Expects each line to contain a triple with the source first, then the relation, then the target, separated by white spaces.

		Returns: a tuple with:
		1: a dictionary with the entities as keys (their names) as degree information as values.
		Each value is a dictionary with the outwards degree ("out_degree key"), inwards degree ("in_degree key"), total degree ("degree" key), and the data properties ("data_properties" key).
		2: a set with the name of the relations in the graph
		3: a set with the edges in the graph. Each edge is a tuple with the name of the relation, the source entity, and the target entity.
		"""
		entities = dict()
		relations = set()
		edges = set()
		rdf_graph = Graph()
		parse_filepath = os.path.splitext(self.file_path)
		file_extension = parse_filepath[1].replace(".", "")
		
		# Following file extensions are accepted: ‘xml’, ‘n3’, ‘nt’, ‘trix’, ‘rdfa’
		print("Parsing RDF from file format: " + file_extension)
		# TODO: should work for any RDF format
		rdf_graph.parse(self.file_path, format=file_extension)
		for source, relation, target in rdf_graph:
			if(self.prob == 1.0 or random() < self.prob):
				is_dataprop = target.startswith('"')
				if source not in entities:
					entities[source] = dict(degree=0, out_degree=0, in_degree=0, data_properties={})
				entities[source]["out_degree"] += 1
				entities[source]["degree"] += 1
				if not is_dataprop:
					if target not in entities:
						entities[target] = dict(degree=0, out_degree=0, in_degree=0, data_properties={})
					entities[target]["in_degree"] += 1
					entities[target]["degree"] += 1
					relations.add(relation)
					edges.add((relation, source, target))
				else:
					if(self.include_dataprop):
						entities[source]["data_properties"][relation] = target

		return (entities, relations, edges)