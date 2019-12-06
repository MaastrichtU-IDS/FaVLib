
import argparse
import pykeen

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-tr',required=True, dest='rdf_file', help='enter folder path for rdf file')
    parser.add_argument('-o',required=True, dest='out_folder', help='enter folder path for embeddings to be stored')
    parser.add_argument('-seed', required=True, dest='random_seed', help='enter random seed')
    parser.add_argument('-emb_model', required=True, dest='embedding_model_name', help='enter kg embedding model name')
    parser.add_argument('-emb_dim', required=True, dest='embedding_dim', help='enter kg embedding dimension')
    parser.add_argument('-scoring_fun', required=True, dest='scoring_function', help='enter scoring function')
    parser.add_argument('-norm_of_entities', required=True, dest='norm_of_entities', help='enter normalization of entities')
    parser.add_argument('-margin_loss', required=True, dest='margin_loss', help='enter margin loss')
    parser.add_argument('-learning_rate', required=True, dest='learning_rate', help='enter learning rate')
    parser.add_argument('-num_epochs', required=True, dest='num_epochs', help='enter num_epochs')
    parser.add_argument('-batch_size', required=True, dest='batch_size', help='enter batch size')
    parser.add_argument('-preferred_device', required=True, dest='preferred_device', help='enter preferred_device')
    args = parser.parse_args()

    random_seed =args
    
    config = dict(
    training_set_path           = args.rdf_file,
    execution_mode              = 'Training_mode',
    random_seed                 = int(args.random_seed),
    kg_embedding_model_name     = args.embedding_model_name,
    embedding_dim               = int(args.embedding_dim),
    scoring_function            = int(args.scoring_function),  # corresponds to L1
    normalization_of_entities   = int(args.norm_of_entities),  # corresponds to L2
    margin_loss                 = float(args.margin_loss),
    learning_rate               = float(args.learning_rate),
    num_epochs                  = int(args.num_epochs),  
    batch_size                  = int(args.batch_size),
    filter_negative_triples     = True,
    preferred_device            = args.preferred_device
    )
    if config['kg_embedding_model_name'] in ['TransD','TransR']:
        config['relation_embedding_dim']=20

    if config['kg_embedding_model_name'] =='TransH':
        config['weighting_soft_constraint']=0.015625

    print (args.out_folder)
    print (config)


    results = pykeen.run(
    config=config,
    output_directory= args.out_folder,
    )
    print ("Embedding learning is complete")
