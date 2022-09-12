import argparse

def args_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default="Demo",
                        help='Dataset')
        parser.add_argument('--input_type', type=str, default='image', choices=['image', 'adj'])
        parser.add_argument('--view', type=int, default=0,
                        help = 'view index in the dataset')

        parser.add_argument('--comms_round', type=int, default=5,
                                help='number of validation folds.')
        parser.add_argument('--hospital_num', type=int, default=3,
                                help='number of hospitals.')       

        parser.add_argument('--model', type=str, default="GCN", choices=["GCN", "DiffPool"]) 
        parser.add_argument('--num_epochs', type=int, default=100, #50
                                help='Training Epochs')
        parser.add_argument('--batch_size', type=int, default=1,
                                help='Training Batches')
        parser.add_argument('--cv_number', type=int, default=3,
                                help='number of validation folds.')
        parser.add_argument('--NormalizeInputGraphs', default=False, action='store_true',
                                help='Normalize Input adjacency matrices of graphs')  
        parser.add_argument('--evaluation_method', type=str, default='model assessment',
                                help='evaluation method, possible values : model selection, model assessment')
        parser.add_argument('--lr', type=float, default = 0.0001,
                        help='Initial learning rate.')
        parser.add_argument('--weight_decay', type=float, default = 0.00001,
                        help='Weight decay (L2 loss on parameters).')
        parser.add_argument('--threshold', dest='threshold', default='median',
                help='threshold the graph adjacency matrix. Possible values: no_threshold, median, mean')
        parser.add_argument('--dropout', dest='dropout', type=float, default=0.1,
                                help='Dropout rate.')
        parser.add_argument('--num-classes', dest='num_classes', type=int, default=2,
                                help='Number of label classes')

        parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=256,
                                help='Hidden dimension')
        parser.add_argument('--output-dim', dest='output_dim', type=int, default=512,
                                help='Output dimension')
        parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=3,
                                help='Number of graph convolution layers before each pooling')
        parser.add_argument('--assign-ratio', dest='assign_ratio', type=float, default=0.1,
                                help='ratio of number of nodes in consecutive layers')
        parser.add_argument('--num-pool', dest='num_pool', type=int, default=1,
                                help='number of pooling layers')
        parser.add_argument('--nobn', dest='bn', action='store_const',
                                const=False, default=True,
                                help='Whether batch normalization is used')
        parser.add_argument('--linkpred', dest='linkpred', action='store_const',
                                const=True, default=False,
                                help='Whether link prediction side objective is used')
        parser.add_argument('--nobias', dest='bias', action='store_const',
                                const=False, default=True,
                                help='Whether to add bias. Default to True.')
        parser.add_argument('--clip', dest='clip', type=float, default=2.0,
                help='Gradient clipping.')
        
        parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
        parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
        return parser.parse_args()