###########################################################################
# Gabriel Matos Leite, PhD candidate (email: gmatos@cos.ufrj.br)
# March 30, 2023
###########################################################################



from neat.graphs import feed_forward_layers
from neat.activations_derivative import ActivationDerivativesFunctionSet
from collections import OrderedDict

class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)
        self.activation_derivatives = ActivationDerivativesFunctionSet()
        self.delta_errors = None

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)

        return [self.values[i] for i in self.output_nodes]

    def backward_td(self, next_q, current_q, genome, action, learning_rate=0.01):
        #must be called after activate call
        self.delta_errors =  dict((key, 0.0) for key in self.values.keys())
        
        activation_derivative = self.activation_derivatives.get(genome.nodes[action].activation)
        tmp_e = (next_q - current_q) * activation_derivative(self.values[action])
        self.delta_errors[action] = tmp_e
        
        previous_layer = OrderedDict.fromkeys(self.output_nodes)

        # calculate error on nodes except output ones
        while len(previous_layer) > 0:
            node, _ = previous_layer.popitem(last=False)
            if node in self.node_evals:
                _, _, _, _, links = self.node_evals[node]
                for i, w in links:
                    if i not in self.input_nodes and i in self.values:
                        activation_derivative = self.activation_derivatives.get(genome.nodes[i].activation)
                        tmp_e = (w * self.delta_errors[node]) * activation_derivative(self.values[i])
                        self.delta_errors[i] += tmp_e
                        previous_layer[i] = None # include seen node in processing queue
        
        # update weights using error calculated
        for (i_node, o_node), cg in genome.connections.items():
            if i_node in self.values and o_node in self.delta_errors:
                cg.weight -= learning_rate * self.delta_errors[o_node] * self.values[i_node]
                genome.nodes[o_node].bias -= learning_rate * self.delta_errors[o_node]
        
        return genome

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)