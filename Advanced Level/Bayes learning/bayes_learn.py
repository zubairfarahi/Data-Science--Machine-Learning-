from typing import List, Dict, Tuple, Any, Iterator
from networkx.algorithms.tree.mst import prim_mst_edges

from numpy.core.fromnumeric import var
from bayes_net import BayesNet, BayesNode
from itertools import product
import numpy as np
from argparse import ArgumentParser, Namespace


def all_dicts(variables: List[str]) -> Iterator[Dict[str, int]]:
    for keys in product(*([[0, 1]] * len(variables))):
        yield dict(zip(variables, keys))


def cross_entropy(bn1: BayesNet, bn2: BayesNet, nsamples: int = None) -> float:
    cross_ent = .0
    if nsamples is None:
        bn1_vars = bn1.nodes.keys()
        for sample in all_dicts(bn1_vars):
            cross_ent -= np.exp(bn1.sample_log_prob(sample)) * bn2.sample_log_prob(sample)
    else:
        for _ in range(nsamples):
            cross_ent -= bn2.sample_log_prob(bn1.sample())
        cross_ent /= nsamples
    return cross_ent


def read_samples(file_name: str) -> List[Dict[str, int]]:
    samples = []
    with open(file_name, "r") as handler:
        lines = handler.readlines()
        # read first line of file to get variables in order
        variables = [str(v) for v in lines[0].split()]

        for i in range(1, len(lines)):
            vals = [int(v) for v in lines[i].split()]
            sample = dict(zip(variables, vals))
            samples.append(sample)

    return samples

from collections import defaultdict

class MLEBayesNet(BayesNet):
    """
    Placeholder class for the Bayesian Network that will learn CPDs using the frequentist MLE
    """
    def __init__(self, bn_file: str="data/bnet"):
        super(MLEBayesNet, self).__init__(bn_file=bn_file)
        self._reset_cpds()

    def _reset_cpds(self) -> None:
        """
        Reset the conditional probability distributions to a value of 0.5
        The values have to be **learned** from the samples.
        """
        for node_name in self.nodes:
            self.nodes[node_name].cpd["prob"] = 0.5

    def learn_cpds(self, samples: List[Dict[str, int]], alpha: float = 1.0) -> None:
        cpd = {}
        var_names = []
        
        for var_name in self.nodes:
            node = self.nodes[var_name]
            parent_nodes = [n.var_name for n in node.parent_nodes]
            var_nodes = [node.var_name]
            var_names = parent_nodes + var_nodes
            max = ",".join(var_names)
            #print(max)
            cpd[max] = defaultdict(lambda: 0)
            for sample in samples:
                samp = tuple([sample[var_name] for var_name in var_names])
                cpd[max][samp] += 1
            #print(cpd)
            l2 = []
            #print(node.cpd)
            for idx in node.cpd.index:
                l2 = list(idx)
                l2[0] = 1 - l2[0]
                l2 = tuple(l2)
                #print(cpd[max][idx])
                cpd_value = (cpd[max][idx] + alpha) / (cpd[max][idx] + cpd[max][l2] + 2 * alpha)
                node.cpd[tuple(idx)] = cpd_value
                #print(node.cpd[tuple(idx)])
            
           # lst = [idx for idx in node.cpd.index]
            #print(lst)
     
    


class ParametricBayesNet(BayesNet):
    """
    Placeholder class for the Bayesian Network that will learn CPDs using the frequentist MLE
    """
    def __init__(self, bn_file: str="data/bnet"):
        super(ParametricBayesNet, self).__init__(bn_file=bn_file)
        self._random_cpds()

    def _random_cpds(self):
        """
        Reset the conditional probability distributions to a random value between 0 and 1 that will be passed as the
        parameter to the sigmoid function
        """
        for node_name in self.nodes:
            self.nodes[node_name].cpd["prob"] = np.random.random_sample(self.nodes[node_name].cpd.shape[0])

    def prob(self, var_name: str, parent_values: List[int] = None) -> float:
        if parent_values is None:
            parent_values = []
        index_line = tuple([1] + parent_values)
        score = self.nodes[var_name].cpd.loc[index_line]["prob"]

        return 1. / (1. + np.exp(-score))

    def learn(self, sample: Dict[str, int],
              learning_rate: float = 1e-3) -> None:

        for var_name in self.nodes:
            parent_names = [p.var_name for p in self.nodes[var_name].parent_nodes]
            parent_vals = [sample[p] for p in parent_names]
            index_line = tuple([1] + parent_vals)
            score = self.nodes[var_name].cpd.loc[index_line]["prob"]

            # TODO 2: YOUR CODE HERE
            # compute the gradient with respect to the parameters modeling the CPD of variable var_name
            # grad = ...
            grad = sample[var_name] - 1/(1 + np.exp(-score))
            # update the parameters
            # self.nodes[var_name].cpd.loc[index_line]["prob"] = ...
            self.nodes[var_name].cpd.loc[index_line]["prob"] = score + learning_rate * grad

    def pretty_print(self):
        res = "Bayesian Network:\n"
        for var_name in self.nodes:
            res += str(self.nodes[var_name]) + "\n"
            cpd = self.nodes[var_name].cpd.copy()
            cpd["true_prob"] = 1. / (1. + np.exp(-cpd["prob"]))
            cpd.loc[0, "true_prob"] = 1 - cpd.loc[1, "true_prob"]

            res += str(cpd)
            res += "\n"

        return res


def get_args() -> Namespace:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-f", "--file",
                            type=str,
                            default="data/bn_learning",
                            dest="file_name",
                            help="Input file")
    arg_parser.add_argument("-s", "--samplefile",
                            type=str,
                            default="data/samples_bn_learning",
                            dest="samples_file_name",
                            help="Samples file")
    arg_parser.add_argument("-n", "--nsteps",
                            type=int,
                            default=10000,
                            dest="nsteps",
                            help="Number of optimization steps")
    arg_parser.add_argument("--lr",
                            type=float,
                            default=.005,
                            dest="lr",
                            help="Learning rate")

    return arg_parser.parse_args()


def main():
    args = get_args()
    table_bn = BayesNet(bn_file=args.file_name)
    mle_bn = MLEBayesNet(bn_file=args.file_name)
    parametric_bn = ParametricBayesNet(bn_file=args.file_name)

    print("Initial params MLE bn:")
    mle_bn.pretty_print()
    #print(mle_bn)
    print("Initial params parametric bn:")
    parametric_bn.pretty_print()
    #print(parametric_bn)
    print("========== Frequentist MLE ==========")
    samples = read_samples(args.samples_file_name)
    mle_bn.learn_cpds(samples)
    #print(mle_bn)
    print("Reference BN")
    table_bn.pretty_print()

    print("MLE BayesNet after learning CPDs")
    mle_bn.pretty_print()

    print("========== Parametric MLE ==========")


    ref_cent = cross_entropy(table_bn, table_bn)
    # cent = cross_entropy(table_bn, parametric_bn, nsamples=100)
    # print("Step %6d | CE: %6.3f / %6.3f" % (0, cent, ref_cent))

    for step in range(1, args.nsteps + 1):
        sample = table_bn.sample()
        parametric_bn.learn(sample, learning_rate=args.lr)

        if step % 500 == 0:
            cent = cross_entropy(table_bn, parametric_bn, nsampl es=200)
            # print(f"Step {step:6d} | CE: {cent:6.3f} / {ref_cent:6.3f}")
            print("Step %6d | CE: %6.3f / %6.3f" % (step, cent, ref_cent))

    print("Reference BN")
   # table_bn.pretty_print()

    print("Parametric BayesNet after learning CPDs")
   # parametric_bn.pretty_print()


if __name__ == "__main__":
    main()
