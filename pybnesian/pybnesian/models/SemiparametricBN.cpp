#include <models/SemiparametricBN.hpp>
#include <learning/scores/scores.hpp>
#include <learning/scores/cv_likelihood.hpp>
#include <learning/operators/operators.hpp>

using graph::DirectedGraph;
using learning::operators::Operator, learning::operators::AddArc, learning::operators::ChangeNodeType,
    learning::operators::OperatorPool, learning::operators::OperatorSet, learning::operators::ChangeNodeTypeSet,
    learning::operators::ArcOperatorSet;
using learning::scores::Score;
namespace models {

std::shared_ptr<BayesianNetworkBase> SemiparametricBNType::new_bn(const std::vector<std::string>& nodes) const {
    return std::make_shared<SemiparametricBN>(nodes);
}

std::shared_ptr<ConditionalBayesianNetworkBase> SemiparametricBNType::new_cbn(
    const std::vector<std::string>& nodes, const std::vector<std::string>& interface_nodes) const {
    return std::make_shared<ConditionalSemiparametricBN>(nodes, interface_nodes);
}

template <bool recursive>
void select_best_node_type(OperatorPool& op_pool,
                           const Score& score,
                           std::shared_ptr<SemiparametricBN>& spbn,
                           const std::string node_name,
                           const std::vector<std::string>& explored) {
    int node_idx = spbn->collapsed_index(node_name);

    auto type_op = std::dynamic_pointer_cast<ChangeNodeTypeSet>(op_pool.get_op_sets()[1]);
    auto type_change_deltas = type_op->get_delta();

    if (type_change_deltas[node_idx].rows() > 0) {
        int local_max;
        type_change_deltas[node_idx].maxCoeff(&local_max);
        if (type_change_deltas[node_idx](local_max) > util::machine_tol) {
            double delta_op = type_change_deltas[node_idx](local_max);
            auto alt_node_types = spbn->type()->alternative_node_type(*spbn, node_name);
            auto op_type_change = ChangeNodeType(node_name, alt_node_types[local_max], delta_op);
            op_type_change.apply(*spbn);
            auto nodes_changed = op_type_change.nodes_changed(*spbn);
            op_pool.update_scores(*spbn, score, nodes_changed);
            if (recursive) {
                for (const auto& child_name : spbn->children(node_name)) {
                    select_best_node_type<recursive>(op_pool, score, spbn, child_name, explored);
                }
            }
        }
    }
}

const std::string select_best_arc_orientation(OperatorPool& op_pool,
                                 const Score& score,
                                 std::shared_ptr<SemiparametricBN>& spbn,
                                 const std::string node1,
                                 const std::string node2) {
    int node1_idx = spbn->collapsed_index(node1);
    int node2_idx = spbn->collapsed_index(node2);

    auto arc_op = std::dynamic_pointer_cast<ArcOperatorSet>(op_pool.get_op_sets()[0]);
    auto arc_deltas = arc_op->get_delta();
    double arc1_delta = arc_deltas(node1_idx, node2_idx);
    double arc2_delta = arc_deltas(node2_idx, node1_idx);
    auto source = node1;
    auto dest = node2;
    if ((arc1_delta < arc2_delta && spbn->graph().can_add_arc(dest, source)) || !spbn->graph().can_add_arc(source, dest)) {
        std::swap(source, dest);
        std::swap(arc1_delta, arc2_delta);
    }

    auto op_arc = AddArc(source, dest, arc1_delta);
    op_arc.apply(*spbn);
    auto nodes_changed = op_arc.nodes_changed(*spbn);
    op_pool.update_scores(*spbn, score, nodes_changed);
    return dest;
}
std::shared_ptr<BayesianNetworkBase> SemiparametricBNType::spbn_from_pdag(const DataFrame& df,
                                                                          PartiallyDirectedGraph& pdag,
                                                                          const ArcStringVector& arc_blacklist) const {
    auto spbn = std::make_shared<SemiparametricBN>(pdag.nodes());
    spbn->set_unknown_node_types(df, FactorTypeVector());
    DirectedGraph directed(pdag.nodes());
    ArcSet arc_blackset;

    for (const auto& arc : arc_blacklist) {
        if (!pdag.contains_node(arc.first))
            throw std::invalid_argument("Node " + arc.first + " not present in the graph.");

        if (!pdag.contains_node(arc.second))
            throw std::invalid_argument("Node " + arc.second + " not present in the graph.");

        auto s = pdag.index(arc.first);
        auto t = pdag.index(arc.second);

        if (pdag.has_edge_unsafe(s, t)) {
            pdag.direct(t, s);
        }
        arc_blackset.insert({s, t});
    }

    for (const auto& arc : pdag.arcs()) {
        directed.add_arc_unsafe(directed.index(arc.first), directed.index(arc.second));
    }

    std::vector<int> incoming_arcs(pdag.num_nodes());

    for (const auto& n : pdag.nodes()) {
        int coll_idx = pdag.collapsed_index(n);
        const auto& pa = pdag.parent_set(n);

        incoming_arcs[coll_idx] = pa.size();
    }

    // Create a pseudo topological sort.
    std::vector<std::string> top_sort;
    dynamic_bitset in_top_sort(static_cast<size_t>(pdag.num_nodes()));
    in_top_sort.reset(0, pdag.num_nodes());

    top_sort.reserve(pdag.num_nodes());

    std::vector<int> stack;
    for (auto r : pdag.roots()) {
        stack.push_back(pdag.collapsed_from_index(r));
    }

    while (static_cast<int>(top_sort.size()) != pdag.num_nodes()) {
        // Possible cycle found. This would have not happened in a DAG.
        // Find the next node among the children of the already explored nodes.
        if (stack.empty()) {
            auto min_cardinality = std::numeric_limits<int>::max();
            auto min_cardinality_coll_index = std::numeric_limits<int>::max();
            for (const auto& explored : top_sort) {
                for (auto ch : directed.children_set(explored)) {
                    const auto& ch_name = directed.name(ch);
                    auto ch_coll_this_index = pdag.collapsed_index(ch_name);
                    if (!in_top_sort[ch_coll_this_index] && directed.num_parents(ch) < min_cardinality) {
                        min_cardinality = directed.num_parents(ch);
                        min_cardinality_coll_index = ch_coll_this_index;
                    }
                }
            }

            if (min_cardinality_coll_index == std::numeric_limits<int>::max()) {
                // Find the node with least parents
                for (int i = 0; i < pdag.num_nodes(); ++i) {
                    if (!in_top_sort[i] && directed.num_parents(pdag.collapsed_name(i)) < min_cardinality) {
                        min_cardinality = directed.num_parents(pdag.collapsed_name(i));
                        min_cardinality_coll_index = i;
                    }
                }
            }

            stack.push_back(min_cardinality_coll_index);
            --incoming_arcs[min_cardinality_coll_index];
        }

        int coll_idx = stack.back();
        int idx = pdag.index_from_collapsed(coll_idx);
        stack.pop_back();

        top_sort.push_back(pdag.name(idx));
        in_top_sort.set(coll_idx);

        for (const auto& children : pdag.children_set(idx)) {
            auto coll_ch = pdag.collapsed_from_index(children);
            --incoming_arcs[coll_ch];

            if (in_top_sort[coll_ch] && arc_blackset.count({children, idx}) == 0) {
                const auto& idx_name = pdag.name(idx);
                const auto& children_name = pdag.name(children);
                directed.flip_arc_unsafe(directed.index(idx_name), directed.index(children_name));
            } else if (incoming_arcs[coll_ch] == 0) {
                stack.push_back(coll_ch);
            }
        }
    }

    // directed is DAG now, with topological sort equal to top_sort

    for (const auto& arc : directed.arcs()) {
        spbn->add_arc(arc.first, arc.second);
    }

    std::vector<std::string> explored;

    std::vector<std::shared_ptr<OperatorSet>> ops;
    ops.push_back(std::make_shared<ArcOperatorSet>());
    ops.push_back(std::make_shared<ChangeNodeTypeSet>());

    OperatorPool op_pool(ops);

    learning::scores::CVLikelihood score(df, 4);  // TODO CHANGE ARBITRARY FOR SPBNS

    op_pool.cache_scores(*spbn, score);

    for (auto& top_sort_node : top_sort) {
        select_best_node_type<false>(op_pool, score, spbn, top_sort_node, explored);

        for (const auto& explored_node : explored) {
            if (pdag.has_edge(top_sort_node, explored_node)) {
                const auto arc_destination = select_best_arc_orientation(op_pool, score, spbn, top_sort_node, explored_node);
                select_best_node_type<true>(op_pool, score, spbn, arc_destination, explored);
                pdag.remove_edge(top_sort_node, explored_node);
            }
        }

        explored.push_back(top_sort_node);
    }

    return spbn;
}
}  // namespace models