from scipy.special import comb
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# ONLY_INTRANIDAL indicates whether or not to display extranidal nodes in the graph.
ONLY_INTRANIDAL = False

# VISCOSITY is the viscosity of blood in Poise.
VISCOSITY = 0.035

# If PREDEFINED_RESISTANCE is True, radius and length data is ignored, and the given resistance values are used; otherwise, radius and length are plugged into the Hagen–Poiseuille equation: resistance = (8 * length * viscosity) / (pi r^4)
PREDEFINED_RESISTANCE = True

# x mmHg = x * MMHG_TO_DYNCM dyn/cm^2.
MMHG_TO_DYN_PER_SQUARE_CM = 1333.22

# ABS_PRESSURE indicates whether or not to display calculated absolute pressures (NON-FUNCTIONAL).
ABS_PRESSURE = False

# vessel has vessel type IDs.
class vessel:
    other = 0
    fistulous = 1
    plexiform = 2
    feeder = 3
    drainer = 4

# LABEL determines the labels. When LABEL is True, the edge labels are displayed. When it is False, the flow and pressure information is displayed.
LABEL = True

def calc_resistance(radius: float, length: float) -> float:
    """Uses the Hagen-Poiseuille equation to calculate resistance: resistance = (8 * length * viscosity) / (pi r^4)
    
    Args:
        radius: The radius of the blood vessel in cm.
        length: The length of the blood vessel in cm.
    
    Returns:
        resistance: The resistance of the blood vessel in dyn * s / cm^5.
    """
    return 8 * length * VISCOSITY / np.pi / (radius ** 4)


def pressure_forward(graph: nx.DiGraph, node, pressure, pressures: dict[str, float]):
    """Steps through the graph to calculate absolute pressure.

    Args:
        graph: The graph.
        node: The name of the original node.
        pressure The pressure of the original node.
        pressures: A dictionary of known node : pressure values.
    """
    nx.set_node_attributes(graph, {node: pressure}, "pressure")
    edges = graph.edges([node])
    nx.set_edge_attributes(graph, {
                           edge: pressure for edge in edges if graph.edges[edge]["start"] == node}, "pressure")
    for edge in edges:
        next_node = edge[1]
        if "pressure" not in graph.nodes[next_node] and graph.edges[edge]["start"] == node:
            next_pressure = pressure + graph.edges[edge]["pressure"] * (-1 if graph.edges[edge]["start"] == node else 1)
            pressure_forward(graph, next_node, next_pressure, pressures)


def calc_pressures(graph: nx.Graph, pressures: dict[str, float]):
    """Begins the calculation of absolute pressure through the graph.

    Args:
        graph: The graph.
        pressures: A dictionary of known node : pressure values.
    """
    pressure_forward(graph, "SP", pressures["SP"], pressures)


def display(graph: nx.Graph, intranidal_nodes: list = [], node_pos={}, title = None, cmap_min = None, cmap_max = None, color_is_flow = True):
    """Displays the graph.

    Args:
        graph: The graph.
        intranidal_nodes: The nodes in the nidus. This is only for color-coding.
        node_pos: A dictionary of node : [x, y] positions.
        title: Optional title for the graph.
        color_is_flow: If `True`, the color will represent flow, and the edge thickness will represent pressure; if `False`, it will be the other way around.
    """
    pos = nx.spring_layout(graph, 1, node_pos, node_pos.keys(), seed = 1)

    # Fix node positions
    xs, ys = [coords[0] for node, coords in pos.items() if node not in node_pos], [
        coords[1] for node, coords in pos.items() if node not in node_pos]
    if xs:
        mx, my = min(xs), min(ys)
        Mx, My = (max(xs) - mx) or 1, (max(ys) - my) or 1
        for node, coords in pos.items():
            if node not in node_pos.keys():
                pos[node] = [0.1 + 0.55 * (coords[0] - mx) / Mx, 1.15 * (coords[1] - my) / My - 0.75]

    # Nodes
    if ABS_PRESSURE:
        node_colors = {
            node: pressure or 0 for node, pressure in graph.nodes("pressure")
        }
    else:
        node_colors = {
            node: "lightgreen" if node in intranidal_nodes else "pink" for node in graph.nodes()
        }
    nx.draw_networkx_nodes(graph, pos, node_color=[node_colors[node] for node in graph.nodes()])
    nx.draw_networkx_labels(graph, pos)

    # Edges
    pressures = [edge[2]["pressure"] for edge in graph.edges(data=True)]
    flows = [edge[2]["flow"] for edge in graph.edges(data=True)]
    min_pressure, max_pressure = min(pressures), max(pressures)
    min_flow, max_flow = min(flows), max(flows)

    if color_is_flow:
        edge_widths = [np.interp(edge[2]["pressure"], [min_pressure, max_pressure], [0.5, 5]) for edge in graph.edges(data=True)]
        edge_colors = [edge[2]["flow"] for edge in graph.edges(data=True)]
    else:
        edge_widths = [np.interp(edge[2]["flow"], [min_flow, max_flow], [0.5, 5]) for edge in graph.edges(data=True)]
        edge_colors = [edge[2]["pressure"] for edge in graph.edges(data=True)]
    nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color=edge_colors, edge_cmap=plt.cm.cool if color_is_flow else plt.cm.Reds, edge_vmin=min(edge_colors) if cmap_min is None else cmap_min, edge_vmax=max(edge_colors) if cmap_max is None else cmap_max)
    if LABEL:
        edge_labels = {(edge[0], edge[1]): edge[2]["label"] for edge in graph.edges(data=True)}
    else:
        edge_labels = {(edge[0], edge[1]): str(round(edge[2]["flow"], 3)) + "\n" + str(round(edge[2]["pressure"] * (-1 if edge[2]["flow"] < 0 else 1), 3)) for edge in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels)
    if color_is_flow:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.cool, norm=plt.Normalize(vmin=min(edge_colors) if cmap_min is None else cmap_min, vmax=max(edge_colors) if cmap_max is None else cmap_max))
        sm.set_array([])
        colorbar = plt.colorbar(sm, ax = plt.gca(), label="Flow (mL/min)")
    else:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=min_pressure if cmap_min is None else cmap_min, vmax=max_pressure if cmap_max is None else cmap_max))
        sm.set_array([])
        colorbar = plt.colorbar(sm, ax = plt.gca(), label="Pressure (mm Hg)")
    # plt.title(title)
    return colorbar

def edges_to_graph(edges: list) -> nx.Graph:
    """Converts a list of edges to a graph in the right format ([starting node, ending node, radius (cm), length (cm), resistance (dyn * s / cm^5), label]).

    Args:
        edges: List of edges.

    Returns:
        graph: NetworkX graph generated from edges.
    """
    graph = nx.Graph()
    graph.add_edges_from([(
        edge[0],
        edge[1],
        {
            "start": edge[0],
            "end": edge[1],
            "radius": edge[2],
            "length": edge[3],
            "resistance": edge[4] if PREDEFINED_RESISTANCE else calc_resistance(edge[2], edge[3]),
            "label": edge[5],
            "type": edge[6] if len(edge) > 6 else vessel.other
        }) for edge in edges
    ])
    return graph

def generate_nidus_2n_connections(graph: nx.Graph, intranidal_nodes: list, plexiform_resistance: float = 81600) -> nx.Graph:
    """Generates a new graph with a random nidus from a given graph and its intranidal nodes by going through each node and connecting it to two other new intranidal nodes.

    Args:
        graph: The original graph.
        intranidal_nodes: List of the nodes between which to generate edges.
        num_expected_edges: Expected number of edges to generate.

    Returns:
        graph: The graph with a random nidus formation.
    """
    num_intranidal_verts = len(intranidal_nodes)
    graph = graph.copy()
    for j in range(num_intranidal_verts - 1):
        for i in range(2):
            unconnected = [node for node in intranidal_nodes if not graph.has_edge(intranidal_nodes[j], node) and node is not intranidal_nodes[j]]
            node = random.choice(unconnected)
            # radius, length = random.normalvariate(0.05, 0.01), random.normalvariate(5, 1)
            radius, length = 0.05, 5
            graph.add_edge(
                intranidal_nodes[j],
                node,
                start=intranidal_nodes[j],
                end=node,
                radius=radius,
                length=length,
                resistance=plexiform_resistance if PREDEFINED_RESISTANCE else calc_resistance(radius, length),
                label="",
                type=vessel.plexiform
            )
    return graph

def generate_nidus_stochastic(graph: nx.Graph, intranidal_nodes: list, sizes: list[int], p: list[list[float]], plexiform_resistance: float = 81600) -> nx.Graph:
    """Generates a new graph with a random nidus from a given graph and its intranidal nodes with a stochastic block model

    Args:
        graph: The original graph.
        intranidal_nodes: List of the nodes between which to generate edges.
        sizes: List of block sizes (see NetworkX's stochastic_block_model).
        p: Stochastic matrix (see NetworkX's stochastic_block_model).

    Returns:
        graph: The graph with a random nidus formation.
    """
    random_graph = nx.stochastic_block_model(sizes, p, intranidal_nodes)
    graph = graph.copy()
    for edge in random_graph.edges:
            radius, length = 0.05, 5
            graph.add_edge(
                edge[0],
                edge[1],
                start=edge[0],
                end=edge[1],
                radius=radius,
                length=length,
                resistance=plexiform_resistance if PREDEFINED_RESISTANCE else calc_resistance(radius, length),
                label="",
                type=vessel.plexiform
            )
    return graph

def generate_nidus_linear(graph: nx.Graph, intranidal_nodes: list, plexiform_resistance: float = 81600) -> nx.Graph:
    """Generates a new graph from a given graph and its intranidal nodes by connecting the intranidal nodes in a single path.

    Args:
        graph: The original graph.
        intranidal_nodes: List of the nodes between which to generate edges.

    Returns:
        graph: The graph with a nidus formation.
    """
    graph = graph.copy()
    for n1, n2 in zip(intranidal_nodes[:-1], intranidal_nodes[1:]):
            radius, length = 0.05, 5
            graph.add_edge(
                n1,
                n2,
                start=n1,
                end=n2,
                radius=radius,
                length=length,
                resistance=plexiform_resistance if PREDEFINED_RESISTANCE else calc_resistance(radius, length),
                label="",
                type=vessel.plexiform
            )
    return graph

def generate_nidus(graph: nx.Graph, intranidal_nodes: list, num_expected_edges: int, plexiform_resistance: float = 81600) -> nx.Graph:
    """Generates a new graph with a random nidus from a given graph and its intranidal nodes.

    Args:
        graph: The original graph.
        intranidal_nodes: List of the nodes between which to generate edges.
        num_expected_edges: Expected number of edges to generate.

    Returns:
        graph: The graph with a random nidus formation.
    """
    num_intranidal_verts = len(intranidal_nodes)
    probability = num_expected_edges / comb(num_intranidal_verts, 2)
    graph = graph.copy()
    for j in range(num_intranidal_verts - 1):
        for k in range(j + 1, num_intranidal_verts):
            if random.random() < probability:
                # radius, length = random.normalvariate(0.05, 0.01), random.normalvariate(5, 1)
                radius, length = 0.05, 5
                graph.add_edge(
                    intranidal_nodes[j],
                    intranidal_nodes[k],
                    start=intranidal_nodes[j],
                    end=intranidal_nodes[k],
                    radius=radius,
                    length=length,
                    resistance=plexiform_resistance if PREDEFINED_RESISTANCE else calc_resistance(radius, length),
                    label="",
                    type=vessel.plexiform
                )
    return graph


def simulate(graph: nx.Graph, intranidal_nodes: list, p_ext: dict[str, float]) -> tuple[np.ndarray, np.ndarray, list[tuple], nx.Graph]:
    """Simulates the nidus.

    Args:
        graph: Graph from `edges_to_graph()` or `generate_nidus()`.
        intranidal_nodes: List of nodes in the nidus to generate edges between.
        p_ext: A dictionary of known node : pressure values in dyn / cm^2.
        num_nidi: Number of nidi to simulate.
        num_expected_edges: Expected number of edges to generate in the nidus.

    Returns:
        flow: The calculated flow values in cm^3/s.
        pressure: The calculated pressure gradients in dyn / cm^2.
        all_edges: The list of edges in the same order as the flow and pressure arrays.
        graph: The directed graph of flow (mL/min) and pressure (mmHg).
    """
    flow, pressure, all_edges = get_Q_and_P(graph, p_ext)
    if ABS_PRESSURE:
        calc_pressures(graph, p_ext)

    pnodes = [node for node in graph.nodes("pressure") if not ONLY_INTRANIDAL or node[0] in intranidal_nodes]
    graph = nx.DiGraph([(
        edge[2]["end" if edge[2]["flow"] < 0 else "start"],
        edge[2]["start" if edge[2]["flow"] < 0 else "end"],
        {
            "radius": edge[2]["radius"],
            "length": edge[2]["length"],
            "resistance": edge[2]["resistance"],
            "label": edge[2]["label"],
            "flow": abs(edge[2]["flow"]) * 60,  # x cm^3/s = 60x mL/min
            "pressure": abs(edge[2]["pressure"]) / MMHG_TO_DYN_PER_SQUARE_CM,
            "type": edge[2]["type"]
        }) for edge in graph.edges(data=True) if not ONLY_INTRANIDAL or (edge[0] in intranidal_nodes and edge[1] in intranidal_nodes)
    ])
    graph.add_nodes_from([(node[0], {"pressure": node[1]}) for node in pnodes])
    return flow, pressure, all_edges, graph

def get_stats(graph: nx.DiGraph):
    """Returns stats for different vessels (count and min/mean/max/total flow/pressure of all/fistulous/plexiform/feeder/drainer vessels) given the `graph` result of `simulate()`.
    
    Args:
        graph: Graph (result of `simulate()`).
    
    Returns:
        stats: A dictionary of stats.
    """
    count = 0
    flow_total, flow_min, flow_max = 0, float('inf'), float('-inf')
    pressure_total, pressure_min, pressure_max = 0, float('inf'), float('-inf')

    fi_count = 0
    fi_flow_total, fi_flow_min, fi_flow_max = 0, flow_min, flow_max
    fi_pressure_total, fi_pressure_min, fi_pressure_max = 0, flow_min, flow_max
    
    pl_count = 0
    pl_flow_total, pl_flow_min, pl_flow_max = 0, flow_min, flow_max
    pl_pressure_total, pl_pressure_min, pl_pressure_max = 0, flow_min, flow_max
    
    fe_count = 0
    fe_flow_total, fe_flow_min, fe_flow_max = 0, flow_min, flow_max
    fe_pressure_total, fe_pressure_min, fe_pressure_max = 0, flow_min, flow_max
    
    dr_count = 0
    dr_flow_total, dr_flow_min, dr_flow_max = 0, flow_min, flow_max
    dr_pressure_total, dr_pressure_min, dr_pressure_max = 0, flow_min, flow_max

    for _, _, attr in graph.edges(data=True):
        count += 1
        flow_total += attr["flow"]
        flow_min = min(flow_min, attr["flow"])
        flow_max = max(flow_max, attr["flow"])
        pressure_total += attr["pressure"]
        pressure_min = min(pressure_min, attr["pressure"])
        pressure_max = max(pressure_max, attr["pressure"])
        if attr["type"] == vessel.fistulous:
            fi_count += 1
            fi_flow_total += attr["flow"]
            fi_flow_min = min(fi_flow_min, attr["flow"])
            fi_flow_max = max(fi_flow_max, attr["flow"])
            fi_pressure_total += attr["pressure"]
            fi_pressure_min = min(fi_pressure_min, attr["pressure"])
            fi_pressure_max = max(fi_pressure_max, attr["pressure"])
        elif attr["type"] == vessel.plexiform:
            pl_count += 1
            pl_flow_total += attr["flow"]
            pl_flow_min = min(pl_flow_min, attr["flow"])
            pl_flow_max = max(pl_flow_max, attr["flow"])
            pl_pressure_total += attr["pressure"]
            pl_pressure_min = min(pl_pressure_min, attr["pressure"])
            pl_pressure_max = max(pl_pressure_max, attr["pressure"])
        elif attr["type"] == vessel.feeder:
            fe_count += 1
            fe_flow_total += attr["flow"]
            fe_flow_min = min(fe_flow_min, attr["flow"])
            fe_flow_max = max(fe_flow_max, attr["flow"])
            fe_pressure_total += attr["pressure"]
            fe_pressure_min = min(fe_pressure_min, attr["pressure"])
            fe_pressure_max = max(fe_pressure_max, attr["pressure"])
        elif attr["type"] == vessel.drainer:
            dr_count += 1
            dr_flow_total += attr["flow"]
            dr_flow_min = min(dr_flow_min, attr["flow"])
            dr_flow_max = max(dr_flow_max, attr["flow"])
            dr_pressure_total += attr["pressure"]
            dr_pressure_min = min(dr_pressure_min, attr["pressure"])
            dr_pressure_max = max(dr_pressure_max, attr["pressure"])
    return {
        "Number of vessels": count,
        "Flow stats (mL/min)": (flow_min, flow_total / count, flow_max) if count else 0,
        "Pressure stats (mmHg)": (pressure_min, pressure_total / count, pressure_max) if count else 0,
        # "spacer1": "",

        "Number of fistulous vessels": fi_count,
        "Fistulous flow stats (mL/min)": (fi_flow_min, fi_flow_total / fi_count, fi_flow_max) if fi_count else 0,
        "Fistulous pressure stats (mmHg)": (fi_pressure_min, fi_pressure_total / fi_count, fi_pressure_max) if fi_count else 0,
        # "spacer2": "",
        
        "Number of plexiform vessels": pl_count,
        "Plexiform flow stats (mL/min)": (pl_flow_min, pl_flow_total / pl_count, pl_flow_max) if pl_count else 0,
        "Plexiform pressure stats (mmHg)": (pl_pressure_min, pl_pressure_total / pl_count, pl_pressure_max) if pl_count else 0,
        # "spacer3": "",

        "Number of feeders": fe_count,
        "Feeder flow stats (mL/min)": (fe_flow_min, dr_flow_total / fe_count, fe_flow_max) if fe_count else 0,
        "Feeder pressure stats (mmHg)": (fe_pressure_min, fe_pressure_total / fe_count, fe_pressure_max) if fe_count else 0,
        "Feeder total flow (mL/min)": fe_flow_total,
        # "spacer4": "",

        "Number of drainers": dr_count,
        "Drainer flow stats (mL/min)": (dr_flow_min, dr_flow_total / dr_count, dr_flow_max) if dr_count else 0,
        "Drainer pressure stats (mmHg)": (dr_pressure_min, dr_pressure_total / dr_count, dr_pressure_max) if dr_count else 0,
        "Drainer total flow (mL/min)": dr_flow_total,
        # "spacer5": "",
    }

def calc_flow(graph: nx.Graph, all_edges, p_ext) -> np.ndarray:
    """Calculates flow and pressure differences for each vessel.

    Args:
        graph: The graph.
        all_edges: A list of (starting node, ending node) tuples, with resistances in dyn * s / cm^5.
        p_ext: A dictionary of known node : pressure values in dyn / cm^2.

    Returns:
        flow: The calculated flow values in cm^3 / s.
    """
    Rv = []
    ΔΔP = []
    num_edges = len(all_edges)

    # First law: sum of flow towards each node is 0
    ΔΔP = [0 for i in range(graph.number_of_nodes())]
    for i, node in enumerate(graph.nodes()):
        edges = graph.edges([node])
        flow = [0 for _ in range(num_edges)]
        for edge in edges:
            if edge in all_edges:
                index = all_edges.index(edge)
                flow[index] = 1
            else:
                index = all_edges.index((edge[1], edge[0]))
                flow[index] = -1
        Rv.append(flow)

    # Second law: total pressure change after traversing any closed loop is 0
    # ==> The only nonzero values of ΔΔP are those where there is an external pressure source
    for cycle in nx.cycle_basis(graph):  # nx.recursive_simple_cycles(graph)
        total_pressure = 0
        flow = [0 for _ in range(num_edges)]
        for i, node1 in enumerate(cycle):
            node2 = cycle[(i + 1) % len(cycle)]
            edge = (node1, node2)
            if edge in all_edges:
                index = all_edges.index(edge)
                flow[index] = -graph.edges[edge]["resistance"]
            else:
                index = all_edges.index((edge[1], edge[0]))
                flow[index] = graph.edges[edge]["resistance"]
            if edge in p_ext:
                total_pressure -= p_ext[edge]
            elif (node2, node1) in p_ext:
                total_pressure += p_ext[(node2, node1)]
        Rv.append(flow)
        ΔΔP.append(total_pressure)

    result = np.linalg.lstsq(np.array(Rv), np.array([ΔΔP]).T, rcond=None)
    # print(f"Error: {result[1]}")
    return result[0].flatten()


def get_Q_and_P(graph: nx.Graph, p_ext) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """Sets flows and pressure differences for each edge in the graph.

    Args:
        graph: The graph.
        p_ext: A dictionary of known node : pressure values in dyn / cm^2.

    Returns:
        flow: The calculated flow values in cm^3 / s
        pressure: The calculated pressure gradients in dyn / cm^2.
        all_edges: A list of the edges in the same order as the flows and pressures.
    """
    all_edges = [(edge[2]["start"], edge[2]["end"]) for edge in graph.edges(data=True)]
    flow = calc_flow(graph, all_edges, p_ext)

    # flow = difference in pressure between ends / resistance
    # Hagen-Poiseuille equation
    pressure = flow * np.array([graph.edges[edge]["resistance"] for edge in all_edges])
    nx.set_edge_attributes(graph, {edge: flow[i] for i, edge in enumerate(all_edges)}, "flow")
    nx.set_edge_attributes(graph, {edge: pressure[i] for i, edge in enumerate(all_edges)}, "pressure")
    return flow, pressure, all_edges
