##Djisktra's algorithm

Implement Djisktra's single-source shortest path algorithm (with the simple case where the distance along any edge is assumed to be one).  You can refer to the Wikipedia page on Djikstra's algorithm for reference (https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm), but the basic idea of the algorithm is to keep a priority queue of the current known distances to a node.  We continually pop off the smallest element `i` in the queue, and then update all its successor nodes to have a distance of 1 + distance[i].

For the priority queue, you should use the heapdict library (https://github.com/DanielStutzbach/heapdict) that is included above (and included with Anaconda).  You can update the priority of a heapdict element and pop off the smallest element using the syntax

    d = heapdict({"a":5, "b":6})
    d["b"] = 4
    d.popitem() # -> ("b", 4)
    ...
    
    
The function should return a list both of the shortest distances and the previous nodes in the shortest path from the source node to this node. 

* The distance for unreachable nodes should be inf
* The path for the source node should be None

For instance, executing the algorithm on our graph above gives the following result.
```python
>>> G.shortest_path("A")
({'A': 0, 'B': 1, 'C': 2, 'D': 1, 'E': 2},
 {'A': None, 'B': 'A', 'C': 'B', 'D': 'A', 'E': 'D'})
```

```
import numpy as np
import scipy.sparse as sp
import heapdict

class Graph:
    def __init__(self):
        """ Initialize with an empty edge dictionary. """
        self.edges = {}
    
    def add_edges(self, edges_list):
        """ Add a list of edges to the network. Use 1.0 to indiciate the presence of an edge. 
        
        Args:
            edges_list: list of (a,b) tuples, where a->b is an edge to add
        """
        appearence = {}
        for edge in edges_list:
            key = edge[0]
            value = edge[1]
            appearence[key] = None
            appearence[value] = None
            temp_dict = dict({value: 1.0})
            if self.edges.has_key(key):
                self.edges[key].update(temp_dict)
            else:
                self.edges[key] = temp_dict

        for node in appearence:
            if self.edges.has_key(node):
                continue
            else:
                self.edges[node] = {}
        
        
    def shortest_path(self, source):
        """ Compute the single-source shorting path.
        
        This function uses Djikstra's algorithm to compute the distance from 
        source to all other nodes in the network.
        
        Args:
            source: node index for the source
            
        Returns: tuple: dist, path
            dist: dictionary of node:distance values for each node in the graph, 
                  where distance denotes the shortest path distance from source
            path: dictionary of node:prev_node values, where prev_node indicates
                  the previous node on the path from source to node
        """
        set = heapdict.heapdict({})
        dist = {}
        prev = {}
        for each in self.edges:
            set[each] = float('inf')
            prev[each] = None
        set[source] = 0
        while set:
            least = set.popitem()
            root = least[0]
            distance = least[1]
            dist[root] = distance
            for node in self.edges[root]:
                if node in set and distance + 1 < set[node]:
                    set[node] = distance + 1
                    prev[node] = root
        # print 'Dist:', dist
        # print 'Prev:', prev
        return dist, prev
```
