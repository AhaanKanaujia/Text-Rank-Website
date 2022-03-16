# Text-Rank-Website

<b>
   Graph Based Ranking Algorithms
</b>

<hr>

Graph based ranking algorithms decide the importance of a vertex within a graph, based on global information drawn from the graph. 
It works on the principal of voting. When a vertex links to another through an edge, it casts a vote for the other. 
The more the number of votes for a vertex, the more the importance of the vertex. 
The importance of the vertex casting the vote also determines the weight given to the vote. 

Formally, let G=(V,E) be a directed graph with set of vertices V and set of edges E, where E is a subset of V×V. 
For a given vertex Vi, let In(Vi) be the set of vertices that point to it (predecessors), and let Out(Vi) be the set of vertices that vertex 
Vi points to (successors). The score of a Vi can be calculated as: 

<p align="center">
  <img width="500" src="page rank formula.png">
</p>

where d is a damping factor, usually set as 0.85, which integrates into the model the probability of jumping from a given vertex to another 
random vertex in the graph. In graph-based approaches, the words present in a piece of text are represented as nodes in a graph and the edges 
connecting these nodes are decided based on a co-occurrence sliding window that traverses the entire document. Edges are added between all 
the nodes present in any particular sliding window and the graph formed is unweighted and undirected in nature. Next, we iterate through the 
graph until convergence and get the most common nodes.

<hr>

<b>
   Text Rank Implementation
</b>

<hr>

Text Rank: In Text Rank, a piece of text is tokenized and annotated with part of speech tags – a preprocessing step that is required to 
enable application of syntactic filters. All lexical units (words) that pass the syntactic filters are added to the graph and an edge is 
added between those lexical units, as nodes, that co-occur within a window of n words, that create an undirected and unweighted graph is 
constructed. Next, a score is calculated for each vertex and the PageRank algorithm is applied for many iterations until it converges. 
Lastly, the vertices are sorted in reverse order of their score and the top T vertices are extracted. 

Analysis of the Results of Text Rank: The text rank algorithm gives results similar to those in the research paper and online implementations 
for window size = 5 and by applying a syntactic filter to only consider nouns and propositions. Moreover, a larger window decreases the accuracy 
of the results since all words in the text receive a higher score, while a smaller window fails to capture the importance of side-by-side words 
in the same context.

Text Rank is efficient for fast and lightweight extraction of keywords. It can be applied on documents, articles, and any piece of text to get 
the underlying keywords of the piece of text that are representative of the document. It is also completely unsupervised and draws information 
only from text itself. However, text rank still cannot achieve the same results as that of supervised models, since a limitation is imposed on 
the number of keywords to be selected, which creates a smaller dataset. Text Rank can be improved by modifying the algorithm to consider the 
position of the words in a piece of text or by considering a set of documents instead of a single one to extract global information. 
