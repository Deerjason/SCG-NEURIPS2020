graphs = ['bitcoin', 'cloister', 'congress', 'epinions', 'highlandtribes']
algs = ['SCG-B', 'SCG-MA', 'SCG-MO', 'SCG-R']

f = open("manual.sh", "w")

for graph in graphs:
    for alg in algs:

        text = """
mv {graph}_{alg}_subgraphs ../ccr_copy/signedNucleus
""".format(graph=graph, alg=alg)

        f.write(text)

text = """
cd ../ccr_copy/signedNucleus
"""
f.write(text)
for graph in graphs:
    for alg in algs:

        text = """
./nucleus ../datasets/{graph}.txt -1 {graph}_{alg}_subgraphs
mv {graph}.txt_{graph}_{alg}_subgraphs_METRICS ../metrics/nuclei/{alg}/{graph}_{alg}_NUCLEI
""".format(graph=graph, alg=alg)

        f.write(text)
text = """
cd ../metrics
"""
f.write(text)

for graph in graphs:
    for alg in algs:

        text = """
python metrics.py ../datasets/{graph}.txt nuclei/{alg}/{graph}_{alg}_NUCLEI metrics/{alg}/{graph}_{alg}_metrics
""".format(graph=graph, alg=alg)

        f.write(text)

f.close()
