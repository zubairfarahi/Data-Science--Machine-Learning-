def depthFirstSearch(graph, start):
    stack = [start]

    while(len(stack)):
        curr = stack.pop()
        print(curr)
        for i in graph[curr]:
            stack.append(i)






graph = {
    'a': ['b','c'],
    'b': ['d'],
    'c': ['e'],
    'd': ['f'],
    'e': [],
    'f': []
}

depthFirstSearch(graph, 'a')