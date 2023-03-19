class Solution:

    def bfs(self, sta):
        for child in self.tree[sta]:
            self.dist[sta, child] = 1
            self.res[1] += 1

            for s in range(1, sta):
                d = self.dist[s, sta] + 1
                self.dist[s, child] = d
                self.res[d] += 1

        for child in self.tree[sta]:
            if child != self.n:
                self.bfs(child)


    def countSubgraphsForEachDiameter(self, n, edges):
        import numpy as np
        self.n = n
        self.dist = np.zeros([n + 1, n + 1], dtype=np.int32)
        self.tree = [[] for i in range(n)]
        for sta, end in edges:
            self.tree[sta].append(end)

        self.res = np.zeros(n + 1, dtype=np.int32)
        self.bfs(1)

        
        
        return list(self.res[1:])
        

if __name__ == "__main__":
    Solution().countSubgraphsForEachDiameter(4, [[1,2],[2,3],[2,4]])
