#include <iostream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <unordered_map>

using namespace std;

class Graph {
private:
    // 使用邻接表存储图结构：节点ID -> 相邻节点列表
    unordered_map<int, vector<int>> adjList;

public:
    // 添加节点
    void addNode(int node) {
        if (adjList.find(node) == adjList.end()) {
            adjList[node] = vector<int>();
        }
    }

    // 添加无向边
    void addEdge(int node1, int node2) {
        addNode(node1);
        addNode(node2);
        adjList[node1].push_back(node2);
        adjList[node2].push_back(node1);
    }

    // 按层打印节点
    void printLevelOrder(int startNode) {
        // 检查起始节点是否存在
        if (adjList.find(startNode) == adjList.end()) {
            cout << "node " << startNode << "not in graph" << endl;
            return;
        }

        unordered_set<int> visited;
        queue<int> q;
        q.push(startNode);
        visited.insert(startNode);

        while (!q.empty()) {
            int levelSize = q.size();
            vector<int> currentLevel;

            // 处理当前层的所有节点
            for (int i = 0; i < levelSize; ++i) {
                int currentNode = q.front();
                q.pop();
                currentLevel.push_back(currentNode);

                // 将所有未访问的邻居加入队列
                for (int neighbor : adjList[currentNode]) {
                    if (visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        q.push(neighbor);
                    }
                }
            }

            // 打印当前层的所有节点
            for (int node : currentLevel) {
                cout << node << " ";
            }
            cout << endl;
        }
    }
};

int main() {
    Graph g;
    g.addEdge(1, 2);
    g.addEdge(1, 3);
    g.addEdge(2, 4);
    g.addEdge(2, 5);
    g.addEdge(3, 6);
    g.addEdge(3, 7);

    cout << "node1 level traversal result:" << endl;
    g.printLevelOrder(1);

    return 0;
}