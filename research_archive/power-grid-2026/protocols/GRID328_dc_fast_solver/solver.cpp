// Batched grounded DC solve with one packed Python-to-native call per observation.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using Posv = void (*)(char*, int*, int*, double*, int*, double*, int*, int*);

enum Status : std::uint8_t {
    solved = 0, unknown_reclose = 1, disconnected_injection = 2,
    disconnected_grid = 3, retry_reference = 4
};

int find_root(std::vector<int>& parent, int node) {
    while (parent[node] != node) {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    return node;
}

extern "C" __declspec(dllexport) void score_batch(
    int n_plans, int n_sub, int n_positions, int n_lines, int slack_position,
    const std::int32_t* plans, const std::uint8_t* unknown,
    const std::int32_t* sub, const std::int32_t* origin_position, const std::int32_t* end_position,
    const double* weight, const double* limit, const double* injection, const double* shunt,
    double* scores, std::uint8_t* status, Posv posv) {
    const int n_bus = 2 * n_sub;
    std::vector<int> node(n_positions), origin(n_lines), end(n_lines);
    std::vector<int> parent(n_bus), index(n_bus);
    std::vector<std::uint8_t> active(n_bus);
    std::vector<double> power(n_bus), rhs(n_bus), angle(n_bus), residual(n_bus);
    std::vector<double> matrix(n_bus * n_bus);
    char lower = 'L';
    int one = 1;

    for (int candidate = 0; candidate < n_plans; ++candidate) {
        if (unknown[candidate]) {
            status[candidate] = unknown_reclose;
            continue;
        }
        const std::int32_t* plan = plans + candidate * n_positions;
        if (plan[slack_position] <= 0) {
            status[candidate] = disconnected_injection;
            continue;
        }
        std::fill(power.begin(), power.end(), 0.0);
        bool detached = false;
        for (int i = 0; i < n_positions; ++i) {
            node[i] = plan[i] > 0 ? sub[i] + n_sub * (plan[i] - 1) : -1;
            if (node[i] >= 0) {
                power[node[i]] += injection[i];
            } else if (injection[i] != 0.0) {
                detached = true;
            }
        }
        if (detached) {
            status[candidate] = disconnected_injection;
            continue;
        }

        const int slack = node[slack_position];
        for (int i = 0; i < n_bus; ++i) {
            parent[i] = i;
            power[i] += shunt[i];
            active[i] = (i == slack || power[i] != 0.0);
        }
        for (int i = 0; i < n_lines; ++i) {
            const int a = node[origin_position[i]], b = node[end_position[i]];
            origin[i] = a;
            end[i] = b;
            if (a >= 0 && b >= 0) {
                active[a] = active[b] = true;
                parent[find_root(parent, b)] = find_root(parent, a);
            }
        }
        const int component = find_root(parent, slack);
        int n = 0;
        bool connected = true;
        for (int i = 0; i < n_bus; ++i) {
            index[i] = -1;
            if (active[i] && i != slack) {
                connected = connected && find_root(parent, i) == component;
                index[i] = n;
                rhs[n++] = power[i];
            }
        }
        if (!connected || n == 0) {
            status[candidate] = disconnected_grid;
            continue;
        }

        std::fill_n(matrix.data(), n * n, 0.0);
        for (int i = 0; i < n_lines; ++i) {
            const int a = origin[i], b = end[i];
            if (a < 0 || b < 0 || a == b) continue;
            const int x = index[a], y = index[b];
            const double w = weight[i];
            if (x >= 0) matrix[x + x * n] += w;
            if (y >= 0) matrix[y + y * n] += w;
            if (x >= 0 && y >= 0) matrix[std::max(x, y) + std::min(x, y) * n] -= w;
        }

        int info = 0;
        posv(&lower, &n, &one, matrix.data(), &n, rhs.data(), &n, &info);
        if (info != 0) {
            status[candidate] = retry_reference;
            continue;
        }
        for (int i = 0; i < n_bus; ++i) {
            angle[i] = index[i] >= 0 ? rhs[index[i]] : 0.0;
            residual[i] = -power[i];
        }
        double score = 0.0;
        for (int i = 0; i < n_lines; ++i) {
            const int a = origin[i], b = end[i];
            if (a < 0 || b < 0) continue;
            const double flow = weight[i] * (angle[a] - angle[b]);
            residual[a] += flow;
            residual[b] -= flow;
            score = std::max(score, std::abs(flow) / limit[i]);
        }
        double max_error = 0.0;
        double max_power = 1.0;
        bool finite = std::isfinite(score);
        for (int i = 0; i < n_bus; ++i) {
            max_power = std::max(max_power, std::abs(power[i]));
            if (i != slack && active[i]) {
                finite = finite && std::isfinite(residual[i]);
                max_error = std::max(max_error, std::abs(residual[i]));
            }
        }
        if (!finite || max_error > 1e-6 * max_power) {
            status[candidate] = retry_reference;
            continue;
        }
        scores[candidate] = score;
        status[candidate] = solved;
    }
}

