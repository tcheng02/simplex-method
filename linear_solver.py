import numpy as np
from gurobipy import *


class LP_slover:
    def __init__(self, obj, A, b, eq, mtype):
        """
        初始化模型
        :param obj: 目标函数系数矩阵
        :param A: 系数矩阵
        :param b: 资源约束
        :param eq: 约束符号 'eq' 'geq' 'leq'
        :param mtype: min or max
        """
        self.obj = obj
        self.A = A
        self.b = b
        self.eq = eq
        self.mtype = mtype
        self.artificial_index = []
        self.m = A.shape[0]
        self.generate_model()
        self.entering_base_index = []
        self.rtype = 'SM'

    def generate_model(self):
        """
        标准的问题是max问题
        :return: 标准化后的各个参数
        """
        if self.mtype == 'min':
            self.obj *= -1

        if 'geq' in self.eq:
            number_of_geq = self.eq.count('geq')
            A_primes = np.zeros([self.m, number_of_geq])
            location_geq = []
            temp = 0
            for i in range(0, number_of_geq):
                temp = self.eq.index('geq', temp)
                self.eq[temp] = 'eq'
                location_geq.append(temp)
                A_primes[temp, i] = -1
                self.obj = np.append(self.obj, 0)
                temp += 1
            self.obj = self.obj.reshape(1, -1)
            self.A = np.concatenate([self.A, A_primes], axis=1)  # 更新矩阵
        for i in self.eq:
            if i == 'eq':
                # self.obj=np.append(self.obj,(np.max(np.abs(self.obj)) * 10e13))   # 价格系数是M
                self.obj = np.append(self.obj, -10e5)
            elif i == 'leq':
                self.obj = np.append(self.obj, 0)  # 松弛变量没有价格
        self.obj = self.obj.reshape(1, -1)
        if 'eq' in self.eq:
            self.artificial_index = list(np.argwhere(self.obj == -10e5)[:, 1])

        self.m = self.A.shape[0]  # 有多少个约束
        self.n = self.A.shape[1]  # 有多少个变量
        self.init_B = np.eye(self.m)
        self.B_index = [x for x in range(self.n, self.m + self.n)]
        self.AI = np.concatenate([self.A, self.init_B], axis=1)


    def iteration_solve(self, B_):
        """
        计算Revised SM的某一次迭代
        :param B_: 当前的B inverse
        :return:
        """
        print(self.rtype)
        CB = np.array([self.obj[0, x] for x in self.B_index]).reshape(1, -1)  # 基变量价格系数
        # c_x = (self.obj[0, 0:self.n].reshape(1, -1))
        z = -np.matmul(np.matmul(CB, B_), self.AI) + self.obj
        # z_s = np.matmul(CB, B_)
        # z = np.concatenate([z_x, z_s], axis=1)
        RHS = np.matmul(B_, self.b)
        if 0 in RHS:
            self.rtype = 'Bland'
        if np.max(z) > 1e-13:
            # 进基变量
            if self.rtype == 'Bland':
                pivot_col_index = np.argwhere(z > 1e-13)[0][1]  # 如果进基变量重复了，触发Bland Rule 这里还有的会有数值误差
            else:
                pivot_col_index = np.argwhere(z == np.max(z))[0][1]
            # if pivot_col_index in self.entering_base_index:
            #     self.rtype = 'Bland'
            # self.entering_base_index.append(pivot_col_index)

            # 计算theta

            pivot_col = np.matmul(B_, self.AI[:, pivot_col_index].reshape(-1, 1))
            if np.max(pivot_col) <= 0:
                flag = 'Unbounded'
                return RHS, flag
            else:
                pivot_col = np.where(pivot_col < 0, 0, pivot_col)
                theta = RHS / pivot_col
                # 出基变量
                index_list = np.argwhere(theta == np.min(np.min(theta[theta >= 0])))
                if len(index_list) == 1:
                    pivot_row_index = index_list[0][0]  # 这个是列表的索引，不是变量的索引
                else:
                    temp = list(np.sum(index_list, axis=1))
                    pivot_row_index = self.B_index.index(np.min([self.B_index[i] for i in temp]))
                self.B_index[pivot_row_index] = pivot_col_index

                # 计算下一次迭代的B_
                A_new = np.matmul(B_, self.AI)
                B_new_ = np.eye(self.m)
                for i in range(0, self.m):
                    if i == pivot_row_index:
                        B_new_[i, pivot_row_index] = 1 / A_new[pivot_row_index, pivot_col_index]
                    else:
                        B_new_[i, pivot_row_index] = -1 * (
                                A_new[i, pivot_col_index] / A_new[pivot_row_index, pivot_col_index])
                B_new_ = np.matmul(B_new_, B_)  # 下一个迭代的B_inverse
                flag = 'processing'
                return B_new_, flag
        else:
            if len(np.argwhere(z == 0)) > self.m:
                flag = 'Optimal Solution Founded but alternative optimum solution'
            elif 0 in RHS or self.rtype == 'Bland':
                flag = 'Optimal Solution Founded but Degenerate optimum solution'
            else:
                flag = 'Optimal Solution Founded'
            return RHS, flag

    def solve(self):
        """
        外层循环
        :return:
        """
        B_ = self.init_B
        flag = 'processing'
        while flag == 'processing':
            B_, flag = self.iteration_solve(B_)
        # 下面整理结果
        RHS = B_.reshape(-1)
        final_dict = {}
        for i in range(0, len(self.obj[0, :])):
            if i in self.B_index:
                final_dict[i] = RHS[self.B_index.index(i)]
            else:
                final_dict[i] = 0
        if len(self.artificial_index) != 0:
            if np.sum([i in self.B_index for i in self.artificial_index]):
                flag = 'Problem is infeasible'
                return final_dict, None, flag
        if flag == 'Unbounded':
            obj = np.inf
        else:
            x = np.array(list(final_dict.values())).reshape(-1, 1)
            obj = np.matmul(self.obj, x)
        if self.mtype == 'min':
            return final_dict, -1 * obj, flag
        else:
            return final_dict, obj, flag


if __name__ == 'main':
    # 定义模型
    A = np.array([[-1, 2, 3], [12, 5, 10]])
    b = np.array([[20], [90]])
    eq = ['leq', 'leq']
    c_obj = np.array([[-5, 6, 13]])
    mtype = 'max'
    model = LP_slover(c_obj, A, b, eq, mtype)
    model.solve()
    # 测试Big - M法
    A = np.array([[3, 1, 2], [6, 3, 5]])
    b = np.array([[100], [6]])
    eq = ['geq', 'eq']
    c_obj = np.array([[5, 2, 4]])
    mtype = 'min'
    model = LP_slover(c_obj, A, b, eq, mtype)
    model.solve()

    # 测试无穷多最优解
    A = np.array([[-3, 2], [-2, 4]])
    b = np.array([[8], [20]])
    eq = ['leq', 'leq']
    c_obj = np.array([[-2, 4]])
    mtype = 'max'
    model = LP_slover(c_obj, A, b, eq, mtype)
    model.solve()
    # 退化解
    # 1
    A = np.array([[1, 4], [1, 2]])
    b = np.array([[8], [4]])
    eq = ['leq', 'leq']
    c_obj = np.array([[3, 9]])
    mtype = 'max'
    model = LP_slover(c_obj, A, b, eq, mtype)
    model.solve()

    # 退化解
    # 2
    A = np.array([[4, -1], [4, 3], [4, 1]])
    b = np.array([[8], [12], [8]])
    eq = ['leq', 'leq', 'leq']
    c_obj = np.array([[3, 2]])
    mtype = 'max'
    model = LP_slover(c_obj, A, b, eq, mtype)
    model.solve()

    # 退化解
    # 3
    A = np.array([[1 / 4, -8, -1, 9], [1 / 2, -12, -1 / 2, 3], [0, 0, 1, 0]])
    b = np.array([[0], [0], [1]])
    eq = ['leq', 'leq', 'leq']
    c_obj = np.array([[3 / 4, -20, 1 / 2, -6]])
    mtype = 'max'
    model = LP_slover(c_obj, A, b, eq, mtype)
    model.solve()

    A = np.array([[1 / 2, -11 / 2, -5 / 2, 9], [1 / 2, -3 / 2, -1 / 2, 1], [1, 0, 0, 0]])
    b = np.array([[0], [0], [1]])
    eq = ['leq', 'leq', 'leq']
    c_obj = np.array([[10, -57, -9, -24]])
    mtype = 'max'
    model = LP_slover(c_obj, A, b, eq, mtype)
    model.solve()

    # gurobi
    x = list(range(4))
    c_obj = [3 / 4, -20, 1 / 2, -6]
    m = Model("loop")
    x = m.addVars(x, lb=0, vtype=GRB.CONTINUOUS)
    m.update()
    m.setObjective(sum(x[i] * c_obj[i] for i in range(0, len(x))),
                   GRB.MAXIMIZE)
    b = [0, 0, 1]
    m.addConstrs(sum(x[j] * A[i, j] for j in range(0, len(x))) <= b[i] for i in range(0, len(b)))

    m.optimize()

    # 输出结果
    solution = m.getAttr('x', x)
