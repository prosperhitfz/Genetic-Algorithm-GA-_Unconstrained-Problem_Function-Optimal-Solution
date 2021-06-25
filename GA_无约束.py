import random
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


population_capacity = 25  # 一个种群内的染色体数目
population = []  # 种群
chromo_num = 0  # 染色体代号
encode_length = 46  # 一个染色体上的二进制基因数（位数）
initial_best_chromo = 0  # 初始化最初代最优个体
best_chromo_num = 0  # 最优染色体代号
best_chromo_code = []  # 最优染色体的基因型（二进制编码）
pc = 0.65  # 交叉概率
pm = 0.1  # 变异概率
epochs = 50000  # 迭代次数


def raw_function_display():  # 绘制要求的无约束优化函数的某个范围的图像做可视化
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.arange(-3, 3, 0.1)
    y = np.arange(-3, 3, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = 5 * (np.exp(-(X - 1)**2 - (Y - 1)**2) - np.exp(-X**2 - Y**2))
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
    ax.set_xlabel('X')
    ax.set_xlim(-3, 3)
    ax.set_ylabel('Y')
    ax.set_ylim(-3, 3)
    ax.set_zlabel('Z')
    ax.set_zlim(-5, 5)
    plt.show()


class GeneticAlgorithm(object):  # 遗传算法
    def __init__(self):
        self.population_capacity = population_capacity
        self.population = population
        self.chromo_number = chromo_num
        self.encode_length = encode_length
        self.initial_best_chromo = initial_best_chromo
        self.best_chromo_number = best_chromo_num
        self.best_chromo_code = best_chromo_code
        self.pc = pc
        self.pm = pm

    def initial_population(self):  # 种群初始化
        for loop in range(self.population_capacity):
            res = ""
            for loop1 in range(self.encode_length):
                if random.random() > 0.5:
                    res += "0"
                else:
                    res += "1"
            self.population.append(res)

    def fitness_function(self, chromo_code):  # 适应度函数计算
        # 设定二进制数前23位为x的二进制字符串，后23位为y的二进制字符串
        a = int(chromo_code[0:23], 2)
        b = int(chromo_code[23:46], 2)
        x = -5.0 + a * (5.0 - (-5.0)) / (pow(2, 23) - 1)  # x的基因
        y = -5.0 + b * (5.0 - (-5.0)) / (pow(2, 23) - 1)  # y的基因
        # 设定适应度函数（由于我要优化的问题是目标函数的最小值，所以直接取目标函数的负值作为适应度函数即可）
        fitness = - (5 * (np.exp(-(x - 1)**2 - (y - 1)**2) - np.exp(-x**2 - y**2)))
        return [x, y, fitness]

    def choose(self):  # 轮盘赌选择
        fitness_values = []  # 所有染色体适应度大小
        p = []  # 各染色体选择概率
        q = []  # 各染色体累计概率
        fitness_sum = 0  # 累计适应值总和
        for loop in range(self.population_capacity):
            fitness_values.append(self.fitness_function(self.population[loop])[2])  # 填入所有染色体适应度
            fitness_sum = fitness_sum + fitness_values[loop]  # 所有染色体适应值总和
            if fitness_values[loop] > self.initial_best_chromo:  # 更新适应度最高的染色体相关信息
                self.initial_best_chromo = fitness_values[loop]  # 最优个体适应度值
                self.best_chromo_number = self.chromo_number  # 最优个体代号
                self.best_chromo_code = self.population[loop]  # 最优个体基因型编码
        for loop in range(self.population_capacity):
            # 计算选择概率
            p.append(fitness_values[loop] / fitness_sum)
            # 计算累计概率
            if loop == 0:
                q.append(p[loop])
            else:
                q.append(q[loop - 1] + p[loop])
        for loop in range(self.population_capacity):  # 产生随机数进行轮盘赌选择，得到新一代种群
            random_num = random.random()
            if random_num <= q[0]:
                self.population[loop] = self.population[0]
            else:
                for loop1 in range(1, self.population_capacity):
                    if random_num < q[loop1]:
                        self.population[loop] = self.population[loop1]

    def cross(self):  # 交叉，交叉概率pc0.65（二进制串直接交叉就可）
        for loop in range(self.population_capacity):
            if random.random() < self.pc:
                cross_bit = int(random.random() * self.encode_length + 1)  # cross_bit位点前后二进制串交叉
                cross_chromo1 = self.population[loop][0:cross_bit] + \
                        self.population[(loop + 1) % self.population_capacity][cross_bit:]
                cross_chromo2 = self.population[(loop + 1) % self.population_capacity][0:cross_bit] + \
                        self.population[loop][cross_bit:]
                # 将交叉后的染色体对应替换掉原来的
                self.population[loop] = cross_chromo1
                self.population[(loop + 1) // self.population_capacity] = cross_chromo2

    def mutate(self):  # 变异，变异概率pm0.1
        p = random.random()
        if p < pm:
            for loop in range(0, 4):
                num = int(random.random() * self.encode_length * self.population_capacity + 1)
                chromosome_num = int(num / self.encode_length) + 1  # 染色体编号
                mutation_bit = num - (chromosome_num - 1) * self.encode_length  # 基因突变位置
                if mutation_bit == 0:
                    mutation_bit = 1
                chromosome_num = chromosome_num - 1
                if chromosome_num >= self.population_capacity:  # 避免发生变异的染色体编号越界（大于种群容量）
                    # 规定当超出时则为种群里面最后那个染色体发生变异
                    chromosome_num = self.population_capacity - 1
                # 变异操作
                if self.population[chromosome_num][mutation_bit - 1] == '0':  # 当变异位点为0时
                    a = '1'
                else:
                    a = '0'
                if mutation_bit == 1:  # 当变异位点在首、中段和尾时的突变情况
                    temp = a + self.population[chromosome_num][mutation_bit:]
                else:
                    if mutation_bit != self.encode_length:
                        temp = self.population[chromosome_num][0:mutation_bit - 1] + a + \
                               self.population[chromosome_num][mutation_bit:]
                    else:
                        temp = self.population[chromosome_num][0:mutation_bit - 1] + a
                self.population[chromosome_num] = temp  # 记录下变异后的染色体


raw_function_display()
GA = GeneticAlgorithm()
GA.initial_population()  # 种群初始化，产生初代种群
total_fitness_value = []  # 总体适应度记录，用于画图
total_epoch = []
for epoch in range(epochs):  # 按epochs代数对GA开始迭代运行
    GA.choose()
    GA.cross()
    GA.mutate()
    GA.chromo_number = epoch
    result = GA.fitness_function(GA.best_chromo_code)
    total_fitness_value.append(-result[2])
    total_epoch.append(epoch)
    if epoch % 100 == 0:
        print("第" + str(epoch) + "代，函数最小值=" + str(-GA.initial_best_chromo))
        print("第" + str(GA.best_chromo_number) + "个染色体:[" + str(GA.best_chromo_code) + "]")
        print("解码后x=" + str(result[0]))
        print("解码后y=" + str(result[1]))
        print('\n')
# 迭代过程可视化
plt.title('遗传算法优化过程')
plt.xlabel('迭代次数')
plt.ylabel('最优值(即最大适应度的相反数)')
plt.plot(total_epoch, total_fitness_value)
plt.show()
