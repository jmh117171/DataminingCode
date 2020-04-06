import numpy as np
import math
from ast import literal_eval

#定义高斯核函数
def GaussianKernel(Z):
    result = math.exp(-np.dot(Z,Z)/2) / math.pow(2*math.pi,len(Z)/2)
    return result

#寻找密度吸引子
def findAttractor(x,D,h,minRange,sortedKNNIndex):
    # 保存均值漂移过程中，每一步迭代过程中漂移的距离
    dis_list = []

    # 采用均值漂移的方法来寻找
    t = 0   #记录迭代次数
    x_t = x
    x_t_next = np.zeros(len(x_t))
    #只使用最邻近的K个点来进行核密度估计
    GaussianSum = 0
    for index in sortedKNNIndex:
        x_t_next += (GaussianKernel((x_t - D[index]) / h) * D[index])
        GaussianSum += GaussianKernel((x_t - D[index]) / h)
    x_t_next = x_t_next / GaussianSum

    # 计算x_t与x_t_next之间的距离（二范数）
    bias_curr = np.linalg.norm(x_t_next - x_t,ord=2,axis=0)
    dis_list.append(bias_curr)
    t += 1

    # 迭代，直至均值漂移距离小于设定的minRange，结束迭代，找到密度吸引子
    while(np.linalg.norm(x_t_next - x_t,ord=2,axis=0) > minRange):
        x_t = x_t_next
        x_t_next = np.zeros(len(x_t))
        # 只使用最邻近的K个点来进行核密度估计
        GaussianSum = 0
        for index in sortedKNNIndex:
            x_t_next += (GaussianKernel((x_t - D[index]) / h) * D[index])
            GaussianSum += GaussianKernel((x_t - D[index]) / h)
        x_t_next = x_t_next / GaussianSum

        bias_curr = np.linalg.norm(x_t_next - x_t,ord=2,axis=0)
        dis_list.append(bias_curr)
        t += 1
    print("迭代了：",t)
    # 将迭代过程的中漂移的距离颠倒，即列表中的第一个距离为最后一次迭代过程中的漂移距离
    dis_list.reverse()
    # 计算最后两步迭代的距离累加和，用于后续寻找局部最大密度可达
    return_count = 0
    for a in dis_list:
        return_count += a

    return  x_t_next,return_count

def DENCLUE(D,h,minValue,minRange,Label,adjacencyMatrix,K):
    attractor_set = set()   #存放所有的密度吸引子
    attractor_dic = {}      #存放属于密度吸引子的其余点
    attractor_label_dic = {}    #存放点的标签
    point_index = 0     #记录点的索引
    for eachVectorPoint in D:
        sortedKNNIndex = np.array(adjacencyMatrix[point_index]).argsort()[:K]   #获取距离某点从小到大的K个点的索引
        x_attractor,lastTwoStep = findAttractor(eachVectorPoint,D,h,minRange,sortedKNNIndex)
        # x_attractor = np.around(x_attractor,decimals=1)
        print(x_attractor,lastTwoStep)
        # 计算密度吸引子处的核密度估计值
        SumGaussian = 0
        for k in sortedKNNIndex:
            SumGaussian += GaussianKernel((x_attractor - D[k]) / h)
        # 只取大于设定minValue的密度吸引子
        if ((SumGaussian / (len(sortedKNNIndex) * math.pow(h,len(eachVectorPoint)))) >= minValue):
            # 将密度吸引子（np.array）形式转换成字符串，存放在密度吸引子集合中
            attractor_str = ''.join(np.array2string(x_attractor, separator=',').splitlines())
            attractor_set.add(attractor_str)
            # 密度吸引子作为字典的键，将属于同一个密度吸引子的数据存放在一起，作为字典结构中的值，同时将最后两步的距离和也存放进去
            if (attractor_str not in attractor_dic):
                # 存放最后两步迭代的距离累计和
                attractor_dic[attractor_str] = [lastTwoStep]
                # 存放属于该密度吸引子的其余点
                attractor_dic[attractor_str].append(eachVectorPoint)
                # 存放数据的标签，与属于密度吸引子内的点对应
                attractor_label_dic[attractor_str] = [Label[point_index]]
            else:
                attractor_dic[attractor_str].append(eachVectorPoint)
                attractor_label_dic[attractor_str].append(Label[point_index])
                # 因为每个点都会有最后两步迭代累计和，选取最大的最为该密度吸引子最终的累计和
                if (lastTwoStep > attractor_dic[attractor_str][0]):
                    attractor_dic[attractor_str][0] = lastTwoStep
                # attractor_dic[attractor_str][0] += lastTwoStep
        point_index += 1
    print("**********************************************")
    # 寻找局部最大密度可达，合并密度吸引子
    eachCulsterAttractors,clusterResultSet,clusterResultLabel = findLocalMaximum(attractor_set,attractor_dic,attractor_label_dic)
    return eachCulsterAttractors, clusterResultSet, clusterResultLabel

#寻找局部最大密度可达，合并密度吸引子
def findLocalMaximum(attractor_set,attractor_dic,attractor_label_dic):
    cluster_set = []    #用于存放合并在一起的密度吸引子
    # 循环遍历每个密度吸引子
    while attractor_set:
        # 取除一个密度吸引子
        pop_attractor = attractor_set.pop()
        temp_cluster = []   #临时存放可以合并的密度吸引子
        temp_cluster.append(pop_attractor)
        # 遍历剩余的密度吸引子，寻找可以合并的密度吸引子
        for rest_each in attractor_set:
            # 将当前要判断的密度吸引子和已经合并在一起的每个密度吸引子进行判断
            for key in temp_cluster:
                attractor1_array = np.array(literal_eval(key))  #类型转换成np.array
                attractor2_array = np.array(literal_eval(rest_each))
                # 计算两个密度吸引子之间的距离
                attractor_diffValue = np.linalg.norm(attractor1_array - attractor2_array, ord=2, axis=0)
                # 计算两个密度吸引子所包含范围的半径之和
                radius_sum = attractor_dic[key][0] + attractor_dic[rest_each][0]
                # print("diff:",attractor_diffValue)
                # 密度吸引子之间的距离小于半径之和，则两个密度吸引子密度可达，可以合并
                if (attractor_diffValue <= radius_sum):
                    temp_cluster.append(rest_each)
                    # attractor_set.remove(rest_each)
                    break
        # 将已经合并在一起的密度吸引子从集合中删除
        for i in range(1,len(temp_cluster)):
            attractor_set.remove(temp_cluster[i])
        cluster_set.append(temp_cluster)

    cluster_result = []     #存放每个聚类中的所有点
    cluster_label_result = []   #存放每个聚类中所有点对应的标签
    for each_cluster in cluster_set:
        temp_result = []    #临时存放一个聚类中的所有点
        temp_label_result = []      #临时存放一个聚类中所有点对应的标签
        # 遍历合并在一个类簇中的所有密度吸引子
        for each_key in each_cluster:
            # 获取被吸引到同一个密度吸引子的所有点
            points = attractor_dic[each_key][1:]
            # 获取这些点对应的标签
            labels = attractor_label_dic[each_key][:]
            # 将所有点和对应标签加到对应的列表中
            for k in range(len(points)):
                temp_result.append(points[k])
                temp_label_result.append(labels[k])
        # 将一个类簇中的所有点和标签添加到最终结果的集合中
        cluster_result.append(temp_result)
        cluster_label_result.append(temp_label_result)

    # for bbb in cluster_result:
    #     print(bbb)
    #     print(len(bbb))
    #
    # for ccc in cluster_label_result:
    #     print(ccc)
    #     print(len(ccc))
    return cluster_set, cluster_result, cluster_label_result




if __name__=="__main__":
    # 获取iris数据
    iris_data = np.loadtxt("iris.txt",str,delimiter=",",usecols=(0,1,2,3,4))
    D = []      #存放iris四个维度的属性值
    Label = []  #存放iris数据对应的花的种类标签
    for i in range(len(iris_data)):
        D.append(np.array([float(iris_data[i][0]),float(iris_data[i][1]),float(iris_data[i][2]),float(iris_data[i][3])]))
        Label.append(iris_data[i][-1])

    #生成邻接矩阵
    adjacencyMatrix = []
    for i in range(len(D)):
        row_dist = []
        for j in range(len(D)):
            dist = np.linalg.norm(D[i] - D[j], ord=2, axis=0)
            row_dist.append(dist)
        adjacencyMatrix.append(row_dist)
    K = 50  #定义最近邻数K
    # 调用DENCLUE方法进行聚类
    eachCulsterAttractors,clusterResultSet,clusterResultLabel = DENCLUE(D,0.4,0.25,0.0001,Label,adjacencyMatrix,K)

    print("the number of the cluster: ", len(clusterResultSet))
    for i in range(len(clusterResultSet)):
        print("cluster ", i + 1, " size: ", len(clusterResultSet[i]))

    print("each cluster attractors:")
    for i in range(len(eachCulsterAttractors)):
        print("cluster ", i + 1, " attractors:")
        print(eachCulsterAttractors[i])
        print("cluster ", i + 1, " points:")
        print(clusterResultSet[i])

    print("Purity:")
    eachClusterPurity = []
    for each in clusterResultLabel:
        clusterPurity = {}
        for each1 in each:
            if each1 not in clusterPurity:
                clusterPurity[each1] = 1
            else:
                clusterPurity[each1] += 1
        for key in clusterPurity:
            clusterPurity[key] /= len(each)
        eachClusterPurity.append(clusterPurity)

    for i in range(len(eachClusterPurity)):
        print("cluster ", i + 1, " :")
        for key in eachClusterPurity[i]:
            print("label: ", key, " and purity: ", eachClusterPurity[i][key])


