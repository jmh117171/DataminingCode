import numpy as np
import math

#这是标准化的齐次二次核函数
def kernelFunction1(point1,point2):
    return ((np.dot(point1,point2))**2) / (( ((np.dot(point1,point1))**2) * ((np.dot(point2,point2))**2) )**0.5)

#这是齐次二次核函数
def kernelFunction(point1,point2):
    return ((np.dot(point1,point2))**2)

#用二次齐次核将点映射到高维空间
def featureSpaceFunction(point):
    return [point[0]**2,point[1]**2,point[2]**2,point[3]**2,
                     np.sqrt(2)*point[0]*point[1],np.sqrt(2)*point[0]*point[2],np.sqrt(2)*point[0]*point[3],
                     np.sqrt(2)*point[1]*point[2],np.sqrt(2)*point[1]*point[3],np.sqrt(2)*point[2]*point[3]]

#normalized 方法还较为模糊，再好好看看概念

if __name__=="__main__":
    iris_data = np.loadtxt("../datasets/iris.txt",float,delimiter=",",usecols=(0,1,2,3))

    #Compute the centered and normalized homogeneous quadratic kernel matrix K for the dataset using the kernel function in input space.
    #利用二次齐次核函数计算核矩阵
    kernelMatrix = []
    for i in range(len(iris_data)):
        kernelVector = []
        for j in range(len(iris_data)):
            # print(kernelFunction(centeredIris_data[i][:],centeredIris_data[j][:]))
            kernelVector.append(kernelFunction(iris_data[i][:],iris_data[j][:]))
        kernelMatrix.append(kernelVector)
    kernelMatrix = np.array(kernelMatrix)

    #计算高维空间点中心化后的核矩阵
    # sum_all = 0
    # for i in range(len(kernelMatrix)):
    #     for j in range(len(kernelMatrix[i])):
    #         sum_all += kernelMatrix[i][j]
    # temp_kernelMatrix = []
    # n = len(kernelMatrix)
    # for i in range(len(kernelMatrix)):
    #     temp_kernelVector = []
    #     for j in range(len(kernelMatrix[i])):
    #         sum1 = 0
    #         sum2 = 0
    #         for k in range(len(kernelMatrix[i])):
    #             sum1 += kernelMatrix[i][k]
    #             sum2 += kernelMatrix[j][k]
    #         temp_kernelVector.append(kernelMatrix[i][j] - (1/n)*sum1 - (1/n)*sum2 + (1/(n*n))*sum_all)
    #         # kernelMatrix[i][j] = kernelMatrix[i][j] - (1/n)*sum1 - (1/n)*sum2 + (1/(n*n))*sum_all     #这样写会改变现有的值，影响后面计算
    #     temp_kernelMatrix.append(temp_kernelVector)
    # kernelMatrix = np.array(temp_kernelMatrix)

    # 计算高维空间点中心化后的核矩阵
    #https://zhuanlan.zhihu.com/p/59775730
    n = len(kernelMatrix)
    #temp_matrix初始化，每个位置的值都为1/n
    temp_matrix = []
    for i in range(n):
        temp_vector = []
        for j in range(n):
            temp_vector.append(1/n)
        temp_matrix.append(temp_vector)
    temp_matrix = np.array(temp_matrix)
    #将之前计算的核矩阵中心化
    kernelMatrix = (kernelMatrix - np.matmul(kernelMatrix,temp_matrix) - np.matmul(temp_matrix,kernelMatrix) + np.matmul(np.matmul(temp_matrix,kernelMatrix),temp_matrix))

    # 核矩阵标准化
    kernelMatrix_new = []
    for m in range(len(kernelMatrix)):
        kernelVector_new = []
        for n in range(len(kernelMatrix[m])):
            kernelVector_new.append((kernelMatrix[m][n]) / (math.sqrt(kernelMatrix[m][m]*kernelMatrix[n][n])))
        kernelMatrix_new.append(kernelVector_new)
    kernelMatrix_new = np.array(kernelMatrix_new)
    print("the centered and normalized homogeneous quadratic kernel matrix K :")
    print(kernelMatrix_new)

    #Transform each point x to the feature space ϕ(x), using the homogeneous quadratic kernel. Center these points and normalize them.
    print("Transform each point x to the feature space ϕ(x), using the homogeneous quadratic kernel. Center these points and normalize them :")
    #先将所有的点映射到高维空间
    featureSpaceDataset = []
    for i in range(len(iris_data)):
        # print(featureSpaceFunction(iris_data[i][:]))
        featureSpaceDataset.append(featureSpaceFunction(iris_data[i][:]))
    featureSpaceDataset = np.array(featureSpaceDataset)

    # 高维空间点的中心化（centered point）
    featureSpaceDataset = (featureSpaceDataset - np.mean(featureSpaceDataset, axis=0))
    # 高维空间点的标准化（normalized）
    for i in range(len(featureSpaceDataset)):
        #标准化（向量除以它自己的模长）
        featureSpaceDataset[i][:] = featureSpaceDataset[i][:] / (np.dot(featureSpaceDataset[i][:], featureSpaceDataset[i][:]) ** 0.5)
    print(featureSpaceDataset)

    #Verify that the pair-wise dot products of the centered and normalized points in feature space yield
    # the same kernel matrix computed directly in input space via the kernel function.
    print("Verify if the kernel matrix is the same :")
    #计算中心化和标准化后，任意两点的内积
    kernelMatrix1 = []
    for i in range(len(featureSpaceDataset)):
        dot_list = []
        for j in range(len(featureSpaceDataset)):
            dot_list.append(np.dot(featureSpaceDataset[i][:], featureSpaceDataset[j][:]))
        kernelMatrix1.append(dot_list)
    kernelMatrix1 = np.array(kernelMatrix1)
    print(kernelMatrix1)
