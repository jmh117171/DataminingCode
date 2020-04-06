import numpy as np

class Node(object):
    def __init__(self,condition,index,classType,leafPurity,leafSize):
        self.condition = condition #表示对应的元素
        self.feature_index = index  #表示选择的哪个属性
        self.classType = classType  #表示叶子节点的分类标签
        self.leafPurity = leafPurity    #表示叶子节点的纯度
        self.leafSize = leafSize    #表示叶子节点的大小
        self.left=None #表示左子节点
        self.right=None #表示右子节点
    def __str__(self):
        return "条件："+str(self.condition)+" 属性索引："+str(self.feature_index)+" 属性类别："+str(self.classType)+ \
        " 纯度："+str(self.leafPurity)+" 大小："+str(self.leafSize)#print 一个 Node 类时会打印 __str__ 的返回值

class Tree(object):
    def __init__(self):
        self.root=Node("null","null",'root',"null","null")  #根节点定义为 root 永不删除，作为哨兵使用。
        # self.root = None

    #添加节点，需要给出添加的节点、父节点以及添加的左右位置
    def add(self, addNode,parentNode,leftOrRight):
        node = addNode
        if self.root is None:  # 如果二叉树为空，那么生成的二叉树最终为新插入树的点
            self.root = node
        else:
            if leftOrRight == "left":
                parentNode.left = node
            elif leftOrRight == "right":
                parentNode.right = node
        return node

    # 前序遍历
    def preTraverse(self,root):
        if root is None:
        # if root.left is None or root.right is None:
            return
        print(root)
        self.preTraverse(root.left)
        self.preTraverse(root.right)

    #获取根节点
    def getRoot(self):
        return self.root

#输出二叉树的结构
def outputTree(subTree,str):
    if (subTree is None):
        return
    #如果是根节点，只打印出root
    if (subTree.classType == "root"):
        print(str, subTree.classType, end="")
    #如果是叶子节点，打印出叶子节点的Label、purity和size
    elif (subTree.classType != "null"):
        print(str,subTree.classType,"purity:",subTree.leafPurity,"size:",subTree.leafSize,end="")
    #如果是非叶子节点，打印出分割点条件
    else:
        featureIndex = ""
        if (int(subTree.feature_index) == 0):
            featureIndex = "Sepal.Length"
        elif (int(subTree.feature_index) == 1):
            featureIndex = "Sepal.Width"
        elif (int(subTree.feature_index) == 2):
            featureIndex = "Petal.Length"
        elif (int(subTree.feature_index) == 3):
            featureIndex = "Petal.Width"
        print(str,featureIndex,subTree.condition,end="")
    if (subTree.right is not None):
        print("─┐")
        if (subTree.left is not None):
            outputTree(subTree.right,str+"  | ")
        else:
            outputTree(subTree.right,str+"  ")
    if (subTree.left is not None):
        print()
        print(str,"└─┐")
        outputTree(subTree.left,str+"  ")


#求集合的所有真子集
def PowerSetsBinary(items):
    N = len(items)
    set_all=[]
    for i in range(2**N):
        combo = []
        for j in range(N):
            if(i >> j ) % 2 == 1:
                combo.append(items[j])
        if (combo == []):
            pass
        elif (combo == items):
            pass
        else:
            set_all.append(combo)
    return set_all

#获取数据的某列属性
def getColumn(D,k):
    column = [i[k] for i in D]
    return column

#连续型属性找最佳分割点方法
def evaluateNumericAttribute(D,X):
    midpoints = []      #分割点
    eachLabelCount = []     #每个类别的个数
    uniqueLabel = []    #类别标签

    #连续属性从小到大排序
    X_sorted_index = np.argsort(X)
    X.sort()

    #找到一共有多少类别
    Label = getColumn(D,-1)
    for each in Label:
        if each not in uniqueLabel:
            uniqueLabel.append(each)
    #每个类别计数前的初始化
    for i in range(len(uniqueLabel)):
        eachLabelCount.append(0)
    #类别计数以及添加所有的中值点（分割点）
    for i in range(len(D)-1):
        eachLabelCount[uniqueLabel.index(D[X_sorted_index[i]][-1])] += 1
        if (X[i] != X[i+1]):
            V = ((X[i] + X[i+1]) / 2)
            temp_eachLabelCount = []  # 每个类别的个数
            #遍历当前各类别的计数，并赋值
            for k in range(len(eachLabelCount)):
                temp_eachLabelCount.append(eachLabelCount[k])
            #记录所有的分割点以及分割点左半空间各类别的数量
            midpoints.append([V,temp_eachLabelCount])
    #因为循环n-1次，需要统计最后一个点的类别
    eachLabelCount[uniqueLabel.index(D[X_sorted_index[-1]][-1])] += 1
    # print(X)
    # print(midpoints)
    # print(eachLabelCount)

    #计算数据集D的熵
    Dprob = 0
    for i in range(len(eachLabelCount)):
        prob = eachLabelCount[i] / len(D)
        if prob != 0 :
            Dprob += (prob*np.log2(prob))
    Dprob = 0 - Dprob

    split_point = 0     #用来记录最佳分割点的变量
    score = 0       #用来记录最佳分割点所对应的信息增益值
    #遍历所有的分割点，利用信息增益寻找最佳分割点
    for eachSplitPoint in midpoints:
        DyProb = 0  #左半空间的熵
        DnProb = 0  #右半空间的熵
        for i in range(len(eachLabelCount)):
            Yprob = eachSplitPoint[1][i] / np.sum(eachSplitPoint[1])
            Nprob = (eachLabelCount[i] - eachSplitPoint[1][i]) / (np.sum(eachLabelCount)-np.sum(eachSplitPoint[1]))
            if Yprob != 0:
                DyProb += (Yprob*np.log2(Yprob))
            if Nprob != 0:
                DnProb += (Nprob*np.log2(Nprob))
        DyProb = 0 - DyProb
        DnProb = 0 - DnProb
        #计算信息增益
        temp_score = Dprob - ( ((np.sum(eachSplitPoint[1])/len(D))*DyProb) + (((len(D)-np.sum(eachSplitPoint[1]))/len(D))*DnProb) )
        #判断并记录最佳分割点和信息增益值
        if (temp_score > score):
            score = temp_score
            split_point = eachSplitPoint[0]
        # print(temp_score)
        # print(eachSplitPoint[0])

    print(split_point)
    print(score)
    return split_point,score

def evaluateCategoricalAttribute(D,X):
    midpoints = []      #分割点
    eachLabelCount = []     #每个类别的个数
    uniqueLabel = []    #类别标签
    attributeType = []  #类别属性的种类
    eachAttrEachLabelNumDict = {}

    #找到类别属性的种类有哪些
    for each in X:
        if each not in attributeType:
            attributeType.append(each)
    print(attributeType)

    #找到一共有多少类别
    Label = getColumn(D,-1)
    for each in Label:
        if each not in uniqueLabel:
            uniqueLabel.append(each)
    print(uniqueLabel)

    #初始化每个类别属性中各个标签类别的数量
    for eachAttr in attributeType:
        tempNum = []
        for k in uniqueLabel:
            tempNum.append(0)
        eachAttrEachLabelNumDict[eachAttr] = tempNum
    print(eachAttrEachLabelNumDict)

    #每个类别计数前的初始化
    for i in range(len(uniqueLabel)):
        eachLabelCount.append(0)
    print(eachLabelCount)

    #统计类别属性中的各标签的数量
    for i in range(len(D)):
        eachLabelCount[uniqueLabel.index(D[i][-1])] += 1
        eachAttrEachLabelNumDict[X[i]][uniqueLabel.index(D[i][-1])] += 1
    print(eachAttrEachLabelNumDict)
    print(eachLabelCount)

    temp_midpoints = []     #用来专门判断半空间划分是否有重复的
    #先找到类别型属性集合的所有真子集
    subset_attributeType = PowerSetsBinary(attributeType)
    print(subset_attributeType)
    #找到所有的分割点以及左半空间各类别的数量
    for i in range(len(subset_attributeType)):
        #类别型属性集合的补集（右半空间分割条件）
        temp_set = []
        for k in range(len(attributeType)):
            if (attributeType[k] not in subset_attributeType[i]):
                temp_set.append(attributeType[k])
        #寻找分割方式不重复的真子集分割点
        if ((subset_attributeType[i] not in temp_midpoints) and (temp_set not in temp_midpoints)):
            #选择真子集与其补集中数量小的作为分割点
            if (len(subset_attributeType[i]) < len(temp_set)):
                temp_eachLabelCount = []    #用来统计各类别的数量
                #初始化temo_eachLabelCount
                for uniqueLabelIndex in range(len(uniqueLabel)):
                    temp_eachLabelCount.append(0)
                #统计分割点左半空间各类别的数量
                for key in subset_attributeType[i]:
                    eachCount = eachAttrEachLabelNumDict[key]
                    for eachCountIndex in range(len(eachCount)):
                        temp_eachLabelCount[eachCountIndex] += eachCount[eachCountIndex]
                #添加确定好的分割点
                midpoints.append([subset_attributeType[i],temp_eachLabelCount])
                temp_midpoints.append(subset_attributeType[i])
            else:
                temp_eachLabelCount = []
                for uniqueLabelIndex in range(len(uniqueLabel)):
                    temp_eachLabelCount.append(0)
                for key in temp_set:
                    eachCount = eachAttrEachLabelNumDict[key]
                    for eachCountIndex in range(len(eachCount)):
                        temp_eachLabelCount[eachCountIndex] += eachCount[eachCountIndex]
                midpoints.append([temp_set,temp_eachLabelCount])
                temp_midpoints.append(temp_set)
    print(midpoints)

    # 计算数据集D的熵
    Dprob = 0
    for i in range(len(eachLabelCount)):
        prob = eachLabelCount[i] / len(D)
        if prob != 0 :
            Dprob += (prob*np.log2(prob))
    Dprob = 0 - Dprob

    split_point = 0     #用来记录最佳分割点的变量
    score = 0   #用来记录最佳分割点所对应的信息增益值
    # 遍历所有的分割点，利用信息增益寻找最佳分割点
    for eachSplitPoint in midpoints:
        DyProb = 0  #左半空间的熵
        DnProb = 0  #右半空间的熵
        for i in range(len(eachLabelCount)):
            Yprob = eachSplitPoint[1][i] / np.sum(eachSplitPoint[1])
            Nprob = (eachLabelCount[i] - eachSplitPoint[1][i]) / (np.sum(eachLabelCount)-np.sum(eachSplitPoint[1]))
            if Yprob != 0:
                DyProb += (Yprob*np.log2(Yprob))
            if Nprob != 0:
                DnProb += (Nprob*np.log2(Nprob))
        DyProb = 0 - DyProb
        DnProb = 0 - DnProb
        # 计算信息增益
        temp_score = Dprob - ( ((np.sum(eachSplitPoint[1])/len(D))*DyProb) + (((len(D)-np.sum(eachSplitPoint[1]))/len(D))*DnProb) )
        # 判断并记录最佳分割点和信息增益值
        if (temp_score > score):
            score = temp_score
            split_point = eachSplitPoint[0]
        print(temp_score)
        print(eachSplitPoint[0])

    print(split_point)
    print(score)
    return split_point,score

def DecisionTree(D,maxnPoints,minPurity,tree,root):
    uniqueLabel = []  # 类别标签
    eachLabelCount = []  # 每个类别的个数

    # 找到一共有多少类别
    Label = getColumn(D, -1)
    for each in Label:
        if each not in uniqueLabel:
            uniqueLabel.append(each)

    # 每个类别计数前的初始化
    for i in range(len(uniqueLabel)):
        eachLabelCount.append(0)
    #计算每个类别的个数
    for i in range(len(D)):
        eachLabelCount[uniqueLabel.index(D[i][-1])] += 1
    #计算纯度
    purity = 0
    label_index = 0
    for i in range(len(eachLabelCount)):
        temp_purity = eachLabelCount[i] / len(D)
        if temp_purity > purity:
            purity = temp_purity
            label_index = i

    #满足条件添加叶子节点，结束递归
    if purity >= minPurity or len(D) <= maxnPoints:
        #这里默认叶子节点都插在左边
        # root.classType = uniqueLabel[label_index]
        # root.leafPurity = purity
        # root.leafSize = len(D)
        #因为叶子节点没有分割条件，所以第一个属性为"null"，第二个属性为-1
        leafNode = Node("null",-1,uniqueLabel[label_index],purity,len(D))
        tree.add(leafNode,root,"left")
        return

    split_point = 0     #用来记录所有属性中最好的分割点
    score = 0       #用来记录最好分割点对应的信息增益
    feature_index = 0   #记录最好分割点属性的索引（即第几个属性）
    #遍历所有的属性，寻找最佳分割点
    for i in range(len(D[0])-1):
        #获取某一列属性
        attributeColumn = getColumn(D,i)
        #判断属性是连续型变量
        if isinstance(attributeColumn[0], float):
            temp_split_point,temp_score = evaluateNumericAttribute(D,attributeColumn)
            if temp_score > score:
                score = temp_score
                split_point = temp_split_point
                feature_index = i
        # 判断属性是类别型变量
        elif isinstance(attributeColumn[0], str):
            temp_split_point, temp_score = evaluateCategoricalAttribute(D, attributeColumn)
            if temp_score > score:
                score = temp_score
                split_point = temp_split_point
                feature_index = i
    print("****************")

    #按照条件划分左右半空间
    DY = []
    DN = []
    for i in range(len(D)):
        if D[i][feature_index] <= split_point:
            DY.append(D[i])
        else:
            DN.append(D[i])
    #添加中间节点
    leftNode = Node("<="+str(split_point),feature_index,"null","null","null")
    leftParentNode = tree.add(leftNode,root,"left")
    DecisionTree(DY,maxnPoints,minPurity,tree,leftParentNode)
    rightNode = Node(">"+str(split_point),feature_index,"null","null","null")
    rightParentNode = tree.add(rightNode,root,"right")
    DecisionTree(DN,maxnPoints,minPurity,tree,rightParentNode)

    # print(uniqueLabel)
    # print(eachLabelCount)
    # print(purity)
    # print(label_index)

if __name__=="__main__":
    #获取iris数据
    iris_data = np.loadtxt("iris.txt", str, delimiter=",", usecols=(0, 1, 2, 3, 4))
    D = []
    #将iris数据的前四个维度的变量转换成数值型
    for i in range(len(iris_data)):
        temp_row = [float(iris_data[i][0]),float(iris_data[i][1]),float(iris_data[i][2]),float(iris_data[i][3]),iris_data[i][-1]]
        D.append(temp_row)

    #创建树
    tree = Tree()
    #构建决策树
    DecisionTree(D,5,0.95,tree,tree.getRoot())
    #前序遍历决策树
    tree.preTraverse(tree.getRoot())
    #输出树形结构
    outputTree(tree.getRoot()," ")

    # # 类别型数据找分割点测试
    # for i in range(len(D)):
    #     if (D[i][-1] != '"Iris-setosa"'):
    #         D[i][-1] = "others"
    #
    # X = getColumn(D,0)
    # for i in range(len(X)):
    #     if (4.3 <= float(X[i]) <= 5.2):
    #         X[i] = "Very Short"
    #     elif (5.2 < float(X[i]) <= 6.1):
    #         X[i] = "Short"
    #     elif (6.1 < float(X[i]) <= 7.0):
    #         X[i] = "Long"
    #     elif (7.0 < float(X[i]) <= 7.9):
    #         X[i] = "Very Long"
    # evaluateCategoricalAttribute(D,X)


    # #假设做测试,疑问，ppt里面只考虑了两类
    # X = getColumn(D,0)
    # # print(X)
    # evaluateNumericAttribute(D,X)
    # # print(getColumn(D,4))
    # # print(isinstance(getColumn(D,4)[0],float))
    #
    # # #取某列
    # # print(D[:,4])
    # # print(Label.index('"Iris-versicolor"'))
    # # print(Label)
    #
    # # 二叉树构建测试
    # # tree = Tree()
    # # node1 = Node("1.5",1,"null")
    # # node2 = Node("3.3",2,"null")
    # # node3 = Node("1.2",5,"aaaa")
    # # root = tree.getRoot()
    # # parentNode = tree.add(node1,root,"left")
    # # tree.add(node2,root,"right")
    # # tree.add(node3,parentNode,"left")
    # #
    # # tree.preTraverse(root)