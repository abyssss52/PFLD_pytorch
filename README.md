# PFLD_pytorch
modified PFLD model which focus on the euler angle by myself based on the PFLD model  

1. 使用 wing loss
2. 没有使用辅助网络，应该数据集的欧拉角没办法保证准确
3. 修改了网络结构
4. 目前采用了mobilenet_v2 x050 ，可改成100或者025
5. 数据集中，只提取了98个点中，检测需要的17个点，14个点用该计算欧拉角，5个点用来人脸对齐，其中两个点相同
