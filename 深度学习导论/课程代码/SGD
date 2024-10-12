%% main
clear clc
% 设置初始样本数据的矩阵
X = [0,0,1;
     0,1,1;
     1,0,1;
     1,1,1];
% 设置数据正确结果
D = [0,0,1,1];
% 设置初始权重矩阵
W = [0,0,0];     % W = 2*rand(1,3) - 1 

% 训练10000轮得到权重矩阵
for epoch = 1:10000
    W = DeltaSGD(W,X,D);
end

% 计算数据集的结果
result = [0,0,0,0];
for k = 1:4
    x = X(k,:)';
    v = W*x;
    result(k) = Sigmoid(v);
end

%% 函数

% Sigmoid函数
function y = Sigmoid(x)
    y = 1/(1+exp(-x));
end

% DeltaSGD函数
function  W = DeltaSGD(W,X,D)
    alpha = 0.9;                  %定义模型学习率为0.9
    for k = 1:4                   %取4个数据
        x = X(k,:)';              %依次取第k行数据
        v = W*x;                  %计算结果
        y = Sigmoid(v);           %用Sigmoid激活函数
        d = D(k);                 %取结果
        e = d - y;                %计算偏差
        delta = y*(1-y)*e;        %依据Delta规则计算改变量
        W = W + alpha*delta*x';
    end
end
