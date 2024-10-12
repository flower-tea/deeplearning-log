%% main
clear clc
% 初始化
X = [0,0,1;0,1,1;1,0,1;1,1,1];
D = [0,1,1,0];
W1 = 2*rand(4,3)-1;
W2 = 2*rand(1,4)-1;
% 训练
for epoch = 1:10000
    [W1,W2] = BackpropXOR(W1,W2,X,D);
end
% 输出结果
result = [0,0,0,0];
for k = 1:4
    x = X(k,:)';
    v1 = W1*x;
    y1 = arrayfun(@Sigmoid,v1);
    v = W2*y1;
    y = arrayfun(@Sigmoid,v);
    result(k) = y;
end

%% 函数
% BackpropXOR函数
% 输出结果仅有一个输出值且只适用于浅层结构，包含输入层，隐层，输出层。
function [W1,W2] = BackpropXOR(W1,W2,X,D)
    alpha = 0.9;                                    %学习率
    for k = 1:4
        x = X(k,:)';   d = D(k);                    %取对应数据
        v1 = W1*x;     y1 = arrayfun(@Sigmoid,v1);  %第一层结果
        v = W2*y1;     y = arrayfun(@Sigmoid,v);    %第二层结果
        e = d-y;                                    %计算误差
        delta = y.*(1-y).*e;                        %计算输出层δ
        e1 = W2'*delta;                             %δ传递给前一层
        delta1 = y1.*(1-y1).*e1;                    %前一层的δ1
        W1 = W1 + alpha.*delta1*x';                 %更新第一层权值
        W2 = W2 + alpha.*delta*y1';                 %更新第二层权值
    end
end

% Sigmoid函数
function y = Sigmoid(x)
    y = 1/(1+exp(-x));
end

